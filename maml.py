from copy import deepcopy
from collections import OrderedDict

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class maml(nn.Module):
    def __init__(self, n, k, model, loss, num_inner_loop, inner_lr, outer_lr, use_cuda):
        super(maml, self).__init__()
        self.n = n
        self.k = k
        self.model = model
        self.loss = loss
        self.num_inner_loop = num_inner_loop
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.use_cuda = use_cuda

        self.weight_name = [name for name, _ in self.model.named_parameters()]
        self.weight_len = len(self.weight_name)
        self.initialize_parameters()

        self.meta_optim = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.outer_lr
        )

        if use_cuda:
            self.cuda()

    def add_default_weights(self, weights):
        # Due to batch normalization.
        for i, name in enumerate(self.weight_for_default_names):
            weights[name] = self.weight_for_default[i]
        return weights

    def load_weights(self):
        tmp = deepcopy(self.model.state_dict())
        weights = []
        for name, value in tmp.items():
            if name in self.weight_name:
                weights.append(value)
        return weights

    def forward(self, support_x, support_y, query_x, num_inner_loop=None, cum=False):
        if num_inner_loop is None:
            num_inner_loop = self.num_inner_loop
        pred_ys = []
        for idx in range(num_inner_loop):
            if idx > 0:
                self.model.load_state_dict(self.updated_state_dict)
            weight_for_autograd = self.load_weights()
            pred_ys.append(self.model(query_x))
            support_y_pred = self.model(support_x)
            loss_for_local_update = self.loss(support_y_pred, support_y)
            grad = torch.autograd.grad(
                loss_for_local_update, 
                self.model.parameters(),
                create_graph=True
            )
            for w_idx in range(self.weight_len):
                self.updated_state_dict[self.weight_name[w_idx]] = weight_for_autograd[w_idx] - self.inner_lr * grad[w_idx]
        self.model.load_state_dict(self.updated_state_dict)
        query_y_pred = self.model(query_x)
        pred_ys.append(query_y_pred)
        self.model.load_state_dict(self.keep_weight)
        if cum:
            return pred_ys
        else:
            return query_y_pred

    def store_state(self):
        self.keep_weight = deepcopy(self.model.state_dict())

    def initialize_parameters(self):
        self.store_state()
        self.weight_for_default = torch.nn.ParameterList([])
        self.weight_for_default_names = []
        for name, value in self.keep_weight.items():
            if not name in self.weight_name:
                self.weight_for_default_names.append(name)
                self.weight_for_default.append(
                    torch.nn.Parameter(value.to(dtype=torch.float))
                )
        self.free_state()

    def free_state(self):
        self.updated_state_dict = OrderedDict()
        self.updated_state_dict = self.add_default_weights(self.updated_state_dict)

    def fit(self, tr_dataset, num_batch):
        for epoch in range(10 * 4):
            db = DataLoader(tr_dataset, num_batch, shuffle=True, num_workers=1, pin_memory=True)
            for step, (x_spt, y_spt, x_qry, y_qry) in tqdm(enumerate(db)):
                if self.use_cuda:
                    x_spt, y_spt = x_spt.cuda(), y_spt.cuda()
                    x_qry, y_qry = x_qry.cuda(), y_qry.cuda()
                loss = 0
                for i in range(num_batch):
                    pred_query_y = self(x_spt[i], y_spt[i], x_qry[i])
                    loss += self.loss(pred_query_y, y_qry[i])
                loss /= num_batch
                self.meta_optim.zero_grad()
                loss.backward()
                self.meta_optim.step()
                self.store_state()

    def prediction_acc(self, ts_dataset, num_inner_loop_test):
        db_test = DataLoader(ts_dataset, 1, shuffle=True, num_workers=1, pin_memory=True)
        correct = 0
        total = 0
        for x_spt, y_spt, x_qry, y_qry in db_test:
            if self.use_cuda:
                x_spt, y_spt = x_spt.cuda(), y_spt.cuda()
                x_qry, y_qry = x_qry.cuda(), y_qry.cuda()
            pred_query_y = self(x_spt[0], y_spt[0], x_qry[0], num_inner_loop_test)
            correct += torch.eq(pred_query_y.argmax(dim=1), y_qry[0]).sum().item()
            total += len(y_qry[0])
        return correct / total
