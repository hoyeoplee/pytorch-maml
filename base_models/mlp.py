import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_block(in_dim, out_dim):
    seq = nn.Sequential(
        nn.Linear(in_dim, out_dim, bias=True),
        nn.ReLU(True),
    )
    return seq


class mlp(nn.Module):
    def __init__(self, in_dim, hid_dim1, hid_dim2, hid_dim3, hid_dim4, out_dim):
        super(mlp, self).__init__()
        self.in_dim = in_dim
        self.linear4block = nn.Sequential(
            linear_block(in_dim, hid_dim1),
            linear_block(hid_dim1, hid_dim2),
            linear_block(hid_dim2, hid_dim3),
            linear_block(hid_dim3, hid_dim4),
        )
        self.linear_out = nn.Linear(hid_dim4, out_dim)

    def forward(self, x):
        x = self.linear4block(x.view(-1, self.in_dim))
        x = self.linear_out(x)
        return x

