import os
import torch
import torch.nn as nn

from dragen_imagenet import MiniImagenet
from base_models.conv4 import conv4
from maml import maml


if __name__ == "__main__":
    ##############
    ## TRAINING ##
    ##############
    # Set parameters.
    n, k = 5, 1
    num_inner_loop = 5
    num_inner_loop_test = 10
    inner_lr = 1e-2
    outer_lr = 1e-4
    num_batch = 4  # 2
    max_iter = 60000
    use_cuda = True

    # Define model. You can use any neural network-based model.
    model = conv4(image_size=84, num_channels=3, num_classes=n,
                  hidden_dim=32, use_dropout=False)
    # Define loss function.
    loss_f = torch.nn.functional.cross_entropy
    # Define MAML.
    maml_model = maml(n, k, model, loss_f, num_inner_loop, inner_lr, outer_lr, use_cuda)
    # Load training dataset.
    tr_dataset = MiniImagenet(batchsz=max_iter // 10)
    # Fit the model according to the given dataset.
    maml_model.fit(tr_dataset, num_batch)

    ##########
    ## TEST ##
    ##########
    # Load test dataset.
    ts_dataset = MiniImagenet(batchsz=600, mode="test")
    maml_model.eval()
    # Predict and calculate accuracy.
    acc = maml_model.prediction_acc(ts_dataset, num_inner_loop_test)
