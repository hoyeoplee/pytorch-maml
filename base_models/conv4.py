import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels, use_dropout):
    seq = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_channels, momentum=1, affine=True),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    if use_dropout:
        list_seq = list(seq.modules())[1:]
        list_seq.append(nn.Dropout(0.1))
        seq = nn.Sequential(*list_seq)
    return seq


class conv4(nn.Module):
    def __init__(self, image_size, num_channels, num_classes, hidden_dim, use_dropout):
        super(conv4, self).__init__()
        self.conv4block = nn.Sequential(
            conv_block(num_channels, hidden_dim, use_dropout),
            conv_block(hidden_dim, hidden_dim, use_dropout),
            conv_block(hidden_dim, hidden_dim, use_dropout),
            conv_block(hidden_dim, hidden_dim, use_dropout),
        )
        finalSize = int(math.floor(image_size / (2*2*2*2)))  # four max_poolings
        self.outSize = finalSize * finalSize * hidden_dim
        self.linear = nn.Linear(self.outSize, num_classes)
 
    def forward(self, x):
        x = self.conv4block(x)
        x = x.view(-1, self.outSize)
        x = self.linear(x)
        return x
