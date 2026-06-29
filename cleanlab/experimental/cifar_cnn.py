# Copyright (C) 2017-2023  Cleanlab Inc.
# This file is part of cleanlab.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with cleanlab.  If not, see <https://www.gnu.org/licenses/>.

"""
A PyTorch CNN which can be used for finding label issues in CIFAR-10 and CleanLearning with co-teaching.

Code adapted from: https://github.com/bhanML/Co-teaching/blob/master/model.py

You must have PyTorch installed: https://pytorch.org/get-started/locally/
"""


import torch.nn as nn
import torch.nn.functional as F


def call_bn(bn, x):
    return bn(x)


class ConvBNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.leaky_relu(call_bn(self.bn, self.conv(x)), negative_slope=0.01)


class CNN(nn.Module):
    """A CNN architecture shown to be a good baseline for a CIFAR-10 benchmark.

    Parameters
    ----------
    input_channel : int
    n_outputs : int
    dropout_rate : float
    top_bn : bool

    Methods
    -------
    forward
      forward pass in PyTorch"""

    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, top_bn=False):
        super(CNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        self.stage1 = nn.ModuleList(
            [
                ConvBNBlock(input_channel, 128, kernel_size=3, padding=1),
                ConvBNBlock(128, 128, kernel_size=3, padding=1),
                ConvBNBlock(128, 128, kernel_size=3, padding=1),
            ]
        )
        self.stage2 = nn.ModuleList(
            [
                ConvBNBlock(128, 256, kernel_size=3, padding=1),
                ConvBNBlock(256, 256, kernel_size=3, padding=1),
                ConvBNBlock(256, 256, kernel_size=3, padding=1),
            ]
        )
        self.stage3 = nn.ModuleList(
            [
                ConvBNBlock(256, 512, kernel_size=3, padding=0),
                ConvBNBlock(512, 256, kernel_size=3, padding=0),
                ConvBNBlock(256, 128, kernel_size=3, padding=0),
            ]
        )
        self.l_c1 = nn.Linear(128, n_outputs)
        self.top_bn_layer = nn.BatchNorm1d(n_outputs) if top_bn else None

    def forward(
        self,
        x,
    ):
        h = x
        for layer in self.stage1:
            h = layer(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        for layer in self.stage2:
            h = layer(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        for layer in self.stage3:
            h = layer(h)
        h = F.avg_pool2d(h, kernel_size=h.shape[2])

        h = h.view(h.size(0), h.size(1))
        logit = self.l_c1(h)
        if self.top_bn_layer is not None:
            logit = call_bn(self.top_bn_layer, logit)
        return logit
