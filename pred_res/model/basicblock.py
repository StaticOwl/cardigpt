"""
File: basicblock.py
Project: potluck
Author: malli
Created: 04-10-2024
Description: Basic building block for ResNet models.
"""
from torch import nn

from pred_res.model.selayer import SELayer
from pred_res.model.util import conv7x1


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv7x1(in_planes, out_planes, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.conv2 = conv7x1(out_planes, out_planes, stride=1)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.se = SELayer(out_planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out