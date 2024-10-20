"""
File: bottleneck.py
Project: potluck
Author: malli
Created: 04-10-2024
Description: ResNet bottleneck block
"""
from torch import nn

from .selayer import SELayer
from .util import conv1x1, conv7x1


class BottleNeck(nn.Module):
    """ResNet bottleneck block."""
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = conv7x1(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.conv2 = conv7x1(out_planes, out_planes, stride=stride)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.conv3 = conv7x1(out_planes, out_planes * self.expansion, stride=1)
        self.bn3 = nn.BatchNorm1d(out_planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.se = SELayer(out_planes * self.expansion)
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out