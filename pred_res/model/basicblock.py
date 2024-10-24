"""
File: basicblock.py
Project: potluck
Author: malli
Created: 04-10-2024
Description: Basic building block for ResNet models with improved descriptions and docstrings.
"""
import torch
from torch import nn

from .selayer import SELayer
from .util import conv7x1


class BasicBlock(nn.Module):
    """
    Basic building block for ResNet models.

    Attributes:
        expansion (int): The expansion factor for the block.
    """

    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        """
        Initialize a BasicBlock.

        Args:
            in_planes (int): Number of input channels.
            out_planes (int): Number of output channels.
            stride (int, optional): Stride of the first convolutional layer. Defaults to 1.
            downsample (nn.Module, optional): Downsample layer to match the shape of the residual. Defaults to None.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv7x1(in_planes, out_planes, stride=stride)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv7x1(out_planes, out_planes, stride=1)
        self.se = SELayer(out_planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_planes * 2, out_planes, 1)
        )

    def forward(self, x):
        """
        Forward pass through the BasicBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        residual = x

        out1 = self.conv1(x)
        out1 = self.bn(out1)
        out1 = self.relu(out1)

        out2 = self.conv1(x)
        out2 = self.bn(out2)
        out2 = self.relu(out2)

        attention = torch.sigmoid(self.fc(torch.cat((out1, out2), dim=1)))

        out = attention * out1 + (1 - attention) * out2

        out = self.dropout(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out
