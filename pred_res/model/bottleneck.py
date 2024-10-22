"""
File: bottleneck.py
Project: potluck
Author: malli
Created: 04-10-2024
Description: Experimental ResNet bottleneck block
"""
from torch import nn

from .selayer import SELayer
from .util import conv1x1, conv7x1


class BottleNeck(nn.Module):
    """Experimental ResNet bottleneck block.

    The block is composed of two convolutional layers with a ReLU activation
    function, followed by a convolutional layer with a linear activation
    function. The output of the block is the sum of the input and the output of
    the last convolutional layer.

    Attributes:
        expansion (int): The expansion factor for the number of channels in the
            output of the block. Defaults to 4.
        conv1 (nn.Conv1d): The first convolutional layer.
        bn1 (nn.BatchNorm1d): The first batch normalization layer.
        conv2 (nn.Conv1d): The second convolutional layer.
        bn2 (nn.BatchNorm1d): The second batch normalization layer.
        conv3 (nn.Conv1d): The third convolutional layer.
        bn3 (nn.BatchNorm1d): The third batch normalization layer.
        relu (nn.ReLU): The ReLU activation function.
        downsample (nn.Module or None): The downsample module. Defaults to None.
        stride (int): The stride of the convolutional layers. Defaults to 1.
        se (SELayer): The squeeze-and-excitation layer.
        dropout (nn.Dropout): The dropout layer.
    """
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = conv1x1(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.conv2 = conv7x1(out_planes, out_planes, stride=stride)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.conv3 = conv1x1(out_planes, out_planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(out_planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.se = SELayer(out_planes * self.expansion)
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        """Forward pass of the ResNet bottleneck block.

        Args:
            x (torch.Tensor): The input to the block.

        Returns:
            torch.Tensor: The output of the block.
        """
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