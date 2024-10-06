"""
File: util.py
Project: potluck
Author: malli
Created: 04-10-2024
Description: write_a_description
"""

import torch.nn as nn

resnet_model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv1x1(in_planes, out_planes, strides=1, padding=0, bias=False):
    """
    1x1 convolution with padding to maintain the length

    Args:
        in_planes (int): number of input channels
        out_planes (int): number of output channels
        strides (int, optional): stride of the convolutional kernel. Defaults to 1.
        padding (int, optional): padding of the convolutional kernel. Defaults to 0.
        bias (bool, optional): whether to use bias. Defaults to False.

    Returns:
        nn.Conv1d: the convolutional layer
    """
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=strides, padding=padding, bias=bias)


def conv3x1relu(in_planes, out_planes, strides=1, padding=1, bias=False):
    """
    3x1 convolution with padding to maintain the length and LeakyReLU activation

    Args:
        in_planes (int): number of input channels
        out_planes (int): number of output channels
        strides (int, optional): stride of the convolutional kernel. Defaults to 1.
        padding (int, optional): padding of the convolutional kernel. Defaults to 1.
        bias (bool, optional): whether to use bias. Defaults to False.

    Returns:
        nn.Sequential: the convolutional layer
    """
    conv3x1same = nn.Sequential(
        nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=strides, padding=padding, bias=bias),
        nn.LeakyReLU(0.2, inplace=True))
    return conv3x1same


def conv7x1(in_planes, out_planes, stride=1):
    """
    7x1 convolution with padding to maintain the length

    Args:
        in_planes (int): number of input channels
        out_planes (int): number of output channels
        stride (int, optional): stride of the convolutional kernel. Defaults to 1.

    Returns:
        nn.Conv1d: the convolutional layer
    """
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)


def conv24x1relu(in_planes, out_planes, strides=2, padding=11, bias=False):
    """
    24x1 convolution maintains the length / 2 with LeakyReLU activation

    Args:
        in_planes (int): number of input channels
        out_planes (int): number of output channels
        strides (int, optional): stride of the convolutional kernel. Defaults to 2.
        padding (int, optional): padding of the convolutional kernel. Defaults to 11.
        bias (bool, optional): whether to use bias. Defaults to False.

    Returns:
        nn.Sequential: the convolutional layer
    """
    conv24x1same = nn.Sequential(
        nn.Conv1d(in_planes, out_planes, kernel_size=24, stride=strides, padding=padding, bias=bias),
        nn.LeakyReLU(0.2, inplace=True))
    return conv24x1same


def conv48x1relu(in_planes, out_planes, strides=2, padding=23, bias=False):
    """
    48x1 convolution with padding to maintain the length / 2 with LeakyReLU activation

    Args:
        in_planes (int): number of input channels
        out_planes (int): number of output channels
        strides (int, optional): stride of the convolutional kernel. Defaults to 2.
        padding (int, optional): padding of the convolutional kernel. Defaults to 23.
        bias (bool, optional): whether to use bias. Defaults to False.

    Returns:
        nn.Sequential: the convolutional layer
    """
    conv24x1same = nn.Sequential(
        nn.Conv1d(in_planes, out_planes, kernel_size=48, stride=strides, padding=padding, bias=bias),
        nn.LeakyReLU(0.2, inplace=True))
    return conv24x1same
