"""
File: selayer.py
Project: potluck
Author: malli
Created: 04-10-2024
Description: Squeeze-and-Excitation (SE) layer.
"""
from torch import nn


class SELayer(nn.Module):
    """Squeeze-and-Excitation (SE) layer."""

    def __init__(self, channel, reduction=16):
        """Initialize the SELayer.

        Args:
            channel (int): Number of channels in the input.
            reduction (int, optional): Reduction ratio for the FC layers.
                Defaults to 16.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y