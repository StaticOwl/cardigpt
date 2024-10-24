"""
File: proto_loss.py
Project: potluck
Author: malli
Created: 12-10-2024
Description: This module defines the prototype loss function used to ensure
             that prototype vectors are sufficiently distinct from each other.
"""

import torch
import torch.nn.functional as F

def proto_loss(prototype, eps=1e-6):
    """
    Calculate the prototype loss, encouraging prototype vectors to be
    well-separated in the feature space.

    Args:
        prototype (torch.Tensor): A tensor of prototype vectors.
        eps (float): A small epsilon value to prevent division by zero.

    Returns:
        torch.Tensor: The calculated prototype loss.
    """
    pairwise_distance = F.pdist(prototype, p=2)
    if pairwise_distance.size(0) < 2:
        return torch.tensor(0.0, requires_grad=True).to(prototype.device)
    min_dist, _ = torch.min(pairwise_distance.view(-1, 1), dim=0)
    avg_min_dist = torch.mean(min_dist)
    loss = 1.0 / (torch.log(1.0 / avg_min_dist + eps) + eps)
    return loss