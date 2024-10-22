"""
File: freeze.py
Project: potluck
Author: malli
Created: 06-10-2024
Description: This module provides functions to set the freeze status of model parameters by layer and update learning rates for specific layers.
"""

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


def set_freeze_by_id(model, layer_num_last):
    """
    Set the requires_grad attribute of model parameters to False up to a specified layer.
    
    Args:
        model (nn.Module): The model whose parameters need to be frozen.
        layer_num_last (int): The number of layers from the end to keep trainable.
    """
    for param in model.parameters():
        param.requires_grad = False
    child_list = list(model.children())[-layer_num_last:]
    if not isinstance(child_list, Iterable):
        child_list = list(child_list)
    for child in child_list:
        for param in child.parameters():
            param.requires_grad = True


def set_lr_by_id(model, layer_num_last):
    """
    Set the requires_grad attribute of parameters in specific layers to True.
    
    Args:
        model (nn.Module): The model whose parameters' learning rates need to be updated.
        layer_num_last (int): The number of layers from the end to update learning rates.
    """
    child_list = list(model.children())[-layer_num_last:]
    if not isinstance(child_list, Iterable):
        child_list = list(child_list)
    for child in child_list:
        for param in child.parameters():
            param.requires_grad = True
