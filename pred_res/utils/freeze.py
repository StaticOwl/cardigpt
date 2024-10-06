"""
File: freeze.py
Project: potluck
Author: malli
Created: 06-10-2024
Description: write_a_description
"""

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


def set_freeze_by_id(model, layer_num_last):
    for param in model.parameters():
        param.requires_grad = False
    child_list = list(model.children())[-layer_num_last:]
    if not isinstance(child_list, Iterable):
        child_list = list(child_list)
    for child in child_list:
        for param in child.parameters():
            param.requires_grad = True


def set_lr_by_id(model, layer_num_last):
    child_list = list(model.children())[-layer_num_last:]
    if not isinstance(child_list, Iterable):
        child_list = list(child_list)
    for child in child_list:
        for param in child.parameters():
            param.requires_grad = True
