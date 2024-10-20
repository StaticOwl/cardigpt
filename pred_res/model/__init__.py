import logging

import torch
from torch.utils import model_zoo

from .basicblock import BasicBlock
from .bottleneck import BottleNeck
from .seresnet1d import ResNet
from .util import resnet_model_urls

logger = logging.getLogger(__name__)
__all__ = ['ResNet', 'resnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def resnet(pretrained=False, **kwargs):
    model = ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], **kwargs)
    return model


def resnet_prototypes(pretrained=False, **kwargs):
    model = ResNet(block=BasicBlock, num_blocks=[2,2,2,2], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load('./model_repo/prev_model.pth', weights_only=True), strict=False)
    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    logger.info('loading resnet18')
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(resnet_model_urls['resnet18']), strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(resnet_model_urls['resnet34']), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BottleNeck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(resnet_model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BottleNeck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(resnet_model_urls['resnet101']), strict=False)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BottleNeck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(resnet_model_urls['resnet152']), strict=False)
    return model
