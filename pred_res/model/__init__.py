import logging

import torch

from .basicblock import BasicBlock
from .bottleneck import BottleNeck
from .seresnet1d import ResNet
from .util import resnet_model_urls

logger = logging.getLogger(__name__)
__all__ = ['ResNet', 'resnet', 'resnet18', 'resnet34']

# prev_model = torch.load('./model_repo/23-14-0.5524.pth', weights_only=True)
# prev_model = torch.load('./model_repo/54-81-0.5748.pth', weights_only=True)
prev_model = torch.load('./model_repo/61-20-0.5913.pth', weights_only=True)


def resnet(pretrained=False, **kwargs):
    model = ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], **kwargs)
    return model


def resnet_prototypes(pretrained=False, **kwargs):
    model = ResNet(block=BasicBlock, num_blocks=[3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(prev_model, strict=False)
    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    logger.info('loading resnet18')
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(prev_model, strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(prev_model, strict=False)
    return model

