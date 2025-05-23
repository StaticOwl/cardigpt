import logging

import torch

from .basicblock import BasicBlock
from .bottleneck import BottleNeck
from .seresnet1d import ResNet
from .seresnet_transformers import ResNet_Transformer
from .seresnet_lstm import ResNet_LSTM
from .util import resnet_model_urls

logger = logging.getLogger(__name__)
__all__ = ['ResNet', 'resnet', 'resnet18', 'resnet34']

# prev_model = torch.load('./model_repo/23-14-0.5524.pth', weights_only=True)
# prev_model = torch.load('./model_repo/54-81-0.5748.pth', weights_only=True)
# prev_model = torch.load('./model_repo/61-20-0.5913.pth', weights_only=True)
# prev_model = torch.load('./model_repo/63-16-0.6015.pth', weights_only=True)
# prev_model = torch.load('./model_repo/64-8-0.6121.pth', weights_only=True)
# prev_model = torch.load('./model_repo/66-19-0.6148.pth', weights_only=True)
# prev_model = torch.load('./model_repo/67-25-0.6176.pth', weights_only=True)
prev_model = torch.load('./model_repo/90-0.6222.pth', weights_only=True)


def resnet(pretrained=False, **kwargs):
    model = ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], **kwargs)
    return model

def resnet_tr(pretrained=False, **kwargs):
    model = ResNet_Transformer(block=BasicBlock, num_blocks=[1, 0, 0, 1], **kwargs)
    if pretrained:
        model.load_state_dict(prev_model, strict=False)
    return model

def resnet_lstm(pretrained=False, **kwargs):
    model = ResNet_LSTM(block=BasicBlock, num_blocks=[3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(prev_model, strict=False)
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

