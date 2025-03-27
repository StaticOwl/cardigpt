"""
File: seresnet_transformers.py
Author: Koustav Mallick
Date: 15/03/2025

Description: [Add a brief description of the file here]
"""
import logging

import torch
import torch.nn as nn
from .basicblock import BasicBlock
from transformers import BertModel, BertConfig


from .util import conv1x1

logger = logging.getLogger(__name__)


class ResNet_Transformer(nn.Module):
    def __init__(self, block, num_blocks, in_channel=1, out_channel=10, num_prototypes=10, prototype=False, ze=False):
        super(ResNet_Transformer, self).__init__()

        self.in_planes = 64
        self.d_model = 512
        self.num_layers = 6
        self.num_heads = 8

        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(0.2)

        self.layers = nn.ModuleList([
            self._make_layer(block, 64, num_blocks[0]),
            self._make_layer(block, 128, num_blocks[1], stride=2),
            self._make_layer(block, 256, num_blocks[2], stride=2),
            self._make_layer(block, 512, num_blocks[3], stride=2)
        ])

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.small_fc = nn.Linear(5, 10)

        self.bert_config = BertConfig(
            hidden_size=self.d_model,
            num_attention_heads=self.num_heads,
            num_hidden_layers=self.num_layers,
            intermediate_size=self.d_model * 4,
            max_position_embeddings=1024,
            vocab_size=1,
            output_hidden_states=False
        )

        self.bert = BertModel(self.bert_config)


        self.fc = nn.Linear(512 * block.expansion + 10, out_channel)

        self.sig = nn.Sigmoid()

        self._init_weights(ze)

    def _load_transformer_weights(self):
        """
        Load weights for the transformer model.
        """

        weights = self.transformer.encoder.layers.state_dict()
        

        self.transformer_encoder.load_state_dict(weights)

    def _init_weights(self, ze):
        """
        Initialize weights and biases for layers in the network.

        :param ze: whether to use zero initializations for the last BN in each
            residual branch
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if ze:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        """
        Make a layer of the ResNet.

        :param block: block type (BasicBlock or BottleNeck)
        :param planes: number of output channels
        :param num_blocks: number of blocks in the layer
        :param stride: stride of the first block in the layer
        :return: a layer of the ResNet
        """
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, strides=stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, ag):
        """
        Forward pass of the ResNet.

        :param x: input tensor
        :return: output tensor
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        ag = self.small_fc(ag)

        x = x.unsqueeze(1) 
        x = x.permute(0, 1, 2)

        transformer_output = self.bert(inputs_embeds=x).last_hidden_state

        transformer_output = transformer_output.mean(dim=1)
        x = self.dropout(transformer_output)

        x = torch.cat((ag, x), dim=1)
            
        x = self.fc(x)

        return x
