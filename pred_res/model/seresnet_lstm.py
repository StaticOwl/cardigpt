
import logging
import torch
import torch.nn as nn
from .basicblock import BasicBlock
from .seresnet1d import ResNet
from .util import conv1x1

logger = logging.getLogger(__name__)

class ResNet_LSTM(nn.Module):
    def __init__(self, block, num_blocks, in_channel=1, out_channel=10, num_prototypes=10, prototype=False, ze=False):
        super(ResNet_LSTM, self).__init__()

        self.in_planes = 64
        self.d_model = 512
        self.hidden_size = 256
        self.num_layers = 2

        # ResNet backbone for feature extraction
        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList([
            self._make_layer(block, 64, num_blocks[0]),
            self._make_layer(block, 128, num_blocks[1], stride=2),
            self._make_layer(block, 256, num_blocks[2], stride=2),
            self._make_layer(block, 512, num_blocks[3], stride=2)
        ])

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.small_fc = nn.Linear(5, 10)

        self.fc = nn.Linear(512 * block.expansion + 10, self.d_model)

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=self.d_model, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

        # Final classification layer
        self.classifier = nn.Linear(self.hidden_size + 10, out_channel)

        self.sig = nn.Sigmoid()

        self._init_weights(ze)

    def _init_weights(self, ze):
        """
        Initialize weights and biases for layers in the network.
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
        Forward pass of the ResNet + LSTM model.
        :param x: input tensor (ECG data)
        :param ag: additional features tensor
        :return: output tensor (logits)
        """
        # Feature extraction from ResNet layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)

        # Apply global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        ag = self.small_fc(ag)

        x = x.unsqueeze(1)

        lstm_out, _ = self.lstm(x)

        x = lstm_out[:, -1, :]

        x = torch.cat((ag, x), dim=1)

        x = self.classifier(x)

        return x
