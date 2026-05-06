from typing import Optional, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from deeparguing.feature_extractor.feature_extractor import FeatureExtractor


class LargeCNN(FeatureExtractor):

    def __init__(
        self,
        in_channels: int = 1,
        output_features: int = 1,
        dropout: float = 0.2,
        weights_path: Optional[str] = None,
        freeze_weights: bool = False,
    ):
        super(LargeCNN, self).__init__(no_features=output_features)

        # ----------------------
        # Convolutional feature extractor
        # ----------------------
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(256)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(dropout)

        # ----------------------
        # Final conv before GAP
        # ----------------------
        self.conv_final = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn_final = nn.BatchNorm2d(512)

        # ----------------------
        # Global Average Pooling + Linear Projection
        # ----------------------
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_out = nn.Linear(512, output_features)

        # ----------------------
        # Load & freeze
        # ----------------------
        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

        self.freeze_weights = freeze_weights

        if freeze_weights:
            for p in self.parameters():
                p.requires_grad = False
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    @override
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_weights:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    @override
    def forward(self, case: Tensor) -> Tensor:

        if case.ndim == 4 and case.shape[-1] in [1, 3]:
            case = case.permute(0, 3, 1, 2).contiguous()

        x = case

        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)

        # Block 2
        x = F.relu(self.bn2(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.maxpool(x)
        x = self.dropout(x)

        # Final conv block
        x = F.relu(self.bn_final(self.conv_final(x)))
        x = self.dropout(x)

        # Global Average Pooling → [B, 512]
        x = self.gap(x)
        x = torch.flatten(x, 1)

        # Linear projection
        x = self.fc_out(x)

        return x.squeeze()

    @override
    def get_output_features(self) -> int:
        return self.fc_out.out_features

    @override
    def plot_parameters(self):
        print("CNN PARAMETERS ARE NOT PRINTED")
        pass


class SimpleCNN(FeatureExtractor):

    def __init__(
        self,
        in_channels: int = 1,
        output_features: int = 1,
        dropout: float = 0.0,
        weights_path: Optional[str] = None,
        freeze_weights: bool = False,
        freeze_bn: bool = False,
    ):
        super(SimpleCNN, self).__init__(no_features=output_features)

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(dropout)

        # ----------------------
        # Global Average Pooling + Linear Projection
        # ----------------------
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_out = nn.Linear(64, output_features)

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

        self.freeze_weights = freeze_weights

        if freeze_weights:
            for p in self.parameters():
                p.requires_grad = False
        self.freeze_bn = freeze_bn
        if freeze_weights or freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    @override
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_weights or self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    @override
    def forward(self, case: Tensor) -> Tensor:
        if case.ndim == 4 and case.shape[-1] in [1, 3]:
            case = case.permute(0, 3, 1, 2).contiguous()

        x = case

        # Block 1
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        # Global Average Pooling → [B, 64]
        x = self.gap(x)
        x = torch.flatten(x, 1)

        # Linear projection
        x = self.fc_out(x)

        return x.squeeze()

    @override
    def get_output_features(self) -> int:
        return self.fc_out.out_features

    @override
    def plot_parameters(self):
        print("CNN PARAMETERS ARE NOT PRINTED")
        pass
