from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from deeparguing.feature_extractor.feature_extractor import FeatureExtractor


class SimpleCNN(FeatureExtractor):

    # Code adpated from https://github.com/pytorch/examples/blob/main/mnist/main.py

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super(SimpleCNN, self).__init__(no_features=out_channels)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=32 * 2, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32 * 2, out_channels=64 * 2, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64 * 2, out_channels=64 * 2, kernel_size=3, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=64 * 2, out_channels=128 * 2, kernel_size=3, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=128 * 2, out_channels=128 * 2, kernel_size=3, padding=1
        )
        self.conv6 = nn.Conv2d(
            in_channels=128 * 2, out_channels=128 * 2, kernel_size=3, padding=1
        )
        self.conv7 = nn.Conv2d(
            in_channels=128 * 2, out_channels=256 * 2, kernel_size=3, padding=1
        )
        self.conv8 = nn.Conv2d(
            in_channels=256 * 2, out_channels=256 * 2, kernel_size=3, padding=1
        )
        self.conv9 = nn.Conv2d(
            in_channels=256 * 2, out_channels=256 * 2, kernel_size=3, padding=1
        )

        self.bn1 = nn.BatchNorm2d(32 * 2)
        self.bn2 = nn.BatchNorm2d(128 * 2)
        self.bn3 = nn.BatchNorm2d(256 * 2)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(0.2)

        self.fc1 = nn.Linear(4096 * 2, 4096 * 2)
        self.fc2 = nn.Linear(4096 * 2, 2048 * 2)
        self.fc3 = nn.Linear(2048 * 2, out_channels)
        self.relu = nn.ReLU()

    @override
    def forward(self, case: Tensor) -> Tensor:
        # if case.ndim == 3:
        #     B, H, W = case.shape
        #     case = case.unsqueeze(1)

        if case.ndim == 4:
            B, H, W, C = case.shape
            case = case.reshape(B, C, H, W)
        x = case
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)

        x = self.relu(self.bn2(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.relu(self.bn3(self.conv7(x)))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze()

    @override
    def get_output_features(self) -> int:
        return self.fc2.out_features

    @override
    def plot_parameters(self):
        print("CNN PARAMETERS ARE NOT PRINTED")
        pass
