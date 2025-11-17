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

        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_channels)

    @override
    def forward(self, case: Tensor) -> Tensor:
        # if case.ndim == 3:
        #     B, H, W = case.shape
        #     case = case.unsqueeze(1)

        if case.ndim == 4:
            B, H, W, C = case.shape
            case = case.reshape(B, C, H, W)
        x = case
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()

    @override
    def get_output_features(self) -> int:
        return self.fc2.out_features

    @override
    def plot_parameters(self):
        print("CNN PARAMETERS ARE NOT PRINTED")
        pass
