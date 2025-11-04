from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from deeparguing.feature_extractor.feature_extractor import FeatureExtractor


class SimpleCNN(FeatureExtractor):

    # Code adpated from https://github.com/pytorch/examples/blob/main/mnist/main.py

    def __init__(self, in_channels: int = 1):
        super(SimpleCNN, self).__init__(no_features=10)
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 128)
        # self.fc2 = nn.Linear(128, 10)
        self.fc2 = nn.Linear(128, 1)

    @override
    def forward(self, case: Tensor) -> Tensor:
        # if case.ndim == 3:
        #     B, H, W = case.shape
        #     case = case.unsqueeze(1)

        if case.ndim == 4:
            B, H, W, C = case.shape
            case = case.reshape(B, C, H, W)

        x = self.conv1(case)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    @override
    def get_output_features(self) -> int:
        return self.fc2.out_features

    @override
    def plot_parameters(self):
        print("CNN PARAMETERS ARE NOT PRINTED")
        pass
