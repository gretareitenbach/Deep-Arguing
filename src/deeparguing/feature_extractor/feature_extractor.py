from typing import override
import torch
from torch import Tensor
from abc import abstractmethod, ABCMeta

class FeatureExtractor(torch.nn.Module, metaclass=ABCMeta):

    def __init__(self, no_features: int):
        super(FeatureExtractor, self).__init__()
        self.no_features = no_features

    @abstractmethod
    @override
    def forward(self, case: Tensor) -> Tensor:
        pass

    @abstractmethod
    def get_output_features(self) -> int:
        pass

    @abstractmethod
    def plot_parameters(self):
        pass
