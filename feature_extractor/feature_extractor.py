import torch
from abc import abstractmethod, ABCMeta
from typing import Any

class FeatureExtractor(torch.nn.Module, metaclass=ABCMeta):


    @abstractmethod
    def forward(self, case: torch.tensor) -> torch.tensor:
        pass

    @abstractmethod
    def get_output_features(self) -> int:
        pass

    @abstractmethod
    def plot_parameters(self):
        pass