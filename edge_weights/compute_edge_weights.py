import torch
from abc import abstractmethod, ABCMeta
from typing import Any

class ComputeEdgeWeights(torch.nn.Module, metaclass=ABCMeta):


    @abstractmethod
    def forward(self, A: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def plot_parameters(self):
        pass