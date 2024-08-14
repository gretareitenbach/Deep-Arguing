
import torch
from abc import abstractmethod, ABCMeta
from typing import Any

class ComputePartialOrder(torch.nn.Module, metaclass=ABCMeta):


    @abstractmethod
    def forward(self, attacker: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def plot_parameters(self):
        pass