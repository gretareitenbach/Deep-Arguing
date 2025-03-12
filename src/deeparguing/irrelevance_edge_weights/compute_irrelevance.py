from typing import Callable
import torch
from abc import abstractmethod, ABCMeta

type IrrelevanceType = ComputeIrrelevance | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

class ComputeIrrelevance(torch.nn.Module, metaclass=ABCMeta):


    @abstractmethod
    def forward(self, new_cases: torch.Tensor, X_train: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def plot_parameters(self):
        pass
