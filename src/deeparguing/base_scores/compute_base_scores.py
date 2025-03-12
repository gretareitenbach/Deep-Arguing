import torch
from abc import abstractmethod, ABCMeta
from typing import Callable

type BaseScoreType = ComputeBaseScores | Callable[[torch.Tensor], torch.Tensor]

class ComputeBaseScores(torch.nn.Module, metaclass=ABCMeta):


    @abstractmethod
    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def plot_parameters(self):
        pass
