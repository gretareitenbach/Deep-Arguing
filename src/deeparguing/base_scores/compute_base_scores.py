from abc import ABCMeta, abstractmethod
from typing import Callable, override

import torch
from torch import Tensor

type BaseScoreType = ComputeBaseScores | Callable[[Tensor], Tensor]


class ComputeBaseScores(torch.nn.Module, metaclass=ABCMeta):

    @abstractmethod
    @override
    def forward(self, nodes: Tensor) -> Tensor:
        pass

    @abstractmethod
    def plot_parameters(self):
        pass
