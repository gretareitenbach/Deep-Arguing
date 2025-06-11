from abc import ABCMeta, abstractmethod
from typing import Callable, override

import torch
from torch import Tensor

type IrrelevanceType = ComputeIrrelevance | Callable[[Tensor, Tensor], Tensor]


class ComputeIrrelevance(torch.nn.Module, metaclass=ABCMeta):

    @abstractmethod
    @override
    def forward(self, new_cases: Tensor, X_train: Tensor) -> Tensor:
        pass

    @abstractmethod
    def plot_parameters(self):
        pass
