from abc import ABCMeta, abstractmethod
from typing import Callable, override

import torch
from torch import Tensor

type PartialOrderType = ComputePartialOrder | Callable[[Tensor, Tensor], Tensor]


class ComputePartialOrder(torch.nn.Module, metaclass=ABCMeta):

    @abstractmethod
    @override
    def forward(self, attacker: Tensor, target: Tensor) -> Tensor:
        pass

    @abstractmethod
    def plot_parameters(self):
        pass


class CompareCases(torch.nn.Module, metaclass=ABCMeta):

    @abstractmethod
    @override
    def forward(self, attacker: Tensor, target: Tensor) -> Tensor:
        pass


class Subtractor(CompareCases):

    def __init__(
        self,
        temperature: float = 1.0,
        activation: Callable[[Tensor], Tensor] = lambda x: x,
    ):
        super(Subtractor, self).__init__()
        self.temperature = temperature
        self.activation = activation

    @override
    def forward(self, attacker: Tensor, target: Tensor) -> Tensor:
        return self.activation((attacker - target) / self.temperature)
