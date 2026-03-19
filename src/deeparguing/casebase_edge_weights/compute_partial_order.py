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
        

class SoftCoordinateDominance(CompareCases):

    def __init__(
        self,
        temperature: float = 1.0,
    ):
        super(SoftCoordinateDominance, self).__init__()
        self.temperature = temperature

    @override
    def forward(self, attacker: Tensor, target: Tensor) -> Tensor:

        d = attacker.shape[-1]

        result = torch.relu(
            (torch.sigmoid((attacker - target) * self.temperature) * 2 - 1).sum(dim=-1) / d
        )

        return result.unsqueeze(-1)
