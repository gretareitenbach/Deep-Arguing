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
        noise_value = 0
    ):
        super(SoftCoordinateDominance, self).__init__()
        self.temperature = temperature
        self.use_noise = noise_value != 0
        self.distribution = torch.distributions.gumbel.Gumbel(torch.tensor([0.0]), torch.tensor([noise_value]))

    @override
    def forward(self, attacker: Tensor, target: Tensor) -> Tensor:

        d = attacker.shape[-1]

        if self.use_noise:
            noise = self.distribution.sample().to(device=attacker.device)
        else:
            noise = torch.zeros_like(attacker, device=attacker.device)

        result = torch.relu(
            (torch.sigmoid((attacker - target + noise) * self.temperature) * 2 - 1).sum(dim=-1) / d
        )

        return result.unsqueeze(-1)
