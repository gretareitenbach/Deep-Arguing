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
        noise_value: float = 0
    ):
        super(SoftCoordinateDominance, self).__init__()
        self.temperature = temperature
        self.use_noise = noise_value != 0
        self.distribution = torch.distributions.gumbel.Gumbel(torch.tensor([0.0]), torch.tensor([noise_value])) if self.use_noise else None

    @override
    def forward(self, attacker: Tensor, target: Tensor) -> Tensor:

        d = attacker.shape[-1]

        diff = attacker - target

        if self.use_noise:
            assert self.distribution
            noise = self.distribution.sample(sample_shape = diff.shape[:-1]).to(device=attacker.device)
            # noise = torch.rand_like(diff) - 0.25
        else:
            noise = torch.zeros_like(diff, device=attacker.device)

        # print(torch.min(diff))
        # print(torch.max(diff))

        result = torch.relu(
            (torch.sigmoid((diff * self.temperature) + noise) * 2 - 1).sum(dim=-1) / d
        )

        # print(torch.min(result))
        # print(torch.max(result))
        #
        # exit()

        return result.unsqueeze(-1)
