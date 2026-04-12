from typing import override

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR
from deeparguing.criterion.criterion import Criterion
from deeparguing.criterion.regularisers.utils import FilterFunc


class ConnectivityRegulariser(Criterion):

    def __init__(self, filter_func: FilterFunc = lambda A: A, epsilon: float = 1e-8):
        super().__init__()
        self.filter_func = filter_func
        self.epsilon = epsilon

    @override
    def forward(self, model: GradualAACBR, predictions: Tensor, targets: Tensor) -> Tensor:
        assert model.A is not None
        A = self.filter_func(model.A)
        A = torch.abs(A)
        A = torch.sum(A, dim=1) + self.epsilon
        result = -torch.sum(torch.log(A))
        result = result / len(model.A)
        return result

    @override
    def step(self, model: GradualAACBR) -> bool:
        return True

    @override
    def reset(self) -> None:
        pass
