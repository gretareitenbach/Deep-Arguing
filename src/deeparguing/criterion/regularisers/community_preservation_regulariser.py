from typing import override

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR
from deeparguing.criterion.criterion import Criterion
from deeparguing.criterion.regularisers.utils import FilterFunc


class CommunityPreservationRegulariser(Criterion):

    def __init__(self, filter_func: FilterFunc = lambda A: A, method: str = "svd"):
        super().__init__()
        self.filter_func = filter_func
        methods = ["svd", "nuc", "fro"]
        if method not in methods:
            raise ValueError(
                f"Unknown community preservation method: {method}. Please select from {methods}."
            )
        if method == "svd":
            self.method: FilterFunc = lambda A: torch.sum(torch.svd(A).S)
        if method == "nuc":
            self.method: FilterFunc = lambda A: torch.linalg.norm(A, ord="nuc")
        if method == "fro":
            self.method: FilterFunc = lambda A: torch.norm(A, p="fro")

    @override
    def forward(self, model: GradualAACBR, predictions: Tensor, targets: Tensor) -> Tensor:
        assert model.A is not None
        A = self.filter_func(model.A)
        A = torch.abs(A)
        return self.method(A)

    @override
    def step(self, model: GradualAACBR) -> bool:
        return True

    @override
    def reset(self) -> None:
        pass
