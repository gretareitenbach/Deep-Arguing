from typing import Tuple, override

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR
from deeparguing.regularisers.regulariser import Regulariser


class RegulariserList(Regulariser):

    def __init__(self, regularisers: list[Tuple[Regulariser, float]]):
        super().__init__()
        self.regularisers = regularisers

    @override
    def forward(self, model: GradualAACBR) -> Tensor:
        assert model.A is not None

        total = torch.zeros(1, device=model.device).squeeze()

        for reg_func, weight in self.regularisers:
            total += weight * reg_func(model)

        return total

    @override
    def step(self, model: GradualAACBR) -> bool:
        """
        Call step() on all child regularisers.
        Returns True only if ALL children have converged.
        """
        all_converged = True
        for reg, _ in self.regularisers:
            converged = reg.step(model)
            all_converged = all_converged and converged
        return all_converged

    @override
    def reset(self) -> None:
        """Reset all child regularisers."""
        for reg, _ in self.regularisers:
            reg.reset()
