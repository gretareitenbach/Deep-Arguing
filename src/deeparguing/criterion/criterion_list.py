from typing import Tuple, override

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR
from deeparguing.criterion.criterion import Criterion


class CriterionList(Criterion):

    def __init__(self, criteria: list[Tuple[Criterion, float]] = []):
        super().__init__()
        self.criteria = criteria

    @override
    def forward(self, model: GradualAACBR, predictions: Tensor, targets: Tensor) -> Tensor:

        # Determine the correct device
        device = predictions.device if predictions is not None else (model.device if model is not None else torch.device("cpu"))
        total = torch.zeros(1, device=device).squeeze()

        for criterion_func, weight in self.criteria:
            if weight == 0:
                continue
            total += weight * criterion_func(model, predictions, targets)

        return total

    @override
    def step(self, model: GradualAACBR) -> bool:
        """
        Call step() on all child criteria.
        Returns True only if ALL children have converged.
        """
        all_converged = True
        for criterion, _ in self.criteria:
            converged = criterion.step(model)
            all_converged = all_converged and converged
        return all_converged

    @override
    def reset(self) -> None:
        """Reset all child criteria."""
        for criterion, _ in self.criteria:
            criterion.reset()
