from typing import override

import torch
from torch import Tensor

from deeparguing.criterion.criterion import Criterion
from deeparguing.gradual_aacbr import GradualAACBR


class TransitivityRegulariser(Criterion):

    def __init__(
        self,
    ):
        super().__init__()

    @override
    def forward(
        self, model: GradualAACBR, predictions: Tensor, targets: Tensor
    ) -> Tensor:

        n = len(model.X_train)
        W = model.casebase_edge_weights(model.X_train, model.X_train).reshape(n, n)
        Wij = W.unsqueeze(2)  # (n, n, 1)
        Wjk = W.unsqueeze(0)  # (1, n, n)
        Wik = W.unsqueeze(1)  # (n, 1, n)

        violations = torch.relu(Wij * Wjk - Wik)
        result = violations.mean()

        return result

    @override
    def step(self, model: GradualAACBR) -> bool:
        return True

    @override
    def reset(self) -> None:
        pass
