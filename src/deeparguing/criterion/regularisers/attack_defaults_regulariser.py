from typing import override

import torch
from torch import Tensor

from deeparguing.criterion.criterion import Criterion
from deeparguing.gradual_aacbr import GradualAACBR


class AttackDefaultsRegulariser(Criterion):

    def __init__(
        self,
    ):
        super().__init__()

    @override
    def forward(self, model: GradualAACBR, predictions: Tensor, targets: Tensor) -> Tensor:

        mse = torch.nn.MSELoss()
        out = model.casebase_edge_weights(model.X_cases, model.X_default)
        result = mse(out, torch.ones_like(out))
        out = model.casebase_edge_weights(model.X_default, model.X_cases)
        result += mse(out, torch.zeros_like(out))

        return result

    @override
    def step(self, model: GradualAACBR) -> bool:
        return True

    @override
    def reset(self) -> None:
        pass
