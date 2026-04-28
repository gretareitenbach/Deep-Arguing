from typing import override

import torch
import torch.nn.functional as F
from torch import Tensor

from deeparguing.criterion.criterion import Criterion
from deeparguing.gradual_aacbr import GradualAACBR


class ShiftLoss(Criterion):

    @override
    def forward(
        self, model: GradualAACBR, predictions: Tensor, targets: Tensor
    ) -> Tensor:

        # predictions: (B, n)
        # targets: (B, n) one-hot

        top2_vals, _ = torch.topk(predictions, k=2, dim=1)
        top2_mean = top2_vals.mean(dim=1, keepdim=True)

        shifted_predictions = predictions - top2_mean

        return F.binary_cross_entropy_with_logits(shifted_predictions, targets)
