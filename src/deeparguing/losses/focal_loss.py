from typing import override

import torch
import torch.nn.functional as F
from torch import Tensor

from deeparguing.losses.loss import Loss


class FocalLoss(Loss):

    def __init__(
        self,
        alpha: float | Tensor = 1.0,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    @override
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:

        ce_loss = F.cross_entropy(
            predictions,
            targets,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        pt = torch.exp(-ce_loss)

        if isinstance(self.alpha, Tensor):
            alpha_t = self.alpha[targets]
        else:
            alpha_t = self.alpha

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
