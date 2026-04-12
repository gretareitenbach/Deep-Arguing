from typing import override

import torch.nn.functional as F
from torch import Tensor

from deeparguing.criterion.criterion import Criterion


class DiceLoss(Criterion):
    """
    Soft Dice loss (also known as F1 loss) for multi-class classification.

    Directly optimizes the Dice coefficient / F1 score per class,
    which can help when dealing with class imbalance.

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice
    """

    def __init__(
        self,
        smooth: float = 1.0,
        reduction: str = "mean",
    ):
        """
        Parameters
        ----------
        smooth : float
            Smoothing factor to avoid division by zero.
        reduction : str
            Reduction method: "mean" (average over classes), "sum", or "none".
        """
        self.smooth = smooth
        self.reduction = reduction

    @override
    def forward(self, model, predictions: Tensor, targets: Tensor) -> Tensor:
        probs = F.softmax(predictions, dim=1)

        if targets.dim() == 1:
            num_classes = predictions.shape[1]
            targets_one_hot = F.one_hot(targets, num_classes).float()
        else:
            targets_one_hot = targets

        intersection = (probs * targets_one_hot).sum(dim=0)
        cardinality = probs.sum(dim=0) + targets_one_hot.sum(dim=0)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_per_class

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss
