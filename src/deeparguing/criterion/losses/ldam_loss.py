from typing import override

import torch
import torch.nn.functional as F
from torch import Tensor

from deeparguing.criterion.criterion import Criterion


class LDAMLoss(Criterion):
    """
    Label-Distribution-Aware Margin (LDAM) Loss.

    Enforces larger margins for minority classes to improve generalization
    on imbalanced datasets.

    Reference: "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss"
    (Cao et al., NeurIPS 2019)

    The margin for class j is: delta_j = C / n_j^(1/4)
    where n_j is the number of samples in class j.
    """

    def __init__(
        self,
        class_counts: Tensor,
        max_margin: float = 0.5,
        scale: float = 30.0,
        reduction: str = "mean",
    ):
        """
        Parameters
        ----------
        class_counts : Tensor
            Number of samples per class, shape (C,).
        max_margin : float
            Maximum margin value (for the rarest class).
        scale : float
            Scaling factor for logits before softmax.
        reduction : str
            Reduction method: "mean", "sum", or "none".
        """
        self.scale = scale
        self.reduction = reduction

        margins = 1.0 / torch.pow(class_counts.float(), 0.25)
        margins = margins / margins.max() * max_margin
        self.margins = margins

    @override
    def forward(self, model, predictions: Tensor, targets: Tensor) -> Tensor:
        device = predictions.device
        margins = self.margins.to(device)

        if targets.dim() > 1:
            target_indices = torch.argmax(targets, dim=1)
        else:
            target_indices = targets

        batch_margins = margins[target_indices]

        predictions_margin = predictions.clone()
        predictions_margin[torch.arange(len(targets), device=device), target_indices] -= batch_margins

        scaled_predictions = predictions_margin * self.scale

        loss = F.cross_entropy(scaled_predictions, target_indices, reduction=self.reduction)

        return loss
