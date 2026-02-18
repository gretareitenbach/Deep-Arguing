from typing import override

import torch
import torch.nn.functional as F
from torch import Tensor

from deeparguing.losses.loss import Loss


class PerClassEntropyRegularisation(Loss):
    """
    Maximize entropy of predictions within each target class.

    This regulariser groups predictions by their target class and computes
    the entropy of the average prediction within each class. This encourages
    the model to make diverse predictions for each class rather than always
    predicting the same output for all samples of a class.

    Returns negative entropy (to be minimized).
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Parameters
        ----------
        epsilon : float
            Small constant for numerical stability in log computation.
        """
        self.epsilon = epsilon

    @override
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        probs = F.softmax(predictions, dim=1)

        if targets.dim() > 1:
            target_indices = torch.argmax(targets, dim=1)
        else:
            target_indices = targets

        num_classes = predictions.shape[1]
        device = predictions.device

        total_negative_entropy = torch.zeros(1, device=device).squeeze()
        num_valid_classes = 0

        for c in range(num_classes):
            class_mask = target_indices == c
            if class_mask.sum() == 0:
                continue

            class_probs = probs[class_mask]
            mean_probs = class_probs.mean(dim=0)

            negative_entropy = (mean_probs * torch.log(mean_probs + self.epsilon)).sum()
            total_negative_entropy = total_negative_entropy + negative_entropy
            num_valid_classes += 1

        if num_valid_classes > 0:
            total_negative_entropy = total_negative_entropy / num_valid_classes

        return total_negative_entropy
