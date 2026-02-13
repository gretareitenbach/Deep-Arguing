from typing import override

import torch
import torch.nn.functional as F
from torch import Tensor

from deeparguing.losses.loss import Loss


class BatchEntropyRegularisation(Loss):

    @override
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:

        probs = F.softmax(predictions, dim=1)

        mean_probs = probs.mean(dim=0)  # average prediction across batch
        negative_entropy = (mean_probs * torch.log(mean_probs + 1e-8)).sum()

        return negative_entropy
