from typing import override

from torch import Tensor
from torch.nn import *

from deeparguing.losses.loss import Loss


class TorchLoss(Loss):

    def __init__(self, class_name: str, **kwargs):
        self.loss_func = globals()[class_name](**kwargs)

    @override
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.loss_func(predictions, targets)
