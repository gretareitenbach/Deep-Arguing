from typing import Any, Callable, override

from torch import Tensor
from torch.optim import Optimizer
from tqdm import tqdm

from deeparguing import GradualAACBR
from deeparguing.regulariser import RegulariserType
from deeparguing.train import Trainer
from deeparguing.train.trainer import CriterionFactory


class SimpleTrainer(Trainer):

    def __init__(
        self,
        real_time_logger: Callable[[Any], Any] = lambda _: None,
    ) -> None:
        super().__init__(real_time_logger)

    @override
    def train(
        self,
        model: GradualAACBR,
        X_casebase: Tensor,
        y_casebase: Tensor,
        X_default: Tensor,
        y_default: Tensor,
        optimizer: Optimizer,
        criterion_factory: CriterionFactory,
        epochs: int,
        regulariser: RegulariserType = lambda _: 0,
        disable_tqdm: bool = False,
    ):

        pbar = tqdm(range(epochs), dynamic_ncols=True, disable=disable_tqdm)

        criterion = criterion_factory()

        for epoch in pbar:

            loss = self._train_step(
                model,
                X_casebase,
                y_casebase,
                X_casebase,
                y_casebase,
                X_default,
                y_default,
                optimizer,
                criterion,
                regulariser=regulariser,
            )

            pbar.set_description(f"Epoch {epoch + 1}, Loss: {round(loss.item(), 6)}")
