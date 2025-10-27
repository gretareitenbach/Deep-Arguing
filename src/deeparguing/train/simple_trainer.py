from typing import Any, Callable, override

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from deeparguing import GradualAACBR
from deeparguing.cli.loggers import ExperimentLogger
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
        X_new_cases: Tensor,
        y_new_cases: Tensor,
        X_default: Tensor,
        y_default: Tensor,
        optimizer: Optimizer,
        criterion_factory: CriterionFactory,
        epochs: int,
        regulariser: RegulariserType = lambda _: 0,
        disable_tqdm: bool = False,
        batch_size: None | int = None,
        scheduler: LRScheduler | None = None,
    ):

        pbar = tqdm(range(epochs), dynamic_ncols=True, disable=disable_tqdm)

        criterion = criterion_factory()

        n_samples = X_new_cases.shape[0]

        batch_size = batch_size if batch_size is not None else n_samples

        for epoch in pbar:
            permutation = torch.randperm(n_samples)

            for i in range(0, n_samples, batch_size):
                indices = permutation[i : i + batch_size]
                batch_X_new_cases = X_new_cases[indices]
                batch_y_new_cases = y_new_cases[indices]

                loss = self._train_step(
                    model,
                    X_casebase,
                    y_casebase,
                    batch_X_new_cases,
                    batch_y_new_cases,
                    X_default,
                    y_default,
                    optimizer,
                    criterion,
                    regulariser=regulariser,
                    scheduler=scheduler,
                )

            pbar.set_description(f"Epoch {epoch + 1}, Loss: {round(loss.item(), 6)}")

            ExperimentLogger.current().log_metrics(
                {"loss_per_epoch": loss.item(), "epoch": epoch}
            )
