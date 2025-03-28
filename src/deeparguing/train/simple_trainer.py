from typing import Callable

import torch
from torch.optim import Optimizer
from tqdm import tqdm

from deeparguing import GradualAACBR
from deeparguing.train import Trainer


class SimpleTrainer(Trainer):

    def __init__(
        self,
        real_time_logger=lambda _: None,
    ) -> None:
        super().__init__(real_time_logger)

    def train(
        self,
        model: GradualAACBR,
        X_casebase: torch.Tensor,
        y_casebase: torch.Tensor,
        X_default: torch.Tensor,
        y_default: torch.Tensor,
        optimizer: Optimizer,
        criterion_factory: Callable,
        epochs,
        regulariser=lambda _: 0,
        disable_tqdm=False,
    ):

        pbar = tqdm(range(epochs), disable=disable_tqdm)

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
