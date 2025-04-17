from typing import Callable

import torch
from sklearn.model_selection import KFold
from torch.optim import Optimizer
from tqdm import tqdm

from deeparguing import GradualAACBR
from deeparguing.train import Trainer


class DynamicTrainer(Trainer):

    def __init__(
        self,
        n_splits,
        random_split_state,
        real_time_logger=lambda _: None,
    ) -> None:
        super().__init__(real_time_logger)
        self.n_splits = n_splits
        self.random_split_state = random_split_state

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
        pbar = tqdm(range(epochs), dynamic_ncols=True, disable=disable_tqdm)
        kf = KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_split_state
        )

        criterion = criterion_factory()

        for epoch in pbar:
            for _, (casebase_index, new_cases_index) in enumerate(kf.split(X_casebase)):

                X_sub_casebase = X_casebase[casebase_index]
                y_sub_casebase = y_casebase[casebase_index]

                X_new_cases = X_casebase[new_cases_index]
                y_new_cases = y_casebase[new_cases_index]

                loss = self._train_step(
                    model,
                    X_sub_casebase,
                    y_sub_casebase,
                    X_new_cases,
                    y_new_cases,
                    X_default,
                    y_default,
                    optimizer,
                    criterion,
                    regulariser=regulariser,
                )

                pbar.set_description(
                    f"Epoch {epoch + 1}, Loss: {round(loss.item(), 6)}"
                )
