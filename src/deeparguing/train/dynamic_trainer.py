from typing import Any, Callable, override

import torch
from sklearn.model_selection import KFold
from torch import Tensor
from torch.optim import Optimizer
from tqdm import tqdm

from deeparguing import GradualAACBR
from deeparguing.regulariser import RegulariserType
from deeparguing.train import Trainer
from deeparguing.train.trainer import CriterionFactory


class DynamicTrainer(Trainer):

    def __init__(
        self,
        n_splits: int,
        random_split_state: float,
        real_time_logger: Callable[[Any], Any] = lambda _: None,
    ) -> None:
        super().__init__(real_time_logger)
        self.n_splits = n_splits
        self.random_split_state = random_split_state

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
