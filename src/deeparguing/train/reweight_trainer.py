from typing import Callable

import torch
from torch.optim import Optimizer
from tqdm import tqdm

from deeparguing import GradualAACBR
from deeparguing.train import Trainer


class ReweightTrainer(Trainer):

    def __init__(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        reweight_epoch=25,
        max_misclass_rate=0.15,
        real_time_logger=lambda _: None,
    ) -> None:
        super().__init__(real_time_logger)
        self.X_val = X_val
        self.y_val = y_val
        self.max_misclass_rate = max_misclass_rate
        self.reweight_epoch = reweight_epoch

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

        built_criterion = None

        for epoch in pbar:

            if (epoch % self.reweight_epoch) == 0:
                built_criterion, stop = self.__build_criterion(
                    model,
                    X_casebase,
                    y_casebase,
                    X_default,
                    y_default,
                    criterion_factory,
                )

                if stop:
                    break

            assert built_criterion != None

            loss = self._train_step(
                model,
                X_casebase,
                y_casebase,
                X_casebase,
                y_casebase,
                X_default,
                y_default,
                optimizer,
                built_criterion,
                regulariser=regulariser,
            )

            pbar.set_description(f"Epoch {epoch + 1}, Loss: {round(loss.item(), 6)}")

    def __build_criterion(
        self,
        model,
        X_casebase: torch.Tensor,
        y_casebase: torch.Tensor,
        X_default: torch.Tensor,
        y_default: torch.Tensor,
        criterion_factory: Callable,
    ):
        model.fit(
            X_casebase,
            y_casebase,
            X_default,
            y_default,
        )
        predictions = model(self.X_val).squeeze()
        y_val = torch.argmax(self.y_val, dim=1)
        predictions = torch.argmax(predictions, dim=1)
        num_classes = len(torch.unique(y_val))
        device = X_casebase.device
        total = torch.zeros(num_classes, device=device)
        misclassified = torch.zeros(num_classes, device=device)
        for cls in range(num_classes):
            total[cls] += (y_val == cls).sum()
            misclassified[cls] += ((predictions != y_val) & (y_val == cls)).sum()

        misclassification_rate = misclassified / total
        weights = misclassification_rate / (misclassification_rate.sum() + 1e-8)
        criterion = criterion_factory(weight=weights)
        stop = torch.all(misclassification_rate < self.max_misclass_rate)
        return criterion, stop
