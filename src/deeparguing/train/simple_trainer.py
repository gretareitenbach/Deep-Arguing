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
        criterion: torch.nn.Module,
        epochs: int,
        regulariser: RegulariserType = lambda _: 0,
        disable_tqdm: bool = False,
        batch_size: None | int = None,
        scheduler: LRScheduler | None = None,
        scheduler_step_per: str | None = None,
        gradient_max_norm: float | None = None,
        X_val: Tensor | None = None,
        y_val: Tensor | None = None,
        log_val_loss: bool = False,
        log_gradients: bool = False,
    ):
        # Validate scheduler_step_per when scheduler is provided
        if scheduler is not None:
            if scheduler_step_per is None:
                raise ValueError(
                    "scheduler_step_per must be specified when using a scheduler. "
                    "Valid values: 'epoch' or 'batch'"
                )
            if scheduler_step_per not in ("epoch", "batch"):
                raise ValueError(
                    f"scheduler_step_per must be 'epoch' or 'batch', got '{scheduler_step_per}'"
                )

        pbar = tqdm(range(epochs), dynamic_ncols=True, disable=disable_tqdm)

        n_samples = X_new_cases.shape[0]

        batch_size = batch_size if batch_size is not None else n_samples
        for epoch in pbar:
            permutation = torch.randperm(n_samples, device=model.device)

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
                    gradient_max_norm=gradient_max_norm,
                    log_gradients=log_gradients,
                )

                # Step scheduler per batch if configured
                if scheduler is not None and scheduler_step_per == "batch":
                    scheduler.step()

            assert loss

            # Step scheduler per epoch if configured
            if scheduler is not None and scheduler_step_per == "epoch":
                scheduler.step()

            _, train_acc = self.log_validation_loss(
                model, batch_size, X_new_cases, y_new_cases, criterion, regulariser
            )

            if log_val_loss and X_val is not None and y_val is not None:
                val_loss_avg, val_acc = self.log_validation_loss(
                    model, batch_size, X_val, y_val, criterion, regulariser
                )
                ExperimentLogger.current().log_metrics(
                    {
                        "loss_per_epoch": float(loss.item()),
                        "epoch": epoch,
                        "train_accuracy_per_epoch": train_acc,
                        "val_loss_per_epoch": float(val_loss_avg),
                        "val_accuracy_per_epoch": val_acc,
                    }
                )
                pbar.set_description(f"Epoch {epoch}, Loss: {round(loss.item(), 6)}, Val Loss: {round(val_loss_avg, 6)}")
            else:
                ExperimentLogger.current().log_metrics(
                    {"loss_per_epoch": float(loss.item()), "epoch": epoch, "train_accuracy_per_epoch": train_acc}
                )
                pbar.set_description(f"Epoch {epoch}, Loss: {round(loss.item(), 6)}")
