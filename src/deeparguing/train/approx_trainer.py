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

from torch.utils.checkpoint import checkpoint


class ApproximateTrainer(Trainer):

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
            permutation = torch.randperm(n_samples)
            total_loss = None

            optimizer.zero_grad()
            model.fit(X_casebase, y_casebase, X_default, y_default)

            for i in range(0, n_samples, batch_size):
                indices = permutation[i : i + batch_size]
                batch_X_new_cases = X_new_cases[indices]
                batch_y_new_cases = y_new_cases[indices]

                predictions = checkpoint(model, batch_X_new_cases).squeeze()

                y_target = torch.argmax(batch_y_new_cases, dim=1)
                loss: Tensor = criterion(predictions, y_target) + regulariser(model)

                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss = total_loss + loss

                # Step scheduler per batch if configured
                if scheduler is not None and scheduler_step_per == "batch":
                    scheduler.step()

            assert total_loss
            total_loss = total_loss / (n_samples / batch_size)

            total_loss.backward()

            optimizer.step()

            # Step scheduler per epoch if configured
            if scheduler is not None and scheduler_step_per == "epoch":
                scheduler.step()

            if gradient_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_max_norm,
                    error_if_nonfinite=False,
                )

            if log_gradients:
                ExperimentLogger.current().log_metrics(
                    {
                        f"Gradient {n}": torch.norm(p.grad.cpu())
                        for n, p in model.named_parameters()
                    }
                )

            pbar.set_description(
                f"Epoch {epoch + 1}, Loss: {round(total_loss.item(), 6)}"
            )

            model.eval()
            with torch.no_grad():
                model.fit(X_casebase, y_casebase, X_default, y_default)
            model.train()

            _, train_acc = self.log_validation_loss(
                model, batch_size, X_new_cases, y_new_cases, criterion, regulariser
            )

            if log_val_loss and X_val is not None and y_val is not None:
                val_loss_avg, val_acc = self.log_validation_loss(
                    model, batch_size, X_val, y_val, criterion, regulariser
                )
                ExperimentLogger.current().log_metrics(
                    {
                        "loss_per_epoch": total_loss.item(),
                        "epoch": epoch,
                        "train_accuracy_per_epoch": train_acc,
                        "val_loss_per_epoch": val_loss_avg,
                        "val_accuracy_per_epoch": val_acc,
                    }
                )
            else:
                ExperimentLogger.current().log_metrics(
                    {"loss_per_epoch": total_loss.item(), "epoch": epoch, "train_accuracy_per_epoch": train_acc}
                )
