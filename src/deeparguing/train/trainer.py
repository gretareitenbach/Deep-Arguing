from abc import ABCMeta, abstractmethod
from typing import Any, Callable

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from deeparguing import GradualAACBR
from deeparguing.cli.loggers import ExperimentLogger
from deeparguing.losses.loss import Loss
from deeparguing.regularisers import RegulariserType


class Trainer(metaclass=ABCMeta):

    def __init__(self, real_time_logger: Callable[[Any], Any] = lambda _: None) -> None:
        self.losses: list[float] = []
        self.grads_over_time: list[Tensor] = []
        # Can be used to log the loss in real time (e.g. if logging to weights and biases)
        self.real_time_logger = real_time_logger

    @abstractmethod
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
        criterion: Loss,
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
    ) -> float:
        pass

    def plot_loss_curve(self):
        plt.plot(self.losses)
        plt.show()

    def plot_grads(self):
        grads_over_time = torch.stack(self.grads_over_time).numpy()
        plt.figure(figsize=(10, 6))
        for i in range(grads_over_time.shape[1]):
            plt.plot(grads_over_time[:, i], label=f"Param {i}")
        plt.xlabel("Epoch")
        plt.ylabel("Gradient Value")
        plt.title("Gradient Flow Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _train_step(
        self,
        model: GradualAACBR,
        X_casebase: Tensor,
        y_casebase: Tensor,
        X_new_cases: Tensor,
        y_new_cases: Tensor,
        X_default: Tensor,
        y_default: Tensor,
        optimizer: Optimizer,
        criterion: Loss,
        regulariser: RegulariserType = lambda _: 0,
        gradient_max_norm: float | None = None,
        log_gradients: bool = False,
    ) -> Tensor:

        # If in the future we have to overwrite train_step for whatever reason,
        # consider using a strategy pattern instead of inheritance

        optimizer.zero_grad()

        # TODO: consider efficiency issues with having to rebuild each time
        # Find a way to accumulate gradients update only when necessary?
        model.fit(X_casebase, y_casebase, X_default, y_default)

        predictions = model(X_new_cases).squeeze()

        y_target = torch.argmax(y_new_cases, dim=1)

        loss: Tensor = criterion(predictions, y_target)
        loss += regulariser(model)

        loss.backward()

        if gradient_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_max_norm,
                error_if_nonfinite=False,
            )

        if log_gradients:
            ExperimentLogger.current().log_metrics(
                {
                    f"gradients/Gradient {n}": (
                        float(torch.norm(p.grad.detach().cpu()))
                        if p.grad is not None
                        else 0.0
                    )
                    for n, p in model.named_parameters()
                }
            )

        optimizer.step()

        return loss

    def log_validation_loss(
        self,
        model: GradualAACBR,
        batch_size: int | None,
        X_val: Tensor,
        y_val: Tensor,
        criterion: Loss,
        regulariser: RegulariserType,
    ) -> tuple[float, float]:

        n_samples = X_val.shape[0]
        batch_size = batch_size if batch_size is not None else n_samples
        model.eval()

        val_loss_total = 0.0
        num_batches = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                X_batch = X_val[i : i + batch_size]
                y_batch = y_val[i : i + batch_size]

                predictions = model(X_batch).squeeze()
                y_target = torch.argmax(y_batch, dim=1)
                batch_loss = criterion(predictions, y_target) + regulariser(model)

                val_loss_total += batch_loss.item()
                num_batches += 1

                predicted_classes = torch.argmax(predictions, dim=1)
                correct += (predicted_classes == y_target).sum().item()
                total += y_target.size(0)

        val_loss_avg = val_loss_total / num_batches if num_batches > 0 else float("nan")
        accuracy = correct / total if total > 0 else float("nan")

        model.train()
        return val_loss_avg, accuracy
