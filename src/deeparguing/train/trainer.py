from abc import ABCMeta, abstractmethod
from typing import Any, Callable

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.profiler import record_function

from deeparguing import GradualAACBR
from deeparguing.cli.loggers import ExperimentLogger
from deeparguing.regulariser import RegulariserType

type CriterionFactory = Callable[..., torch.nn.Module]


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
        criterion: torch.nn.Module,
        epochs: int,
        regulariser: RegulariserType = lambda _: 0,
        disable_tqdm: bool = False,
        batch_size: None | int = None,
        scheduler: LRScheduler | None = None,
        gradient_max_norm: float | None = None,
        X_val: Tensor | None = None,
        y_val: Tensor | None = None,
    ):
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
        criterion: Callable[[Tensor, Tensor], Tensor],
        regulariser: RegulariserType = lambda _: 0,
        scheduler: LRScheduler | None = None,
        gradient_max_norm: float | None = None,
    ) -> Tensor:

        # If in the future we have to overwrite train_step for whatever reason,
        # consider using a strategy pattern instead of inheritance

        with record_function("my_zero_grad"):
            optimizer.zero_grad()

        # TODO: consider efficiency issues with having to rebuild each time
        # Find a way to accumulate gradients update only when necessary?
        with record_function("my_fit"):
            model.fit(X_casebase, y_casebase, X_default, y_default)

        with record_function("my_forward"):
            predictions = model(X_new_cases).squeeze()

        y_target = torch.argmax(y_new_cases, dim=1)
        with record_function("my_loss"):
            loss: Tensor = criterion(predictions, y_target) 

        with record_function("my_regulariser"):
            loss += regulariser(model)

        with record_function("my_backward"):
            loss.backward()

        # self.losses.append(loss.item())
        # with record_function("my_log_loss"):
        #     self.real_time_logger(loss.item())
        #     ExperimentLogger.current().log_metrics({"loss": float(loss.item())})

        with record_function("my_gradient_clip"):
            if gradient_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=gradient_max_norm, error_if_nonfinite=True
                )

        # grads: list[Tensor] = []
        # for param in model.parameters():
        #     if param.grad is not None:
        #         grads.append(param.grad.view(-1))
        # grads = torch.cat(grads).detach().cpu()  # shape: (5 + 1) if bias is included
        # self.grads_over_time.append(grads)

        # with record_function("my_log_gradients"):
            # ExperimentLogger.current().log_metrics(
            #     {
            #         f"Gradient {n}": float(torch.norm(p.grad.detach().cpu()))
            #         for n, p in model.named_parameters()
            #     }
            # )

        with record_function("my_optimizer_step"):
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        return loss

    def log_validation_loss(
        self,
        model: GradualAACBR,
        batch_size: int | None,
        X_val: Tensor,
        y_val: Tensor,
        criterion: torch.nn.Module,
        regulariser: RegulariserType,
    ):
        n_samples = X_val.shape[0]
        batch_size = batch_size if batch_size is not None else n_samples
        model.eval()
        val_loss_total = 0.0
        num_batches = 0

        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                X_batch = X_val[i : i + batch_size]
                y_batch = y_val[i : i + batch_size]

                predictions = model(X_batch).squeeze()
                y_target = torch.argmax(y_batch, dim=1)
                batch_loss = criterion(predictions, y_target) + regulariser(model)

                val_loss_total += batch_loss.item()
                num_batches += 1

        val_loss_avg = val_loss_total / num_batches if num_batches > 0 else float("nan")

        model.train()  # restore training mode
        return val_loss_avg
