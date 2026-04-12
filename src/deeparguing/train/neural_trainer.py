from typing import Any, Callable

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.optim import Optimizer

from deeparguing import GradualAACBR
from deeparguing.cli.loggers import ExperimentLogger
from deeparguing.criterion import CriterionType
from deeparguing.criterion import CriterionType
from deeparguing.train.strategies import ValidationLogStrategy
from deeparguing.train.trainer import Trainer


class NeuralTrainer(Trainer):
    def __init__(
        self,
        validation_log_strategy: ValidationLogStrategy,
    ) -> None:
        super().__init__()
        self.validation_log_strategy = validation_log_strategy
        self.losses: list[float] = []
        self.grads_over_time: list[Tensor] = []
        self.log_val_loss = False
        self.log_gradients_flag = False

    def set_logging_flags(self, log_val_loss: bool, log_gradients: bool) -> None:
        self.log_val_loss = log_val_loss
        self.log_gradients_flag = log_gradients

    def clip_gradients(self, model: GradualAACBR, gradient_max_norm: float):
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_max_norm,
            error_if_nonfinite=False,
        )

    def log_gradients(self, model: GradualAACBR):
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
