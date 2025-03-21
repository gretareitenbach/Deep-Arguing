from abc import ABCMeta, abstractmethod
from typing import Callable

import matplotlib.pyplot as plt
import torch
from torch.optim import Optimizer

from deeparguing import GradualAACBR


class Trainer(metaclass=ABCMeta):

    def __init__(self, real_time_logger=lambda _: None) -> None:
        self.losses = []
        # Can be used to log the loss in real time (e.g. if logging to weights and biases)
        self.real_time_logger = real_time_logger

    @abstractmethod
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
        graph_regualariser=lambda _: 0,
        disable_tqdm=False,
        post_process_func=lambda A: A,
    ):
        pass

    def plot_loss_curve(self):
        plt.plot(self.losses)
        plt.show()

    def _train_step(
        self,
        model: GradualAACBR,
        X_casebase: torch.Tensor,
        y_casebase: torch.Tensor,
        X_new_cases: torch.Tensor,
        y_new_cases: torch.Tensor,
        X_default: torch.Tensor,
        y_default: torch.Tensor,
        optimizer: Optimizer,
        criterion: Callable,
        graph_regulariser=lambda _: 0,
        post_process_func=lambda x: x,
    ):

        # If in the future we have to overwrite train_step for whatever reason,
        # consider using a strategy pattern instead of inheritance

        optimizer.zero_grad()

        # TODO: consider efficiency issues with having to rebuild each time
        # Find a way to accumulate gradients update only when necessary?
        model.fit(X_casebase, y_casebase, X_default, y_default)

        predictions = model(X_new_cases, post_process_func=post_process_func).squeeze()

        y_target = torch.argmax(y_new_cases, dim=1)

        loss = criterion(predictions, y_target) + graph_regulariser(model)
        loss.backward()

        self.losses.append(loss.item())
        self.real_time_logger(loss.item())
        optimizer.step()
        return loss
