from abc import ABCMeta, abstractmethod
from typing import Callable

import matplotlib.pyplot as plt
import torch
from torch.optim import Optimizer

from deeparguing import GradualAACBR


class Trainer(metaclass=ABCMeta):

    def __init__(self, real_time_logger=lambda _: None) -> None:
        self.losses = []
        self.grads_over_time = []
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
        regulariser=lambda _: 0,
        disable_tqdm=False,
    ):
        pass

    def plot_loss_curve(self):
        plt.plot(self.losses)
        plt.show()
    
    def plot_grads(self):
        grads_over_time = torch.stack(self.grads_over_time).numpy()
        plt.figure(figsize=(10, 6))
        for i in range(grads_over_time.shape[1]):
            plt.plot(grads_over_time[:, i], label=f'Param {i}')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Value')
        plt.title('Gradient Flow Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
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
        regulariser=lambda _: 0,
    ):

        # If in the future we have to overwrite train_step for whatever reason,
        # consider using a strategy pattern instead of inheritance

        optimizer.zero_grad()

        # TODO: consider efficiency issues with having to rebuild each time
        # Find a way to accumulate gradients update only when necessary?
        model.fit(X_casebase, y_casebase, X_default, y_default)

        predictions = model(X_new_cases).squeeze()

        y_target = torch.argmax(y_new_cases, dim=1)
        loss = criterion(predictions, y_target) + regulariser(model)
        loss.backward()

        self.losses.append(loss.item())
        self.real_time_logger(loss.item())
        grads = []
        for param in model.parameters():
            grads.append(param.grad.view(-1))
        grads = torch.cat(grads).detach().cpu()  # shape: (5 + 1) if bias is included
        self.grads_over_time.append(grads)
        optimizer.step()
        return loss
