from abc import ABCMeta, abstractmethod
from typing import Callable

import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import KFold
from torch.optim import Optimizer
from tqdm import tqdm

from deeparguing import GradualAACBR


class Trainer(metaclass=ABCMeta):

    # To move to model init:
    # use_symmetric_attacks
    # use_blockers=True,
    # use_supports=False

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


# TODO: Move to separate file
class SimpleTrainer(Trainer):

    def __init__(self) -> None:
        super().__init__()

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

        pbar = tqdm(range(epochs), disable=disable_tqdm)

        criterion = criterion_factory()

        for epoch in pbar:

            loss = self._train_step(
                model,
                X_casebase,
                y_casebase,
                X_casebase,
                y_casebase,
                X_default,
                y_default,
                optimizer,
                criterion,
                graph_regulariser=graph_regualariser,
                post_process_func=post_process_func,
            )

            pbar.set_description(f"Epoch {epoch + 1}, Loss: {round(loss.item(), 6)}")


# TODO: Move to separate file
class DynamicTrainer(Trainer):

    def __init__(self, n_splits, random_split_state) -> None:
        super().__init__()
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
        graph_regualariser=lambda _: 0,
        disable_tqdm=False,
        post_process_func=lambda A: A,
    ):
        pbar = tqdm(range(epochs), disable=disable_tqdm)
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
                    graph_regulariser=graph_regualariser,
                    post_process_func=post_process_func,
                )

                pbar.set_description(
                    f"Epoch {epoch + 1}, Loss: {round(loss.item(), 6)}"
                )


# TODO: Move to separate file
class ReweightTrainer(Trainer):

    def __init__(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        criterion_class,
        reweight_epoch=25,
        max_misclass_rate=0.15,
    ) -> None:
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.criterion_class = criterion_class
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
        graph_regualariser=lambda _: 0,
        disable_tqdm=False,
        post_process_func=lambda A: A,
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
                    post_process_func,
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
                graph_regulariser=graph_regualariser,
                post_process_func=post_process_func,
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
        post_process_func,
    ):
        model.fit(
            X_casebase,
            y_casebase,
            X_default,
            y_default,
        )
        predictions = model(self.X_val, post_process_func=post_process_func).squeeze()
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
