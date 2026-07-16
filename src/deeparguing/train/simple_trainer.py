from typing import override

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from deeparguing import GradualAACBR
from deeparguing.cli.loggers import ExperimentLogger
from deeparguing.criterion import CriterionType
from deeparguing.criterion import CriterionType
from deeparguing.md_log import write_markdown_log
from deeparguing.train.neural_trainer import NeuralTrainer
from deeparguing.train.strategies import StandardValidationLog, ValidationLogStrategy

SUMMARY_LOG_PATH = "outputs/logs/summary.md"


class SimpleTrainer(NeuralTrainer):

    def __init__(
        self,
        epochs: int,
        optimizer: Optimizer,
        criterion: CriterionType,
        validation_log_strategy: ValidationLogStrategy = StandardValidationLog(),
        regulariser: CriterionType = lambda _m, _p, _t: 0,
        batch_size: int | None = None,
        scheduler: LRScheduler | None = None,
        scheduler_step_per: str | None = None,
        gradient_max_norm: float | None = None,
    ) -> None:
        super().__init__(validation_log_strategy)
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.regulariser = regulariser
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.scheduler_step_per = scheduler_step_per
        self.gradient_max_norm = gradient_max_norm

    def _train_step(
        self,
        model: GradualAACBR,
        X_casebase: Tensor,
        y_casebase: Tensor,
        X_new_cases: Tensor,
        y_new_cases: Tensor,
        X_default: Tensor,
        y_default: Tensor,
    ) -> Tensor:
        self.optimizer.zero_grad()

        model.fit(X_casebase, y_casebase, X_default, y_default)

        predictions = model(X_new_cases).squeeze()
        y_target = torch.argmax(y_new_cases, dim=1)

        loss: Tensor = self.criterion(model, predictions, y_target)
        loss += self.regulariser(model, predictions, y_target)

        loss.backward()

        if self.gradient_max_norm is not None:
            self.clip_gradients(model, self.gradient_max_norm)

        if self.log_gradients_flag:
            self.log_gradients(model)

        self.optimizer.step()
        return loss

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
        disable_tqdm: bool = False,
        X_val: Tensor | None = None,
        y_val: Tensor | None = None,
    ) -> tuple[float, float]:

        if self.scheduler is not None:
            if self.scheduler_step_per is None:
                raise ValueError(
                    "scheduler_step_per must be specified when using a scheduler. Valid values: 'epoch' or 'batch'"
                )
            if self.scheduler_step_per not in ("epoch", "batch"):
                raise ValueError(
                    f"scheduler_step_per must be 'epoch' or 'batch', got '{self.scheduler_step_per}'"
                )

        pbar = tqdm(range(self.epochs), dynamic_ncols=True, disable=disable_tqdm)
        n_samples = X_new_cases.shape[0]
        batch_size = self.batch_size if self.batch_size is not None else n_samples
        max_val_acc = 0.0
        max_val_f1 = 0.0

        for epoch in pbar:
            permutation = torch.randperm(n_samples, device=model.device)
            loss: Tensor | None = None

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
                )

                if loss is torch.nan or torch.isnan(loss):
                    print("WARNING: LOSS IS NAN")
                    write_markdown_log(
                        [f"WARNING: loss is NaN at epoch {epoch} -- training aborted"],
                        SUMMARY_LOG_PATH,
                    )
                    return 0.0

                if self.scheduler is not None and self.scheduler_step_per == "batch":
                    self.scheduler.step()

            assert loss is not None

            if self.scheduler is not None and self.scheduler_step_per == "epoch":
                self.scheduler.step()

            model.eval()
            with torch.no_grad():
                model.fit(X_casebase, y_casebase, X_default, y_default)
            model.train()

            post_train_loss, train_acc, train_f1 = self.validation_log_strategy.log(
                model,
                batch_size,
                X_new_cases,
                y_new_cases,
                self.criterion,
                self.regulariser,
            )

            if self.log_val_loss and X_val is not None and y_val is not None:
                val_loss_avg, val_acc, val_f1 = self.validation_log_strategy.log(
                    model, batch_size, X_val, y_val, self.criterion, self.regulariser
                )
                ExperimentLogger.current().log_metrics(
                    {
                        "loss/loss_per_epoch": float(loss.item()),
                        "loss/post_fit_loss_per_epoch": float(post_train_loss),
                        "epoch": epoch,
                        "accuracy/train_accuracy_per_epoch": train_acc,
                        "f1/train_f1_per_epoch": train_f1,
                        "loss/val_loss_per_epoch": float(val_loss_avg),
                        "accuracy/val_accuracy_per_epoch": val_acc,
                        "f1/val_f1_per_epoch": val_f1,
                    }
                )
                max_val_acc = max(max_val_acc, val_acc)
                max_val_f1 = max(max_val_f1, val_f1)
                pbar.set_description(
                    f"Epoch {epoch}, Loss: {round(loss.item(), 6)}, Val Loss: {round(val_loss_avg, 6)}, Val Acc: {round(val_acc, 6)}, Val F1: {round(val_f1, 6)}"
                )
            else:
                ExperimentLogger.current().log_metrics(
                    {
                        "loss/loss_per_epoch": float(loss.item()),
                        "epoch": epoch,
                        "accuracy/train_accuracy_per_epoch": train_acc,
                        "f1/train_f1_per_epoch": train_f1,
                    }
                )
                pbar.set_description(
                    f"Epoch {epoch}, Loss: {round(loss.item(), 6)}, Train Acc: {round(train_acc, 6)}, Train F1: {round(train_f1, 6)}"
                )

        ExperimentLogger.current().log_metrics(
            {
                "evals/max_val_acc": float(max_val_acc),
                "evals/max_val_f1": float(max_val_f1),
            }
        )
        return float(max_val_acc), float(max_val_f1)
