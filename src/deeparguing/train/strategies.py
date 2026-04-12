from abc import ABCMeta, abstractmethod
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer
from sklearn.metrics import accuracy_score, f1_score

from deeparguing import GradualAACBR
from deeparguing.criterion import CriterionType
from deeparguing.criterion import CriterionType
from deeparguing.cli.loggers import ExperimentLogger
from torch.utils.checkpoint import checkpoint


class ValidationLogStrategy(metaclass=ABCMeta):
    @abstractmethod
    def log(
        self,
        model: GradualAACBR,
        batch_size: int | None,
        X_val: Tensor,
        y_val: Tensor,
        criterion: CriterionType,
        regulariser: CriterionType,
        **kwargs
    ) -> tuple[float, float, float]:
        pass


class StandardValidationLog(ValidationLogStrategy):
    def log(
        self,
        model: GradualAACBR,
        batch_size: int | None,
        X_val: Tensor,
        y_val: Tensor,
        criterion: CriterionType,
        regulariser: CriterionType,
        **kwargs
    ) -> tuple[float, float, float]:
        n_samples = X_val.shape[0]
        batch_size = batch_size if batch_size is not None else n_samples
        model.eval()

        val_loss_total = 0.0
        num_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                X_batch = X_val[i : i + batch_size]
                y_batch = y_val[i : i + batch_size]

                predictions = model(X_batch).squeeze()
                y_target = torch.argmax(y_batch, dim=1)
                batch_loss = criterion(model, predictions, y_target) + regulariser(model, predictions, y_target)

                val_loss_total += batch_loss.item()
                num_batches += 1

                predicted_classes = torch.argmax(predictions, dim=1)
                all_preds.extend(predicted_classes.cpu().tolist())
                all_targets.extend(y_target.cpu().tolist())

        val_loss_avg = val_loss_total / num_batches if num_batches > 0 else float("nan")
        if len(all_targets) > 0:
            accuracy = float(accuracy_score(all_targets, all_preds))
            f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0.0))
        else:
            accuracy = float("nan")
            f1 = float("nan")

        model.train()
        return val_loss_avg, accuracy, f1

class CurriculumValidationLog(ValidationLogStrategy):
    def _remap_targets(self, y: Tensor, active_classes: list[int]) -> Tensor:
        original_labels = torch.argmax(y, dim=1)
        active_tensor = torch.tensor(sorted(active_classes), device=y.device)
        return (original_labels.unsqueeze(1) == active_tensor).nonzero()[:, 1]

    def log(
        self,
        model: GradualAACBR,
        batch_size: int | None,
        X_val: Tensor,
        y_val: Tensor,
        criterion: CriterionType,
        regulariser: CriterionType,
        **kwargs
    ) -> tuple[float, float, float]:
        active_classes = kwargs.get("active_classes", list(range(y_val.shape[1])))
        n_samples = X_val.shape[0]
        batch_size = batch_size if batch_size is not None else n_samples
        model.eval()

        val_loss_total = 0.0
        num_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                X_batch = X_val[i : i + batch_size]
                y_batch = y_val[i : i + batch_size]

                predictions = model(X_batch)
                if predictions.dim() > 2 or (predictions.dim() == 2 and predictions.shape[1] > 1):
                    predictions = predictions.squeeze()
                y_target = self._remap_targets(y_batch, active_classes)
                batch_loss = criterion(model, predictions, y_target) + regulariser(model, predictions, y_target)

                val_loss_total += batch_loss.item()
                num_batches += 1

                if predictions.dim() == 1:
                    predictions = predictions.unsqueeze(0)
                predicted_classes = torch.argmax(predictions, dim=1)
                all_preds.extend(predicted_classes.cpu().tolist())
                all_targets.extend(y_target.cpu().tolist())

        val_loss_avg = val_loss_total / num_batches if num_batches > 0 else float("nan")
        if len(all_targets) > 0:
            accuracy = float(accuracy_score(all_targets, all_preds))
            f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0.0))
        else:
            accuracy = float("nan")
            f1 = float("nan")

        model.train()
        return val_loss_avg, accuracy, f1
