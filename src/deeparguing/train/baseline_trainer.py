from typing import override

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor

from deeparguing.cli.loggers import ExperimentLogger
from deeparguing.models.model import Model
from deeparguing.train.trainer import Trainer


class BaselineTrainer(Trainer):

    def __init__(self) -> None:
        super().__init__()

    @override
    def train(
        self,
        model: Model,
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

        batch_size = getattr(self, "batch_size", None)

        # Fit the baseline model
        model.fit(X_casebase, y_casebase, X_default, y_default, batch_size=batch_size)

        max_val_acc = 0.0
        max_val_f1 = 0.0

        if X_val is not None and y_val is not None:

            predictions = model.forward(X_val)

            if not isinstance(predictions, torch.Tensor):
                predictions = torch.tensor(predictions)
            else:
                predictions = predictions.clone().detach()

            if predictions.dim() > 1 and predictions.shape[1] > 1:
                y_pred = torch.argmax(predictions, dim=1)
            else:
                y_pred = predictions

            y_target = (
                torch.argmax(y_val, dim=1)
                if y_val.dim() > 1 and y_val.shape[1] > 1
                else y_val
            )

            max_val_acc = float(accuracy_score(y_target.cpu(), y_pred.cpu()))
            max_val_f1 = float(
                f1_score(
                    y_target.cpu(), y_pred.cpu(), average="macro", zero_division=0.0
                )
            )

        ExperimentLogger.current().log_metrics(
            {
                "evals/max_val_acc": float(max_val_acc),
                "evals/max_val_f1": float(max_val_f1),
            }
        )

        return max_val_acc, max_val_f1
