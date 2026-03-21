from typing import Any, Callable, override

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from deeparguing import GradualAACBR
from deeparguing.cli.loggers import ExperimentLogger
from deeparguing.losses.loss import Loss
from deeparguing.regularisers import Regulariser, RegulariserType
from deeparguing.train import Trainer


class TwoLevelTrainer(Trainer):
    """
    Trainer with two-level optimization loop for augmented Lagrangian methods.

    Structure:
        for outer_iter in range(outer_iterations):
            for epoch in range(epochs):
                for batch in batches:
                    train_step(...)

            regulariser.step(model)  # Update lambda, rho

            if regulariser.converged:
                break  # Early stopping

    Use with NOTEARSRegulariser for DAG learning.
    """

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
        criterion: Loss,
        epochs: int,
        outer_iterations: int,
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
        """
        Train with a two-level optimization loop.

        Parameters
        ----------
        model : GradualAACBR
            The model to train.
        X_casebase : Tensor
            Casebase features.
        y_casebase : Tensor
            Casebase labels.
        X_new_cases : Tensor
            Training case features.
        y_new_cases : Tensor
            Training case labels.
        X_default : Tensor
            Default case features.
        y_default : Tensor
            Default case labels.
        optimizer : Optimizer
            PyTorch optimizer.
        criterion : torch.nn.Module
            Loss function.
        epochs : int
            Number of inner epochs per outer iteration.
        outer_iterations : int
            Number of outer iterations (lambda/rho updates).
        regulariser : RegulariserType
            Regulariser (use NOTEARSRegulariser for DAG learning).
        disable_tqdm : bool
            Disable progress bars.
        batch_size : int | None
            Batch size (None = full batch).
        scheduler : LRScheduler | None
            Learning rate scheduler.
        scheduler_step_per : str | None
            When to step scheduler: 'epoch', 'batch', or 'outer'.
        gradient_max_norm : float | None
            Max gradient norm for clipping.
        X_val : Tensor | None
            Validation features.
        y_val : Tensor | None
            Validation labels.
        log_val_loss : bool
            Whether to log validation loss.
        log_gradients : bool
            Whether to log gradient norms.
        """
        # Validate scheduler_step_per
        if scheduler is not None:
            if scheduler_step_per is None:
                raise ValueError(
                    "scheduler_step_per must be specified when using a scheduler. "
                    "Valid values: 'epoch', 'batch', or 'outer'"
                )
            if scheduler_step_per not in ("epoch", "batch", "outer"):
                raise ValueError(
                    f"scheduler_step_per must be 'epoch', 'batch', or 'outer', "
                    f"got '{scheduler_step_per}'"
                )

        n_samples = X_new_cases.shape[0]
        batch_size = batch_size if batch_size is not None else n_samples

        # Reset regulariser state
        self._reset_regulariser(regulariser)

        # Track loss for reporting
        loss: Tensor | None = None
        max_val_acc = 0.0

        # Outer loop
        outer_pbar = tqdm(
            range(outer_iterations),
            desc="Outer iterations",
            dynamic_ncols=True,
            disable=disable_tqdm,
        )

        for outer_iter in outer_pbar:
            # Inner loop: run 'epochs' epochs
            inner_pbar = tqdm(
                range(epochs),
                desc=f"Outer {outer_iter}",
                dynamic_ncols=True,
                disable=disable_tqdm,
                leave=False,
            )

            for epoch in inner_pbar:
                permutation = torch.randperm(n_samples, device=model.device)

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
                        optimizer,
                        criterion,
                        regulariser=regulariser,
                        gradient_max_norm=gradient_max_norm,
                        log_gradients=log_gradients,
                    )

                    if scheduler is not None and scheduler_step_per == "batch":
                        scheduler.step()

                if scheduler is not None and scheduler_step_per == "epoch":
                    scheduler.step()

                # Update inner progress bar
                assert loss is not None
                inner_pbar.set_description(
                    f"Outer {outer_iter}, Loss: {loss.item():.6f}"
                )

            # After inner loop: step the regulariser
            converged = self._step_regulariser(regulariser, model)

            # Step scheduler per outer iteration if configured
            if scheduler is not None and scheduler_step_per == "outer":
                scheduler.step()

            # Log metrics
            assert loss is not None
            model.eval()
            with torch.no_grad():
                model.fit(X_casebase, y_casebase, X_default, y_default)
            model.train()
            val_acc = self._log_outer_iteration(
                regulariser,
                outer_iter,
                loss,
                model,
                batch_size,
                X_new_cases,
                y_new_cases,
                X_val,
                y_val,
                criterion,
                log_val_loss,
            )
            max_val_acc = max(max_val_acc, val_acc)

            # Update outer progress bar
            outer_pbar.set_description(
                f"Outer {outer_iter}, Loss: {loss.item():.6f}, Converged: {converged}"
            )

            # Early stopping
            if converged:
                break

        ExperimentLogger.current().log_metrics({"evals/max_val_acc": float(max_val_acc)})
        return float(max_val_acc)

    def _step_regulariser(
        self, regulariser: RegulariserType, model: GradualAACBR
    ) -> bool:
        """Step the regulariser, return True if converged."""
        if isinstance(regulariser, Regulariser):
            return regulariser.step(model)
        return True  # Plain callables always "converged"

    def _reset_regulariser(self, regulariser: RegulariserType) -> None:
        """Reset the regulariser state."""
        if isinstance(regulariser, Regulariser):
            regulariser.reset()

    def _log_outer_iteration(
        self,
        regulariser: RegulariserType,
        outer_iter: int,
        loss: Tensor,
        model: GradualAACBR,
        batch_size: int,
        X_train: Tensor,
        y_train: Tensor,
        X_val: Tensor | None,
        y_val: Tensor | None,
        criterion: Loss,
        log_val_loss: bool,
    ) -> float:
        """Log metrics after each outer iteration. Returns validation accuracy or 0.0"""
        _, train_acc = self.log_validation_loss(
            model, batch_size, X_train, y_train, criterion, regulariser
        )

        metrics: dict[str, float | int | bool] = {
            "loss/loss_per_outer": float(loss.item()),
            "outer_iteration": outer_iter,
            "accuracy/train_accuracy_per_outer": train_acc,
        }

        # Log NOTEARS-specific metrics if available
        if isinstance(regulariser, Regulariser):
            if hasattr(regulariser, "lambda_"):
                metrics["notears/lambda"] = float(regulariser.lambda_)
            if hasattr(regulariser, "rho"):
                metrics["notears/rho"] = float(regulariser.rho)
            if hasattr(regulariser, "h_old") and regulariser.h_old is not None:
                metrics["notears/h"] = float(regulariser.h_old)
            if hasattr(regulariser, "converged"):
                metrics["notears/converged"] = regulariser.converged

        val_acc_to_return = 0.0
        if log_val_loss and X_val is not None and y_val is not None:
            val_loss, val_acc = self.log_validation_loss(
                model, batch_size, X_val, y_val, criterion, regulariser
            )
            metrics["loss/val_loss_per_outer"] = float(val_loss)
            metrics["accuracy/val_accuracy_per_outer"] = val_acc
            val_acc_to_return = val_acc

        ExperimentLogger.current().log_metrics(metrics)
        return float(val_acc_to_return)
