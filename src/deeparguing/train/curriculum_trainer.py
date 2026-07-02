import logging
from typing import Any, Callable, cast, override

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from deeparguing import GradualAACBR
from deeparguing.cli.loggers import ExperimentLogger
from deeparguing.criterion import CriterionType
from deeparguing.criterion import CriterionType
from deeparguing.train.curriculum import CurriculumStrategy, DataSelector
from deeparguing.train.neural_trainer import NeuralTrainer
from deeparguing.train.strategies import (
    CurriculumValidationLog,
    ValidationLogStrategy,
)


class CurriculumTrainer(NeuralTrainer):
    """
    Trainer implementing class-based curriculum learning.

    Gradually introduces classes according to a CurriculumStrategy,
    using pre-computed casebase centroids that are incrementally added
    as new classes are introduced.

    Parameters
    ----------
    curriculum_strategy : CurriculumStrategy
        Controls when and which classes are introduced
    data_selector : DataSelector
        Controls data filtering and weighted sampling
    reset_optimizer_on_advance : bool
        If True, clears optimizer state (momentum, etc.) when a new
        class is introduced. Default False.

    Notes
    -----
    - Pre-computed casebase centroids are filtered per phase (not recomputed)
    - Default arguments are filtered to match active classes
    - Target labels are remapped to filtered class space during training
    - Metrics logged each epoch:
        - curriculum_accuracy: accuracy on active classes only
        - curriculum_loss: loss on active classes
        - full_val_accuracy: accuracy on all classes (if validation data provided)
        - full_val_loss: loss on all classes
        - num_active_classes: current number of classes in curriculum
        - active_classes: list of active class indices (for clarity in logs)
    """

    def __init__(
        self,
        curriculum_strategy: CurriculumStrategy,
        data_selector: DataSelector,
        validation_log_strategy: ValidationLogStrategy | None = None,
        reset_optimizer_on_advance: bool = False,
    ):
        super().__init__(
            validation_log_strategy
            if validation_log_strategy is not None
            else CurriculumValidationLog()
        )
        self.curriculum_strategy = curriculum_strategy
        self.data_selector = data_selector
        self.reset_optimizer_on_advance = reset_optimizer_on_advance
        # Store active classes for use in training step
        self._current_active_classes: list[int] = []

    def _filter_defaults(
        self,
        X_default: Tensor,
        y_default: Tensor,
        active_classes: list[int],
    ) -> tuple[Tensor, Tensor]:
        """
        Filters defaults to active classes, sorted by class index.

        Detects the class for each default from y_default's one-hot encoding,
        then returns defaults sorted by class index with remapped labels.

        Parameters
        ----------
        X_default : Tensor
            Default argument features, shape (num_classes, ...)
        y_default : Tensor
            Default argument labels (one-hot), shape (num_classes, num_classes)
        active_classes : list[int]
            Class indices to include

        Returns
        -------
        tuple[Tensor, Tensor]
            Filtered (X_default, y_default) where:
            - X_default is sorted by class index
            - y_default has remapped one-hot labels for the filtered class space
              (shape: num_active_classes, num_active_classes)
        """
        num_total_defaults = y_default.shape[0]

        # Detect actual class for each default from one-hot encoding
        class_to_default_idx: dict[int, int] = {}
        for i in range(num_total_defaults):
            class_idx = int(torch.argmax(y_default[i]).item())
            if class_idx in active_classes:
                class_to_default_idx[class_idx] = i

        # Sort active classes and get corresponding default indices
        sorted_classes = sorted(active_classes)
        indices = [class_to_default_idx[c] for c in sorted_classes]

        # Filter X_default
        X_filtered = X_default[indices]

        # Create new y_default with remapped one-hot labels
        # After filtering, default[i] should have one-hot label [0,...,1,...,0]
        # where the 1 is at position i (since we sorted by class index)
        num_active = len(active_classes)
        y_filtered = torch.eye(
            num_active, device=y_default.device, dtype=y_default.dtype
        )

        return X_filtered, y_filtered

    def _validate_epochs(self, epochs: int) -> None:
        """Validate that total epochs are sufficient for the curriculum."""
        epc = self.curriculum_strategy.epochs_per_class
        if epc is not None:
            num_classes = self.curriculum_strategy.total_classes
            min_required = epc * num_classes
            if epochs < min_required:
                raise ValueError(
                    f"Total epochs ({epochs}) must be >= "
                    f"epochs_per_class ({epc}) x "
                    f"num_classes ({num_classes}) = {min_required}."
                )

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
        criterion: CriterionType,
        epochs: int,
        regulariser: CriterionType = lambda _m, _p, _t: 0,
        disable_tqdm: bool = False,
        batch_size: int | None = None,
        scheduler: LRScheduler | None = None,
        scheduler_step_per: str | None = None,
        gradient_max_norm: float | None = None,
        X_val: Tensor | None = None,
        y_val: Tensor | None = None,
    ) -> tuple[float, float]:
        # Validation
        self._validate_epochs(epochs)
        if scheduler is not None:
            if scheduler_step_per is None:
                raise ValueError(
                    "scheduler_step_per must be specified when using a scheduler."
                )
            if scheduler_step_per not in ("epoch", "batch"):
                raise ValueError(
                    f"scheduler_step_per must be 'epoch' or 'batch', "
                    f"got '{scheduler_step_per}'"
                )

        # Reset curriculum
        self.curriculum_strategy.reset()

        # Initialize state
        active_classes = self.curriculum_strategy.get_active_classes()
        new_classes = self.curriculum_strategy.get_newly_added_classes()
        phase_epoch = 0

        logging.info(f"Curriculum: starting with classes {active_classes}")

        pbar = tqdm(range(epochs), dynamic_ncols=True, disable=disable_tqdm)
        max_val_acc = 0.0
        max_val_f1 = 0.0

        for epoch in pbar:
            # Store active classes for training step
            self._current_active_classes = active_classes

            # Filter training data for active classes
            X_cb_curr, y_cb_curr = self.data_selector.filter_by_classes(
                X_casebase, y_casebase, active_classes
            )
            X_new_curr, y_new_curr = self.data_selector.filter_by_classes(
                X_new_cases, y_new_cases, active_classes
            )
            # Filter defaults to active classes (sorted by class index)
            X_def_curr, y_def_curr = self._filter_defaults(
                X_default, y_default, active_classes
            )

            # Get sample weights
            sample_weights = self.data_selector.get_sample_weights(
                y_new_curr, active_classes, new_classes, phase_epoch
            )

            # Train one epoch
            loss = self._train_curriculum_epoch(
                model=model,
                X_casebase=X_cb_curr,
                y_casebase=y_cb_curr,
                X_new_cases=X_new_curr,
                y_new_cases=y_new_curr,
                X_default=X_def_curr,
                y_default=y_def_curr,
                sample_weights=sample_weights,
                optimizer=optimizer,
                criterion=criterion,
                regulariser=regulariser,
                batch_size=batch_size,
                scheduler=scheduler,
                scheduler_step_per=scheduler_step_per,
                gradient_max_norm=gradient_max_norm,
                active_classes=active_classes,
            )

            # Scheduler step per epoch
            if scheduler is not None and scheduler_step_per == "epoch":
                scheduler.step()

            # Curriculum-scope metrics (use filtered casebase/defaults)
            model.eval()
            with torch.no_grad():
                model.fit(X_cb_curr, y_cb_curr, X_def_curr, y_def_curr)
                curr_loss, curr_acc, curr_f1 = self.validation_log_strategy.log(
                    model,
                    batch_size,
                    X_new_curr,
                    y_new_curr,
                    criterion,
                    regulariser,
                    active_classes=active_classes,
                )
            model.train()

            # Prepare metrics
            metrics: dict[str, Any] = {
                "epoch": epoch,
                "loss/loss_per_epoch": float(loss.item()),
                "accuracy/curriculum_accuracy": curr_acc,
                "f1/curriculum_f1": curr_f1,
                "loss/curriculum_loss": curr_loss,
                "num_active_classes": len(active_classes),
                "active_classes": str(active_classes),  # For clarity in logs
                "phase_epoch": phase_epoch,
            }

            # Full-scope validation (use full casebase/defaults)
            if self.log_val_loss and X_val is not None and y_val is not None:
                with torch.no_grad():
                    model.fit(X_casebase, y_casebase, X_default, y_default)
                    full_loss, full_acc, full_f1 = self.validation_log_strategy.log(
                        model,
                        batch_size,
                        X_val,
                        y_val,
                        criterion,
                        regulariser,
                    )
                metrics["accuracy/full_val_accuracy"] = full_acc
                metrics["f1/full_val_f1"] = full_f1
                metrics["loss/full_val_loss"] = full_loss
                max_val_acc = max(max_val_acc, full_acc)
                max_val_f1 = max(max_val_f1, full_f1)

            ExperimentLogger.current().log_metrics(metrics)

            desc = (
                f"Epoch {epoch} | Classes: {active_classes} | "
                f"Acc: {curr_acc:.3f} | Loss: {loss.item():.4f}"
            )
            pbar.set_description(desc)

            phase_epoch += 1

            # Check advancement
            if self.curriculum_strategy.should_advance(epoch, phase_epoch, curr_acc):
                newly_added = self.curriculum_strategy.advance()
                if newly_added:
                    active_classes = self.curriculum_strategy.get_active_classes()
                    new_classes = newly_added
                    phase_epoch = 0
                    logging.info(
                        f"Curriculum: epoch {epoch}, added classes {newly_added}, "
                        f"active: {active_classes}"
                    )
                    if self.reset_optimizer_on_advance:
                        self._reset_optimizer_state(optimizer)

        logging.info(f"Curriculum complete. Final active classes: {active_classes}")

        ExperimentLogger.current().log_metrics(
            {
                "evals/max_val_acc": float(max_val_acc),
                "evals/max_val_f1": float(max_val_f1),
            }
        )
        return float(max_val_acc), float(max_val_f1)

    def _reset_optimizer_state(self, optimizer: Optimizer) -> None:
        """Reset optimizer state so momentum and other cached values are cleared."""
        for group in optimizer.param_groups:
            group.setdefault("momentum_buffer", None)
            for p in group.get("params", []):
                if p.grad is not None:
                    p.grad = None
        optimizer.zero_grad(set_to_none=True)

    def _train_curriculum_epoch(
        self,
        model: GradualAACBR,
        X_casebase: Tensor,
        y_casebase: Tensor,
        X_new_cases: Tensor,
        y_new_cases: Tensor,
        X_default: Tensor,
        y_default: Tensor,
        sample_weights: Tensor,
        optimizer: Optimizer,
        criterion: CriterionType,
        regulariser: CriterionType,
        batch_size: int | None,
        scheduler: LRScheduler | None,
        scheduler_step_per: str | None,
        gradient_max_norm: float | None,
        active_classes: list[int],
    ) -> Tensor:
        """Trains one epoch with weighted sampling and target remapping."""
        n_samples = X_new_cases.shape[0]
        batch_size = batch_size if batch_size is not None else n_samples

        # Weighted sampling
        weights_list = cast(list[float], sample_weights.tolist())
        sampler = WeightedRandomSampler(
            weights=weights_list,
            num_samples=n_samples,
            replacement=True,
        )
        indices = list(sampler)

        loss = None
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_X = X_new_cases[batch_indices]
            batch_y = y_new_cases[batch_indices]

            loss = self._train_step(
                model,
                X_casebase,
                y_casebase,
                batch_X,
                batch_y,
                X_default,
                y_default,
                optimizer,
                criterion,
                regulariser,
                gradient_max_norm,
                active_classes,
            )

            if scheduler is not None and scheduler_step_per == "batch":
                scheduler.step()

        assert loss is not None
        return loss

    def _remap_targets(
        self,
        y: Tensor,
        active_classes: list[int],
    ) -> Tensor:
        original_labels = torch.argmax(y, dim=1)
        active_tensor = torch.tensor(sorted(active_classes), device=y.device)
        return (original_labels.unsqueeze(1) == active_tensor).nonzero()[:, 1]

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
        criterion: CriterionType,
        regulariser: CriterionType,
        gradient_max_norm: float | None,
        active_classes: list[int] = [],
    ) -> Tensor:
        optimizer.zero_grad()
        model.fit(X_casebase, y_casebase, X_default, y_default)

        predictions = model(X_new_cases)
        if predictions.dim() > 2 or (
            predictions.dim() == 2 and predictions.shape[1] > 1
        ):
            predictions = predictions.squeeze()

        y_target = self._remap_targets(y_new_cases, active_classes)

        try:
            loss: Tensor = criterion(model, predictions, y_target)
        except TypeError:
            loss: Tensor = criterion(predictions, y_target)

        try:
            loss = loss + regulariser(model, predictions, y_target)
        except TypeError:
            loss = loss + regulariser(predictions, y_target)
        loss.backward()

        if gradient_max_norm is not None:
            self.clip_gradients(model, gradient_max_norm)

        if self.log_gradients_flag:
            self.log_gradients(model)

        optimizer.step()
        return loss
