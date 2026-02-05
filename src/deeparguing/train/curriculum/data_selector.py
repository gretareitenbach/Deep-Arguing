from abc import ABC, abstractmethod
from typing import override

import torch
from torch import Tensor


class DataSelector(ABC):
    """
    Abstract strategy for filtering and weighting training data.
    """

    @abstractmethod
    def filter_by_classes(
        self,
        X: Tensor,
        y: Tensor,
        active_classes: list[int],
    ) -> tuple[Tensor, Tensor]:
        """
        Filters data to only include samples from active classes.

        The returned y tensor should have labels remapped to the filtered
        class space. For example, if active_classes=[0, 5], a sample
        with original class 5 should have one-hot label [0, 1] in the output.

        Parameters
        ----------
        X : Tensor
            Input features, shape (N, ...)
        y : Tensor
            One-hot labels, shape (N, num_classes)
        active_classes : list[int]
            Class indices to keep

        Returns
        -------
        tuple[Tensor, Tensor]
            Filtered (X, y) tensors where y has shape (N_filtered, num_active_classes)
        """
        pass

    @abstractmethod
    def get_sample_weights(
        self,
        y: Tensor,
        active_classes: list[int],
        new_classes: list[int],
        phase_epoch: int,
    ) -> Tensor:
        """
        Returns per-sample weights for weighted sampling.

        Parameters
        ----------
        y : Tensor
            One-hot labels for filtered data
        active_classes : list[int]
            All currently active class indices
        new_classes : list[int]
            Classes introduced in most recent advance
        phase_epoch : int
            Epochs since last class introduction

        Returns
        -------
        Tensor
            Per-sample weights, shape (N,)
        """
        pass


class UniformDataSelector(DataSelector):
    """
    Uniform sampling without class weighting.

    All samples are weighted equally regardless of when their
    class was introduced. Useful as a baseline for ablation studies.
    """

    @override
    def filter_by_classes(
        self,
        X: Tensor,
        y: Tensor,
        active_classes: list[int],
    ) -> tuple[Tensor, Tensor]:
        class_indices = torch.argmax(y, dim=1)
        active_set = set(active_classes)
        mask = torch.tensor(
            [c.item() in active_set for c in class_indices],
            device=X.device,
            dtype=torch.bool,
        )
        X_filtered = X[mask]
        y_filtered_original = y[mask]

        # Remap labels to filtered class space
        # sorted_classes[i] -> i in new space
        sorted_classes = sorted(active_classes)
        class_to_new_idx = {c: i for i, c in enumerate(sorted_classes)}

        # Create new one-hot labels
        filtered_class_indices = torch.argmax(y_filtered_original, dim=1)
        num_filtered = X_filtered.shape[0]
        num_active = len(active_classes)
        y_remapped = torch.zeros(
            (num_filtered, num_active), device=y.device, dtype=y.dtype
        )
        for i, c in enumerate(filtered_class_indices):
            new_idx = class_to_new_idx[int(c.item())]
            y_remapped[i, new_idx] = 1.0

        return X_filtered, y_remapped

    @override
    def get_sample_weights(
        self,
        y: Tensor,
        active_classes: list[int],
        new_classes: list[int],
        phase_epoch: int,
    ) -> Tensor:
        return torch.ones(y.shape[0], device=y.device, dtype=torch.float32)


class WeightedDataSelector(DataSelector):
    """
    Over-samples newly introduced classes with optional decay.

    Parameters
    ----------
    new_class_weight : float
        Initial weight multiplier for new classes.
        E.g., 2.0 means new class samples are twice as likely to be selected.
    decay_epochs : int
        Epochs over which weight decays linearly to 1.0.
        If 0, weight remains constant throughout the phase.

    Example
    -------
    With new_class_weight=3.0 and decay_epochs=6:
    - phase_epoch=0: new class weight = 3.0
    - phase_epoch=3: new class weight = 2.0
    - phase_epoch=6+: new class weight = 1.0
    """

    def __init__(
        self,
        new_class_weight: float = 2.0,
        decay_epochs: int = 5,
    ):
        if new_class_weight < 1.0:
            raise ValueError("new_class_weight must be >= 1.0")
        if decay_epochs < 0:
            raise ValueError("decay_epochs must be >= 0")

        self.new_class_weight = new_class_weight
        self.decay_epochs = decay_epochs

    @override
    def filter_by_classes(
        self,
        X: Tensor,
        y: Tensor,
        active_classes: list[int],
    ) -> tuple[Tensor, Tensor]:
        class_indices = torch.argmax(y, dim=1)
        active_set = set(active_classes)
        mask = torch.tensor(
            [c.item() in active_set for c in class_indices],
            device=X.device,
            dtype=torch.bool,
        )
        X_filtered = X[mask]
        y_filtered_original = y[mask]

        # Remap labels to filtered class space
        # sorted_classes[i] -> i in new space
        sorted_classes = sorted(active_classes)
        class_to_new_idx = {c: i for i, c in enumerate(sorted_classes)}

        # Create new one-hot labels
        filtered_class_indices = torch.argmax(y_filtered_original, dim=1)
        num_filtered = X_filtered.shape[0]
        num_active = len(active_classes)
        y_remapped = torch.zeros(
            (num_filtered, num_active), device=y.device, dtype=y.dtype
        )
        for i, c in enumerate(filtered_class_indices):
            new_idx = class_to_new_idx[int(c.item())]
            y_remapped[i, new_idx] = 1.0

        return X_filtered, y_remapped

    @override
    def get_sample_weights(
        self,
        y: Tensor,
        active_classes: list[int],
        new_classes: list[int],
        phase_epoch: int,
    ) -> Tensor:
        """
        Returns per-sample weights for weighted sampling.

        Note: y is expected to have remapped labels (from filter_by_classes),
        where class indices are in [0, len(active_classes)).
        new_classes contains original class indices which are converted
        to remapped indices internally.
        """
        weights = torch.ones(y.shape[0], device=y.device, dtype=torch.float32)

        if not new_classes or self.new_class_weight <= 1.0:
            return weights

        # Convert original class indices to remapped indices
        sorted_classes = sorted(active_classes)
        class_to_new_idx = {c: i for i, c in enumerate(sorted_classes)}
        new_classes_remapped = {class_to_new_idx[c] for c in new_classes if c in class_to_new_idx}

        if not new_classes_remapped:
            return weights

        # Calculate decayed weight
        if self.decay_epochs > 0:
            decay_factor = max(0.0, 1.0 - phase_epoch / self.decay_epochs)
            current_weight = 1.0 + (self.new_class_weight - 1.0) * decay_factor
        else:
            current_weight = self.new_class_weight

        if current_weight > 1.0:
            class_indices = torch.argmax(y, dim=1)
            for i, c in enumerate(class_indices):
                if int(c.item()) in new_classes_remapped:
                    weights[i] = current_weight

        return weights
