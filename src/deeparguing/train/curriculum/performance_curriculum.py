from typing import override

from deeparguing.train.curriculum.curriculum_strategy import CurriculumStrategy


class PerformanceBasedCurriculum(CurriculumStrategy):
    """
    Introduces new classes when accuracy threshold is met.

    Parameters
    ----------
    class_order : list[int]
        Order in which classes are introduced
    accuracy_threshold : float
        Minimum accuracy on current classes before advancing
    min_epochs_per_class : int
        Minimum epochs in phase before advancing (even if threshold met)
    max_epochs_per_class : int
        Maximum epochs before forced advancement (prevents getting stuck)
    initial_classes : int
        Number of classes to start with (default: 1)

    Notes
    -----
    Advances when ANY condition is met:
    1. accuracy >= threshold AND phase_epoch >= min_epochs
    2. phase_epoch >= max_epochs (forced)
    """

    def __init__(
        self,
        class_order: list[int],
        accuracy_threshold: float = 0.85,
        min_epochs_per_class: int = 5,
        max_epochs_per_class: int = 50,
        initial_classes: int = 1,
    ):
        if not 0.0 < accuracy_threshold <= 1.0:
            raise ValueError("accuracy_threshold must be in (0, 1]")
        if min_epochs_per_class < 1:
            raise ValueError("min_epochs_per_class must be >= 1")
        if max_epochs_per_class < min_epochs_per_class:
            raise ValueError("max_epochs_per_class must be >= min_epochs_per_class")
        if initial_classes < 1:
            raise ValueError("initial_classes must be >= 1")
        if initial_classes > len(class_order):
            raise ValueError(
                f"initial_classes ({initial_classes}) cannot exceed "
                f"len(class_order) ({len(class_order)})"
            )

        self._class_order = class_order
        self._accuracy_threshold = accuracy_threshold
        self._min_epochs = min_epochs_per_class
        self._max_epochs = max_epochs_per_class
        self._initial_classes = initial_classes
        self._current_index = initial_classes
        self._newly_added: list[int] = list(class_order[:initial_classes])

    @override
    def get_active_classes(self) -> list[int]:
        return list(self._class_order[: self._current_index])

    @override
    def get_newly_added_classes(self) -> list[int]:
        return list(self._newly_added)

    @override
    def should_advance(
        self,
        epoch: int,
        phase_epoch: int,
        curriculum_accuracy: float | None = None,
    ) -> bool:
        if self.is_complete():
            return False

        # Forced advance at max epochs
        if phase_epoch >= self._max_epochs:
            return True

        # Performance-based advance
        if (
            curriculum_accuracy is not None
            and curriculum_accuracy >= self._accuracy_threshold
            and phase_epoch >= self._min_epochs
        ):
            return True

        return False

    @override
    def advance(self) -> list[int] | None:
        if self.is_complete():
            return None
        new_class = self._class_order[self._current_index]
        self._current_index += 1
        self._newly_added = [new_class]
        return list(self._newly_added)

    @override
    def is_complete(self) -> bool:
        return self._current_index >= len(self._class_order)

    @override
    def reset(self) -> None:
        self._current_index = self._initial_classes
        self._newly_added = list(self._class_order[: self._initial_classes])

    @property
    @override
    def total_classes(self) -> int:
        return len(self._class_order)

    @property
    @override
    def epochs_per_class(self) -> int | None:
        # Return max_epochs for validation (worst case)
        return self._max_epochs
