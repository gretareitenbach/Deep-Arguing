from typing import override

from deeparguing.train.curriculum.curriculum_strategy import CurriculumStrategy


class FixedEpochsCurriculum(CurriculumStrategy):
    """
    Introduces new classes after a fixed number of epochs.

    Parameters
    ----------
    class_order : list[int]
        Order in which classes are introduced. E.g., [8, 9, 0, 1, ...]
        means class 8 is introduced first, then class 9, etc.
    epochs_per_class : int
        Number of epochs to train before introducing next class
    initial_classes : int
        Number of classes to start with (default: 1)

    Example
    -------
    >>> curriculum = FixedEpochsCurriculum(
    ...     class_order=[8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
    ...     epochs_per_class=10,
    ...     initial_classes=2
    ... )
    >>> curriculum.get_active_classes()
    [8, 9]
    >>> curriculum.get_newly_added_classes()
    [8, 9]
    >>> # After 10 phase_epochs...
    >>> curriculum.should_advance(epoch=10, phase_epoch=10)
    True
    >>> curriculum.advance()
    [0]
    >>> curriculum.get_active_classes()
    [8, 9, 0]
    """

    def __init__(
        self,
        class_order: list[int],
        epochs_per_class: int,
        initial_classes: int = 1,
    ):
        if initial_classes < 1:
            raise ValueError("initial_classes must be >= 1")
        if initial_classes > len(class_order):
            raise ValueError(
                f"initial_classes ({initial_classes}) cannot exceed "
                f"len(class_order) ({len(class_order)})"
            )
        if epochs_per_class < 1:
            raise ValueError("epochs_per_class must be >= 1")

        self._class_order = class_order
        self._epochs_per_class = epochs_per_class
        self._initial_classes = initial_classes
        self._current_index = initial_classes  # Points to next class to add
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
        return phase_epoch >= self._epochs_per_class

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
        return self._epochs_per_class
