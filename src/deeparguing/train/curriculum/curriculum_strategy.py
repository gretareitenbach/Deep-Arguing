from abc import ABC, abstractmethod


class CurriculumStrategy(ABC):
    """
    Abstract strategy for controlling class introduction schedule.

    Implementations define when and which classes are introduced
    during curriculum learning.
    """

    @abstractmethod
    def get_active_classes(self) -> list[int]:
        """Returns list of currently active class indices."""
        pass

    @abstractmethod
    def get_newly_added_classes(self) -> list[int]:
        """
        Returns classes added in the most recent advance() call.
        Returns initial classes if advance() hasn't been called yet.
        """
        pass

    @abstractmethod
    def should_advance(
        self,
        epoch: int,
        phase_epoch: int,
        curriculum_accuracy: float | None = None,
    ) -> bool:
        """
        Determines if curriculum should advance to next phase.

        Parameters
        ----------
        epoch : int
            Total epochs elapsed since training started
        phase_epoch : int
            Epochs since last class introduction
        curriculum_accuracy : float | None
            Accuracy on currently active classes (for performance-based)

        Returns
        -------
        bool
            True if should introduce next class(es)
        """
        pass

    @abstractmethod
    def advance(self) -> list[int] | None:
        """
        Advances curriculum to include next class(es).

        Returns
        -------
        list[int] | None
            Newly added class indices, or None if all classes already active
        """
        pass

    @abstractmethod
    def is_complete(self) -> bool:
        """Returns True if all classes have been introduced."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets curriculum to initial state."""
        pass

    @property
    @abstractmethod
    def total_classes(self) -> int:
        """Total number of classes in the curriculum."""
        pass

    @property
    @abstractmethod
    def epochs_per_class(self) -> int | None:
        """
        Returns epochs_per_class if fixed, None if performance-based.
        Used for epoch validation.
        """
        pass
