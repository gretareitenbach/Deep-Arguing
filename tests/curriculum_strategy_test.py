"""Unit tests for curriculum strategy implementations."""

import pytest

from deeparguing.train.curriculum import (
    FixedEpochsCurriculum,
    PerformanceBasedCurriculum,
)


class TestFixedEpochsCurriculum:
    """Tests for FixedEpochsCurriculum."""

    def test_initial_classes(self):
        """Initial active classes should match initial_classes parameter."""
        curriculum = FixedEpochsCurriculum(
            class_order=[0, 1, 2, 3, 4],
            epochs_per_class=10,
            initial_classes=2,
        )
        assert curriculum.get_active_classes() == [0, 1]
        assert curriculum.get_newly_added_classes() == [0, 1]

    def test_initial_classes_custom_order(self):
        """Should respect custom class order."""
        curriculum = FixedEpochsCurriculum(
            class_order=[8, 9, 0, 1, 2],
            epochs_per_class=10,
            initial_classes=2,
        )
        assert curriculum.get_active_classes() == [8, 9]
        assert curriculum.get_newly_added_classes() == [8, 9]

    def test_should_advance_before_threshold(self):
        """Should not advance before epochs_per_class epochs."""
        curriculum = FixedEpochsCurriculum(
            class_order=[0, 1, 2],
            epochs_per_class=10,
            initial_classes=1,
        )
        # Before threshold
        assert not curriculum.should_advance(epoch=5, phase_epoch=5)
        assert not curriculum.should_advance(epoch=9, phase_epoch=9)

    def test_should_advance_at_threshold(self):
        """Should advance at exactly epochs_per_class epochs."""
        curriculum = FixedEpochsCurriculum(
            class_order=[0, 1, 2],
            epochs_per_class=10,
            initial_classes=1,
        )
        assert curriculum.should_advance(epoch=10, phase_epoch=10)

    def test_should_advance_after_threshold(self):
        """Should advance after epochs_per_class epochs."""
        curriculum = FixedEpochsCurriculum(
            class_order=[0, 1, 2],
            epochs_per_class=10,
            initial_classes=1,
        )
        assert curriculum.should_advance(epoch=15, phase_epoch=15)

    def test_advance_returns_new_class(self):
        """advance() should return the newly added class."""
        curriculum = FixedEpochsCurriculum(
            class_order=[0, 1, 2, 3],
            epochs_per_class=10,
            initial_classes=1,
        )
        assert curriculum.get_active_classes() == [0]

        new_classes = curriculum.advance()
        assert new_classes == [1]
        assert curriculum.get_active_classes() == [0, 1]
        assert curriculum.get_newly_added_classes() == [1]

    def test_advance_multiple_times(self):
        """Should correctly advance through all classes."""
        curriculum = FixedEpochsCurriculum(
            class_order=[8, 9, 0],
            epochs_per_class=5,
            initial_classes=1,
        )
        assert curriculum.get_active_classes() == [8]

        curriculum.advance()
        assert curriculum.get_active_classes() == [8, 9]

        curriculum.advance()
        assert curriculum.get_active_classes() == [8, 9, 0]

    def test_is_complete(self):
        """is_complete() should return True when all classes are added."""
        curriculum = FixedEpochsCurriculum(
            class_order=[0, 1],
            epochs_per_class=5,
            initial_classes=1,
        )
        assert not curriculum.is_complete()

        curriculum.advance()
        assert curriculum.is_complete()

    def test_advance_when_complete_returns_none(self):
        """advance() should return None when curriculum is complete."""
        curriculum = FixedEpochsCurriculum(
            class_order=[0, 1],
            epochs_per_class=5,
            initial_classes=2,
        )
        assert curriculum.is_complete()
        assert curriculum.advance() is None

    def test_should_advance_when_complete(self):
        """should_advance() should return False when complete."""
        curriculum = FixedEpochsCurriculum(
            class_order=[0, 1],
            epochs_per_class=5,
            initial_classes=2,
        )
        assert not curriculum.should_advance(epoch=100, phase_epoch=100)

    def test_reset(self):
        """reset() should restore initial state."""
        curriculum = FixedEpochsCurriculum(
            class_order=[0, 1, 2, 3],
            epochs_per_class=5,
            initial_classes=2,
        )
        curriculum.advance()
        curriculum.advance()
        assert curriculum.get_active_classes() == [0, 1, 2, 3]

        curriculum.reset()
        assert curriculum.get_active_classes() == [0, 1]
        assert curriculum.get_newly_added_classes() == [0, 1]
        assert not curriculum.is_complete()

    def test_total_classes(self):
        """total_classes should return length of class_order."""
        curriculum = FixedEpochsCurriculum(
            class_order=[0, 1, 2, 3, 4],
            epochs_per_class=5,
            initial_classes=1,
        )
        assert curriculum.total_classes == 5

    def test_epochs_per_class_property(self):
        """epochs_per_class property should return the configured value."""
        curriculum = FixedEpochsCurriculum(
            class_order=[0, 1, 2],
            epochs_per_class=15,
            initial_classes=1,
        )
        assert curriculum.epochs_per_class == 15

    def test_validation_initial_classes_too_small(self):
        """Should raise error if initial_classes < 1."""
        with pytest.raises(ValueError, match="initial_classes must be >= 1"):
            FixedEpochsCurriculum(
                class_order=[0, 1, 2],
                epochs_per_class=5,
                initial_classes=0,
            )

    def test_validation_initial_classes_too_large(self):
        """Should raise error if initial_classes > len(class_order)."""
        with pytest.raises(ValueError, match="initial_classes.*cannot exceed"):
            FixedEpochsCurriculum(
                class_order=[0, 1, 2],
                epochs_per_class=5,
                initial_classes=5,
            )

    def test_validation_epochs_per_class_too_small(self):
        """Should raise error if epochs_per_class < 1."""
        with pytest.raises(ValueError, match="epochs_per_class must be >= 1"):
            FixedEpochsCurriculum(
                class_order=[0, 1, 2],
                epochs_per_class=0,
                initial_classes=1,
            )


class TestPerformanceBasedCurriculum:
    """Tests for PerformanceBasedCurriculum."""

    def test_initial_classes(self):
        """Initial active classes should match initial_classes parameter."""
        curriculum = PerformanceBasedCurriculum(
            class_order=[0, 1, 2, 3, 4],
            accuracy_threshold=0.85,
            initial_classes=2,
        )
        assert curriculum.get_active_classes() == [0, 1]
        assert curriculum.get_newly_added_classes() == [0, 1]

    def test_should_advance_performance_threshold_met(self):
        """Should advance when accuracy threshold is met and min epochs passed."""
        curriculum = PerformanceBasedCurriculum(
            class_order=[0, 1, 2],
            accuracy_threshold=0.8,
            min_epochs_per_class=5,
            max_epochs_per_class=50,
            initial_classes=1,
        )
        # Not enough epochs yet
        assert not curriculum.should_advance(
            epoch=3, phase_epoch=3, curriculum_accuracy=0.9
        )

        # Enough epochs and threshold met
        assert curriculum.should_advance(
            epoch=5, phase_epoch=5, curriculum_accuracy=0.85
        )

    def test_should_not_advance_accuracy_below_threshold(self):
        """Should not advance if accuracy is below threshold."""
        curriculum = PerformanceBasedCurriculum(
            class_order=[0, 1, 2],
            accuracy_threshold=0.85,
            min_epochs_per_class=5,
            max_epochs_per_class=50,
            initial_classes=1,
        )
        assert not curriculum.should_advance(
            epoch=10, phase_epoch=10, curriculum_accuracy=0.7
        )

    def test_should_advance_forced_at_max_epochs(self):
        """Should force advance at max_epochs_per_class."""
        curriculum = PerformanceBasedCurriculum(
            class_order=[0, 1, 2],
            accuracy_threshold=0.99,  # Very high threshold
            min_epochs_per_class=5,
            max_epochs_per_class=20,
            initial_classes=1,
        )
        # Below max, low accuracy
        assert not curriculum.should_advance(
            epoch=15, phase_epoch=15, curriculum_accuracy=0.5
        )

        # At max, should force advance
        assert curriculum.should_advance(
            epoch=20, phase_epoch=20, curriculum_accuracy=0.5
        )

    def test_should_advance_none_accuracy(self):
        """Should handle None accuracy (only forced advance at max)."""
        curriculum = PerformanceBasedCurriculum(
            class_order=[0, 1, 2],
            accuracy_threshold=0.8,
            min_epochs_per_class=5,
            max_epochs_per_class=20,
            initial_classes=1,
        )
        # No accuracy provided, not at max
        assert not curriculum.should_advance(
            epoch=10, phase_epoch=10, curriculum_accuracy=None
        )

        # No accuracy provided, at max
        assert curriculum.should_advance(
            epoch=20, phase_epoch=20, curriculum_accuracy=None
        )

    def test_advance_returns_new_class(self):
        """advance() should return the newly added class."""
        curriculum = PerformanceBasedCurriculum(
            class_order=[5, 6, 7],
            accuracy_threshold=0.8,
            initial_classes=1,
        )
        assert curriculum.get_active_classes() == [5]

        new_classes = curriculum.advance()
        assert new_classes == [6]
        assert curriculum.get_active_classes() == [5, 6]

    def test_is_complete(self):
        """is_complete() should return True when all classes added."""
        curriculum = PerformanceBasedCurriculum(
            class_order=[0, 1],
            accuracy_threshold=0.8,
            initial_classes=1,
        )
        assert not curriculum.is_complete()

        curriculum.advance()
        assert curriculum.is_complete()

    def test_reset(self):
        """reset() should restore initial state."""
        curriculum = PerformanceBasedCurriculum(
            class_order=[0, 1, 2],
            accuracy_threshold=0.8,
            initial_classes=1,
        )
        curriculum.advance()
        curriculum.advance()
        assert curriculum.is_complete()

        curriculum.reset()
        assert curriculum.get_active_classes() == [0]
        assert not curriculum.is_complete()

    def test_epochs_per_class_returns_max(self):
        """epochs_per_class should return max_epochs for validation."""
        curriculum = PerformanceBasedCurriculum(
            class_order=[0, 1, 2],
            accuracy_threshold=0.8,
            min_epochs_per_class=5,
            max_epochs_per_class=30,
            initial_classes=1,
        )
        assert curriculum.epochs_per_class == 30

    def test_validation_accuracy_threshold_bounds(self):
        """Should raise error for invalid accuracy_threshold."""
        with pytest.raises(ValueError, match="accuracy_threshold must be in"):
            PerformanceBasedCurriculum(
                class_order=[0, 1],
                accuracy_threshold=0.0,
            )
        with pytest.raises(ValueError, match="accuracy_threshold must be in"):
            PerformanceBasedCurriculum(
                class_order=[0, 1],
                accuracy_threshold=1.5,
            )

    def test_validation_min_epochs(self):
        """Should raise error if min_epochs_per_class < 1."""
        with pytest.raises(ValueError, match="min_epochs_per_class must be >= 1"):
            PerformanceBasedCurriculum(
                class_order=[0, 1],
                accuracy_threshold=0.8,
                min_epochs_per_class=0,
            )

    def test_validation_max_less_than_min(self):
        """Should raise error if max < min epochs."""
        with pytest.raises(ValueError, match="max_epochs_per_class must be >= min"):
            PerformanceBasedCurriculum(
                class_order=[0, 1],
                accuracy_threshold=0.8,
                min_epochs_per_class=10,
                max_epochs_per_class=5,
            )

    def test_accuracy_threshold_edge_case_exact(self):
        """Should advance when accuracy exactly equals threshold."""
        curriculum = PerformanceBasedCurriculum(
            class_order=[0, 1, 2],
            accuracy_threshold=0.85,
            min_epochs_per_class=5,
            initial_classes=1,
        )
        assert curriculum.should_advance(
            epoch=5, phase_epoch=5, curriculum_accuracy=0.85
        )
