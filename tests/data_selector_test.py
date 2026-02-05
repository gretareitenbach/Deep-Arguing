"""Unit tests for data selector implementations."""

import pytest
import torch

from deeparguing.train.curriculum import (
    UniformDataSelector,
    WeightedDataSelector,
)


class TestUniformDataSelector:
    """Tests for UniformDataSelector."""

    def test_filter_by_classes_single_class(self):
        """Should filter to only include specified class."""
        selector = UniformDataSelector()

        # 6 samples, 3 classes (one-hot encoded)
        X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y = torch.tensor(
            [
                [1, 0, 0],  # class 0
                [0, 1, 0],  # class 1
                [0, 0, 1],  # class 2
                [1, 0, 0],  # class 0
                [0, 1, 0],  # class 1
                [0, 0, 1],  # class 2
            ]
        )

        X_filtered, y_filtered = selector.filter_by_classes(X, y, active_classes=[0])

        assert X_filtered.shape[0] == 2
        assert torch.all(X_filtered == torch.tensor([[1.0], [4.0]]))

    def test_filter_by_classes_multiple_classes(self):
        """Should filter to include multiple specified classes."""
        selector = UniformDataSelector()

        X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y = torch.tensor(
            [
                [1, 0, 0],  # class 0
                [0, 1, 0],  # class 1
                [0, 0, 1],  # class 2
                [1, 0, 0],  # class 0
                [0, 1, 0],  # class 1
                [0, 0, 1],  # class 2
            ]
        )

        X_filtered, y_filtered = selector.filter_by_classes(X, y, active_classes=[0, 2])

        assert X_filtered.shape[0] == 4
        assert torch.all(X_filtered == torch.tensor([[1.0], [3.0], [4.0], [6.0]]))

    def test_filter_by_classes_remaps_labels(self):
        """Should remap labels to the filtered class space."""
        selector = UniformDataSelector()

        X = torch.tensor([[1.0], [2.0], [3.0]])
        y = torch.tensor(
            [
                [1, 0, 0],  # class 0
                [0, 1, 0],  # class 1
                [0, 0, 1],  # class 2
            ]
        )

        # Filter to class 1 only - should remap to single-class space
        X_filtered, y_filtered = selector.filter_by_classes(X, y, active_classes=[1])

        assert y_filtered.shape == (1, 1)  # Only 1 active class
        assert torch.all(y_filtered == torch.tensor([[1.0]]))  # Remapped one-hot

        # Filter to classes 0 and 2 - should remap to 2-class space
        X_filtered, y_filtered = selector.filter_by_classes(X, y, active_classes=[0, 2])

        assert y_filtered.shape == (2, 2)  # 2 active classes
        # Class 0 -> index 0, Class 2 -> index 1 (sorted order)
        expected_y = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        assert torch.all(y_filtered == expected_y)

    def test_get_sample_weights_returns_ones(self):
        """Should return uniform weights of 1.0."""
        selector = UniformDataSelector()

        y = torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )

        weights = selector.get_sample_weights(
            y, active_classes=[0, 1, 2], new_classes=[2], phase_epoch=0
        )

        assert weights.shape == (3,)
        assert torch.all(weights == 1.0)


class TestWeightedDataSelector:
    """Tests for WeightedDataSelector."""

    def test_filter_by_classes(self):
        """Should filter correctly (same as UniformDataSelector)."""
        selector = WeightedDataSelector(new_class_weight=2.0, decay_epochs=5)

        X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        y = torch.tensor(
            [
                [1, 0],  # class 0
                [0, 1],  # class 1
                [1, 0],  # class 0
                [0, 1],  # class 1
            ]
        )

        X_filtered, y_filtered = selector.filter_by_classes(X, y, active_classes=[0])

        assert X_filtered.shape[0] == 2
        assert torch.all(X_filtered == torch.tensor([[1.0], [3.0]]))

    def test_get_sample_weights_new_class_weighted(self):
        """New class samples should have higher weight."""
        selector = WeightedDataSelector(new_class_weight=2.0, decay_epochs=0)

        y = torch.tensor(
            [
                [1, 0, 0],  # class 0 (old)
                [0, 1, 0],  # class 1 (old)
                [0, 0, 1],  # class 2 (new)
                [0, 0, 1],  # class 2 (new)
            ]
        )

        weights = selector.get_sample_weights(
            y, active_classes=[0, 1, 2], new_classes=[2], phase_epoch=0
        )

        assert weights[0] == 1.0  # class 0
        assert weights[1] == 1.0  # class 1
        assert weights[2] == 2.0  # class 2 (new)
        assert weights[3] == 2.0  # class 2 (new)

    def test_get_sample_weights_decay(self):
        """Weights should decay over epochs."""
        selector = WeightedDataSelector(new_class_weight=3.0, decay_epochs=6)

        y = torch.tensor(
            [
                [1, 0],  # class 0 (old)
                [0, 1],  # class 1 (new)
            ]
        )

        # At phase_epoch=0, weight should be 3.0
        weights_0 = selector.get_sample_weights(
            y, active_classes=[0, 1], new_classes=[1], phase_epoch=0
        )
        assert weights_0[1] == pytest.approx(3.0)

        # At phase_epoch=3, weight should be 2.0 (halfway decay)
        weights_3 = selector.get_sample_weights(
            y, active_classes=[0, 1], new_classes=[1], phase_epoch=3
        )
        assert weights_3[1] == pytest.approx(2.0)

        # At phase_epoch=6, weight should be 1.0 (fully decayed)
        weights_6 = selector.get_sample_weights(
            y, active_classes=[0, 1], new_classes=[1], phase_epoch=6
        )
        assert weights_6[1] == pytest.approx(1.0)

        # At phase_epoch=10, weight should still be 1.0
        weights_10 = selector.get_sample_weights(
            y, active_classes=[0, 1], new_classes=[1], phase_epoch=10
        )
        assert weights_10[1] == pytest.approx(1.0)

    def test_get_sample_weights_no_decay(self):
        """With decay_epochs=0, weight should remain constant."""
        selector = WeightedDataSelector(new_class_weight=2.5, decay_epochs=0)

        y = torch.tensor([[0, 1]])  # class 1 (new)

        weights_0 = selector.get_sample_weights(
            y, active_classes=[0, 1], new_classes=[1], phase_epoch=0
        )
        weights_10 = selector.get_sample_weights(
            y, active_classes=[0, 1], new_classes=[1], phase_epoch=10
        )

        assert weights_0[0] == pytest.approx(2.5)
        assert weights_10[0] == pytest.approx(2.5)

    def test_get_sample_weights_no_new_classes(self):
        """Should return uniform weights when no new classes."""
        selector = WeightedDataSelector(new_class_weight=3.0, decay_epochs=5)

        y = torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )

        weights = selector.get_sample_weights(
            y, active_classes=[0, 1, 2], new_classes=[], phase_epoch=0
        )

        assert torch.all(weights == 1.0)

    def test_get_sample_weights_multiple_new_classes(self):
        """Should weight multiple new classes."""
        selector = WeightedDataSelector(new_class_weight=2.0, decay_epochs=0)

        y = torch.tensor(
            [
                [1, 0, 0, 0],  # class 0 (old)
                [0, 1, 0, 0],  # class 1 (new)
                [0, 0, 1, 0],  # class 2 (new)
                [0, 0, 0, 1],  # class 3 (old)
            ]
        )

        weights = selector.get_sample_weights(
            y, active_classes=[0, 1, 2, 3], new_classes=[1, 2], phase_epoch=0
        )

        assert weights[0] == 1.0  # class 0
        assert weights[1] == 2.0  # class 1 (new)
        assert weights[2] == 2.0  # class 2 (new)
        assert weights[3] == 1.0  # class 3

    def test_validation_new_class_weight_too_small(self):
        """Should raise error if new_class_weight < 1.0."""
        with pytest.raises(ValueError, match="new_class_weight must be >= 1.0"):
            WeightedDataSelector(new_class_weight=0.5, decay_epochs=5)

    def test_validation_decay_epochs_negative(self):
        """Should raise error if decay_epochs < 0."""
        with pytest.raises(ValueError, match="decay_epochs must be >= 0"):
            WeightedDataSelector(new_class_weight=2.0, decay_epochs=-1)

    def test_weight_exactly_one_no_boost(self):
        """With new_class_weight=1.0, should return uniform weights."""
        selector = WeightedDataSelector(new_class_weight=1.0, decay_epochs=5)

        y = torch.tensor([[0, 1]])

        weights = selector.get_sample_weights(
            y, active_classes=[0, 1], new_classes=[1], phase_epoch=0
        )

        assert weights[0] == 1.0
