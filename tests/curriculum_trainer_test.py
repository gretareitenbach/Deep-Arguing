"""Integration tests for CurriculumTrainer."""

import pytest
import torch
from torch import nn

from deeparguing import GradualAACBR
from deeparguing.base_scores import ConstantBaseScore
from deeparguing.casebase_edge_weights import LearnedPartialOrder, Subtractor
from deeparguing.cli.loggers import DummyLogger, ExperimentLogger
from deeparguing.feature_extractor import MLPExtractor
from deeparguing.irrelevance_edge_weights import RegularIrrelevance
from deeparguing.semantics import ReluSemantics
from deeparguing.train import (
    CurriculumTrainer,
    FixedEpochsCurriculum,
    PerformanceBasedCurriculum,
    UniformDataSelector,
    WeightedDataSelector,
)


@pytest.fixture(autouse=True)
def setup_dummy_logger():
    """Setup a dummy logger for all tests."""
    logger = DummyLogger()
    ExperimentLogger.set_current(logger)
    yield
    ExperimentLogger._current = None


def create_simple_model(input_size: int = 4, num_classes: int = 3) -> GradualAACBR:
    """Creates a simple GradualAACBR model for testing."""
    feature_extractor = MLPExtractor(
        input_size=input_size,
        hidden_sizes=[8],
        output_size=1,
        batch_norm=False,
    )

    partial_order = LearnedPartialOrder(
        feature_extractors=[feature_extractor],
        comparison_func=Subtractor(temperature=1.0, activation=torch.sigmoid),
    )

    irrelevance = RegularIrrelevance(compute_partial_order=partial_order)

    base_score = ConstantBaseScore(constant=0.5, dim=1)

    semantics = ReluSemantics(max_iters=10)

    model = GradualAACBR(
        gradual_semantics=semantics,
        compute_base_score=base_score,
        irrelevance_edge_weights=irrelevance,
        casebase_edge_weights=partial_order,
        use_symmetric_attacks=False,
        defaults_not_attack=True,
        use_blockers=True,
        use_supports=False,
        dimensions=1,
    )

    return model


def create_test_data(
    num_samples_per_class: int = 20,
    num_classes: int = 3,
    input_size: int = 4,
):
    """Creates simple test data for curriculum learning."""
    X_list = []
    y_list = []

    for class_idx in range(num_classes):
        # Create samples centered around different points for each class
        class_center = torch.zeros(input_size)
        class_center[class_idx % input_size] = 1.0

        X_class = class_center + 0.1 * torch.randn(num_samples_per_class, input_size)
        y_class = torch.zeros(num_samples_per_class, num_classes)
        y_class[:, class_idx] = 1.0

        X_list.append(X_class)
        y_list.append(y_class)

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)

    return X, y


class TestCurriculumTrainerValidation:
    """Tests for CurriculumTrainer input validation."""

    def test_validate_epochs_insufficient(self):
        """Should raise error if epochs < epochs_per_class * num_classes."""
        curriculum = FixedEpochsCurriculum(
            class_order=[0, 1, 2],
            epochs_per_class=10,
            initial_classes=1,
        )
        selector = UniformDataSelector()
        trainer = CurriculumTrainer(
            curriculum_strategy=curriculum,
            data_selector=selector,
        )

        model = create_simple_model(num_classes=3)
        X, y = create_test_data(num_samples_per_class=10, num_classes=3)

        # Create casebase (2 samples per class)
        X_casebase = X[::10][:6]  # 6 samples
        y_casebase = y[::10][:6]

        # Create defaults
        X_default = X[:3]
        y_default = torch.eye(3).flip([0])

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # epochs=20 < 10 * 3 = 30
        with pytest.raises(ValueError, match="Total epochs.*must be >="):
            trainer.train(
                model=model,
                X_casebase=X_casebase,
                y_casebase=y_casebase,
                X_new_cases=X,
                y_new_cases=y,
                X_default=X_default,
                y_default=y_default,
                optimizer=optimizer,
                criterion=criterion,
                epochs=20,
                disable_tqdm=True,
            )

    def test_validate_scheduler_requires_step_per(self):
        """Should raise error if scheduler given without scheduler_step_per."""
        curriculum = FixedEpochsCurriculum(
            class_order=[0, 1],
            epochs_per_class=5,
            initial_classes=1,
        )
        selector = UniformDataSelector()
        trainer = CurriculumTrainer(
            curriculum_strategy=curriculum,
            data_selector=selector,
        )

        model = create_simple_model(num_classes=2)
        X, y = create_test_data(num_samples_per_class=10, num_classes=2)

        X_casebase = X[::10][:4]
        y_casebase = y[::10][:4]
        X_default = X[:2]
        y_default = torch.eye(2).flip([0])

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        criterion = nn.CrossEntropyLoss()

        with pytest.raises(ValueError, match="scheduler_step_per must be specified"):
            trainer.train(
                model=model,
                X_casebase=X_casebase,
                y_casebase=y_casebase,
                X_new_cases=X,
                y_new_cases=y,
                X_default=X_default,
                y_default=y_default,
                optimizer=optimizer,
                criterion=criterion,
                epochs=10,
                scheduler=scheduler,
                scheduler_step_per=None,
                disable_tqdm=True,
            )


class TestCurriculumTrainerBasicFunctionality:
    """Tests for basic CurriculumTrainer functionality."""

    def test_train_completes_with_fixed_epochs(self):
        """Training should complete successfully with FixedEpochsCurriculum."""
        curriculum = FixedEpochsCurriculum(
            class_order=[0, 1],
            epochs_per_class=3,
            initial_classes=1,
        )
        selector = UniformDataSelector()
        trainer = CurriculumTrainer(
            curriculum_strategy=curriculum,
            data_selector=selector,
        )

        model = create_simple_model(num_classes=2)
        X, y = create_test_data(num_samples_per_class=10, num_classes=2)

        X_casebase = X[::5][:4]  # 4 samples
        y_casebase = y[::5][:4]
        X_default = X[:2]
        y_default = torch.eye(2).flip([0])

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Should not raise
        trainer.train(
            model=model,
            X_casebase=X_casebase,
            y_casebase=y_casebase,
            X_new_cases=X,
            y_new_cases=y,
            X_default=X_default,
            y_default=y_default,
            optimizer=optimizer,
            criterion=criterion,
            epochs=6,  # 3 * 2 = 6 minimum
            disable_tqdm=True,
        )

    def test_train_with_weighted_selector(self):
        """Training should work with WeightedDataSelector."""
        curriculum = FixedEpochsCurriculum(
            class_order=[0, 1],
            epochs_per_class=2,
            initial_classes=1,
        )
        selector = WeightedDataSelector(new_class_weight=2.0, decay_epochs=3)
        trainer = CurriculumTrainer(
            curriculum_strategy=curriculum,
            data_selector=selector,
        )

        model = create_simple_model(num_classes=2)
        X, y = create_test_data(num_samples_per_class=10, num_classes=2)

        X_casebase = X[::5][:4]
        y_casebase = y[::5][:4]
        X_default = X[:2]
        y_default = torch.eye(2).flip([0])

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        trainer.train(
            model=model,
            X_casebase=X_casebase,
            y_casebase=y_casebase,
            X_new_cases=X,
            y_new_cases=y,
            X_default=X_default,
            y_default=y_default,
            optimizer=optimizer,
            criterion=criterion,
            epochs=4,
            disable_tqdm=True,
        )

    def test_train_with_performance_based_curriculum(self):
        """Training should work with PerformanceBasedCurriculum."""
        curriculum = PerformanceBasedCurriculum(
            class_order=[0, 1],
            accuracy_threshold=0.99,  # Very high, will use max_epochs
            min_epochs_per_class=1,
            max_epochs_per_class=2,
            initial_classes=1,
        )
        selector = UniformDataSelector()
        trainer = CurriculumTrainer(
            curriculum_strategy=curriculum,
            data_selector=selector,
        )

        model = create_simple_model(num_classes=2)
        X, y = create_test_data(num_samples_per_class=10, num_classes=2)

        X_casebase = X[::5][:4]
        y_casebase = y[::5][:4]
        X_default = X[:2]
        y_default = torch.eye(2).flip([0])

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        trainer.train(
            model=model,
            X_casebase=X_casebase,
            y_casebase=y_casebase,
            X_new_cases=X,
            y_new_cases=y,
            X_default=X_default,
            y_default=y_default,
            optimizer=optimizer,
            criterion=criterion,
            epochs=4,  # 2 * 2 = 4 max
            disable_tqdm=True,
        )

    def test_train_with_batch_size(self):
        """Training should work with mini-batches."""
        curriculum = FixedEpochsCurriculum(
            class_order=[0, 1],
            epochs_per_class=2,
            initial_classes=1,
        )
        selector = UniformDataSelector()
        trainer = CurriculumTrainer(
            curriculum_strategy=curriculum,
            data_selector=selector,
        )

        model = create_simple_model(num_classes=2)
        X, y = create_test_data(num_samples_per_class=20, num_classes=2)

        X_casebase = X[::10][:4]
        y_casebase = y[::10][:4]
        X_default = X[:2]
        y_default = torch.eye(2).flip([0])

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        trainer.train(
            model=model,
            X_casebase=X_casebase,
            y_casebase=y_casebase,
            X_new_cases=X,
            y_new_cases=y,
            X_default=X_default,
            y_default=y_default,
            optimizer=optimizer,
            criterion=criterion,
            epochs=4,
            batch_size=8,
            disable_tqdm=True,
        )


class TestCurriculumTrainerOptimizerReset:
    """Tests for optimizer reset functionality."""

    def test_optimizer_state_reset_on_advance(self):
        """Optimizer state should be cleared when reset_optimizer_on_advance=True."""
        curriculum = FixedEpochsCurriculum(
            class_order=[0, 1],
            epochs_per_class=2,
            initial_classes=1,
        )
        selector = UniformDataSelector()
        trainer = CurriculumTrainer(
            curriculum_strategy=curriculum,
            data_selector=selector,
            reset_optimizer_on_advance=True,
        )

        model = create_simple_model(num_classes=2)
        X, y = create_test_data(num_samples_per_class=10, num_classes=2)

        X_casebase = X[::5][:4]
        y_casebase = y[::5][:4]
        X_default = X[:2]
        y_default = torch.eye(2).flip([0])

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Train - optimizer state gets cleared when class is added
        trainer.train(
            model=model,
            X_casebase=X_casebase,
            y_casebase=y_casebase,
            X_new_cases=X,
            y_new_cases=y,
            X_default=X_default,
            y_default=y_default,
            optimizer=optimizer,
            criterion=criterion,
            epochs=4,
            disable_tqdm=True,
        )

        # After training completes, state should exist for remaining parameters
        # (this test mainly verifies the code path runs without error)


class TestCurriculumTrainerFilterDefaults:
    """Tests for default argument filtering."""

    def test_filter_defaults_correct_order(self):
        """Defaults should be filtered correctly based on class order."""
        curriculum = FixedEpochsCurriculum(
            class_order=[0, 1, 2],
            epochs_per_class=2,
            initial_classes=2,
        )
        selector = UniformDataSelector()
        trainer = CurriculumTrainer(
            curriculum_strategy=curriculum,
            data_selector=selector,
        )

        # y_default follows production convention (torch.unique pattern):
        # default[i] = class i
        X_default = torch.tensor([[0.0], [1.0], [2.0]])  # Values match class indices
        y_default = torch.eye(3)  # default[0]=class0, default[1]=class1, default[2]=class2

        # Active classes [0, 1] should select defaults for classes 0 and 1
        X_filtered, y_filtered = trainer._filter_defaults(
            X_default, y_default, active_classes=[0, 1]
        )

        assert X_filtered.shape[0] == 2
        # Sorted by class index: class 0 first, then class 1
        assert X_filtered[0].item() == 0.0  # default for class 0
        assert X_filtered[1].item() == 1.0  # default for class 1

        # y_filtered should be remapped eye matrix for 2 classes
        expected_y = torch.eye(2)
        assert torch.allclose(y_filtered, expected_y)

    def test_filter_defaults_flipped_convention(self):
        """Should also work with flipped y_default convention (used in some tests)."""
        curriculum = FixedEpochsCurriculum(
            class_order=[0, 1, 2],
            epochs_per_class=2,
            initial_classes=2,
        )
        selector = UniformDataSelector()
        trainer = CurriculumTrainer(
            curriculum_strategy=curriculum,
            data_selector=selector,
        )

        # y_default with flipped convention: default[0] = class 2, etc.
        X_default = torch.tensor([[2.0], [1.0], [0.0]])  # X values match class indices
        y_default = torch.eye(3).flip([0])  # [[0,0,1], [0,1,0], [1,0,0]]

        # Active classes [0, 1] should select defaults for classes 0 and 1
        X_filtered, y_filtered = trainer._filter_defaults(
            X_default, y_default, active_classes=[0, 1]
        )

        assert X_filtered.shape[0] == 2
        # Sorted by class index: class 0 first, then class 1
        # In flipped convention: default[2] has class 0 (X=0.0), default[1] has class 1 (X=1.0)
        assert X_filtered[0].item() == 0.0  # default for class 0
        assert X_filtered[1].item() == 1.0  # default for class 1

        # y_filtered should be remapped eye matrix for 2 classes
        expected_y = torch.eye(2)
        assert torch.allclose(y_filtered, expected_y)
