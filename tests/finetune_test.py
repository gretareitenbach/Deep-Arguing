"""Tests for ``deeparguing.casebase_edge_weights.finetune``'s ``protect_loss``
and its wiring into ``compute_losses`` -- the output-level backstop added
alongside ``preservation_loss`` (see ``protect_loss``'s docstring for why).

Unlike ``tests/qbaf_fixtures.py``'s shared graph (plain lookup-table
``base_score_fn``/``edge_weights_fn``/``irrelevance_fn``), these tests need a
model whose ``casebase_edge_weights`` is a ``LearnedPartialOrder`` shared
with a ``RegularIrrelevance`` -- the precondition
``assert_shares_partial_order`` (and therefore ``correction_loss``/
``preservation_loss``/``compute_losses``) requires. ``_make_model`` below
mirrors ``tests/curriculum_trainer_test.py::create_simple_model``'s
construction, with a second (trainable) feature extractor appended so
``TRAINABLE_FEATURE_EXTRACTOR_INDEX = 1`` has something real to select.
"""

import pytest
import torch

from deeparguing import GradualAACBR
from deeparguing.base_scores import ConstantBaseScore
from deeparguing.casebase_edge_weights import LearnedPartialOrder, Subtractor
from deeparguing.casebase_edge_weights.finetune import (compute_losses,
                                                         protect_loss)
from deeparguing.contest.core.contest import MARGIN, THRESHOLD, \
    _target_and_rival_batch
from deeparguing.contest.core.grae import (_batched_casebase_base_scores,
                                            _replay_default_strengths)
from deeparguing.feature_extractor import MLPExtractor
from deeparguing.irrelevance_edge_weights import RegularIrrelevance
from deeparguing.semantics import ReluSemantics


def _make_model(seed: int = 0) -> GradualAACBR:
    """6-case, 2-class casebase. ``d=1`` throughout (required by
    ``correction_loss``'s diagonal-extraction assumption)."""
    torch.manual_seed(seed)

    frozen_extractor = MLPExtractor(input_size=3, hidden_sizes=[], output_size=2, bias=False)
    for p in frozen_extractor.parameters():
        p.requires_grad_(False)
    trainable_extractor = MLPExtractor(input_size=2, hidden_sizes=[4], output_size=1)

    partial_order = LearnedPartialOrder(
        feature_extractors=[frozen_extractor, trainable_extractor],
        comparison_func=Subtractor(temperature=1.0, activation=torch.sigmoid),
    )
    irrelevance = RegularIrrelevance(compute_partial_order=partial_order)

    model = GradualAACBR(
        gradual_semantics=ReluSemantics(max_iters=10),
        compute_base_score=ConstantBaseScore(constant=0.5, dim=1),
        irrelevance_edge_weights=irrelevance,
        casebase_edge_weights=partial_order,
        use_symmetric_attacks=False,
        defaults_not_attack=True,
        use_blockers=True,
        use_supports=False,
        dimensions=1,
    )

    X_train = torch.randn(6, 3)
    y_train = torch.zeros(6, 2)
    y_train[:3, 0] = 1.0
    y_train[3:, 1] = 1.0
    X_default = torch.randn(2, 3)
    y_default = torch.eye(2)
    model.fit(X_train, y_train, X_default, y_default)
    return model


def test_protect_loss_empty_batch_returns_zero():
    model = _make_model()
    empty = model.X_train[:0]
    loss = protect_loss(model, empty, [], protect_margin=MARGIN)
    assert loss.item() == 0.0


def test_protect_loss_matches_hand_composed_hinge():
    """Cross-check ``protect_loss`` against its own documented building
    blocks composed by hand (same spirit as ``grae_test.py``'s analytic-vs-
    finite-difference cross-check): if it's wired correctly, calling
    ``protect_loss`` must reproduce exactly the same number as manually
    replaying ``_replay_default_strengths`` + ``_target_and_rival_batch``
    and applying the documented hinge formula.
    """
    model = _make_model()
    protect_samples = model.X_train[:2]
    with torch.no_grad():
        predicted = model(protect_samples).argmax(dim=-1)
    protect_target_classes = predicted.tolist()

    A = model.A.detach()
    with torch.no_grad():
        casebase_base_scores = _batched_casebase_base_scores(model, protect_samples.shape[0])
        new_cases_base_scores = model.compute_base_scores(protect_samples).unsqueeze(-1)
        irrelevance = model.irrelevance_edge_weights(protect_samples, model.X_train)
        strengths = _replay_default_strengths(
            model, A, -irrelevance, casebase_base_scores, new_cases_base_scores
        )
        target, _, rival = _target_and_rival_batch(strengths, protect_target_classes, THRESHOLD)
        achieved_margin = target - rival

    # Margin comfortably below what every sample already achieves: hinge is
    # exactly 0 for all of them.
    small_margin = achieved_margin.min().item() - 0.5
    loss_below = protect_loss(model, protect_samples, protect_target_classes, protect_margin=small_margin)
    assert loss_below.item() == pytest.approx(0.0, abs=1e-6)

    # Margin comfortably above: hinge must match the documented formula
    # applied to the *same* target/rival values, exactly.
    big_margin = 10.0
    expected_above = torch.clamp(big_margin - achieved_margin, min=0.0).mean().item()
    loss_above = protect_loss(model, protect_samples, protect_target_classes, protect_margin=big_margin)
    assert loss_above.item() == pytest.approx(expected_above, abs=1e-6)


def test_compute_losses_protect_lambda_zero_is_inert():
    """``protect_lambda=0.0`` must make ``combined`` exactly
    ``correction + lam * preservation``, not just numerically close -- a
    real, active (nonzero) protect hinge must be fully zeroed out by the
    lambda, mirroring ``batch_contest_test.py``'s
    ``test_batch_contest_protect_lambda_zero_matches_baseline_behavior``.
    """
    model = _make_model()
    new_cases = model.X_train[:2]
    casebase_items = model.X_train[2:4]
    targets = torch.full((2,), 0.5)
    frozen_raw_po = model.casebase_edge_weights(model.X_train, model.X_train).detach().clone()

    protect_samples = model.X_train[:2]
    with torch.no_grad():
        predicted = model(protect_samples).argmax(dim=-1)
    protect_target_classes = predicted.tolist()

    losses = compute_losses(
        model, new_cases, casebase_items, targets,
        model.X_train, frozen_raw_po, lam=1.0,
        protect_samples=protect_samples, protect_target_classes=protect_target_classes,
        protect_margin=10.0,  # deliberately unreachable -> a real, nonzero hinge
        protect_lambda=0.0,
    )

    assert losses.protect.item() > 0.0  # sanity: the hinge really is active
    assert losses.combined.item() == pytest.approx(
        (losses.correction + 1.0 * losses.preservation).item(), abs=1e-6
    )
