"""Tests for ``deeparguing.casebase_edge_weights.finetune``'s ``protect_loss``
and ``casebase_correction_loss``, and their wiring into ``compute_losses``
(see each function's docstring for what it does and why).

Unlike ``tests/qbaf_fixtures.py``'s shared graph (plain lookup-table
``base_score_fn``/``edge_weights_fn``/``irrelevance_fn``), these tests need a
model whose ``casebase_edge_weights`` is a ``LearnedPartialOrder`` shared
with a ``RegularIrrelevance`` -- the precondition
``assert_shares_partial_order`` (and therefore ``correction_loss``/
``compute_losses``) requires. ``_make_model`` below
mirrors ``tests/curriculum_trainer_test.py::create_simple_model``'s
construction, with a second (trainable) feature extractor appended so
``TRAINABLE_FEATURE_EXTRACTOR_INDEX = 1`` has something real to select.
"""

import pytest
import torch

from deeparguing import GradualAACBR
from deeparguing.base_scores import ConstantBaseScore
from deeparguing.casebase_edge_weights import LearnedPartialOrder, Subtractor
from deeparguing.casebase_edge_weights.finetune import (
    CasebaseCorrectionBatch, TRAINABLE_FEATURE_EXTRACTOR_INDEX,
    casebase_correction_loss, compute_losses, freeze_all_except_trainable,
    protect_loss, trainable_parameters)
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
    """``protect_lambda=0.0`` must make ``combined`` exactly ``correction``,
    not just numerically close -- a real, active (nonzero) protect hinge must
    be fully zeroed out by the lambda, mirroring ``batch_contest_test.py``'s
    ``test_batch_contest_protect_lambda_zero_matches_baseline_behavior``.
    """
    model = _make_model()
    new_cases = model.X_train[:2]
    casebase_items = model.X_train[2:4]
    targets = torch.full((2,), 0.5)

    protect_samples = model.X_train[:2]
    with torch.no_grad():
        predicted = model(protect_samples).argmax(dim=-1)
    protect_target_classes = predicted.tolist()

    losses = compute_losses(
        model, new_cases, casebase_items, targets,
        protect_samples=protect_samples, protect_target_classes=protect_target_classes,
        protect_margin=10.0,  # deliberately unreachable -> a real, nonzero hinge
        protect_lambda=0.0,
    )

    assert losses.protect.item() > 0.0  # sanity: the hinge really is active
    assert losses.combined.item() == pytest.approx(losses.correction.item(), abs=1e-6)


def _casebase_batch(model: GradualAACBR, n_edges: int, target_offset: float) -> CasebaseCorrectionBatch:
    """A batch over the first ``n_edges`` currently-live (nonzero) entries of
    ``model.A`` -- with ``use_supports=False`` (see ``_make_model``), every
    nonzero entry is guaranteed to be from the differing-label, non-default-
    source attacks channel, i.e. reachable (see
    ``build_casebase_finetune_dataset.py``'s module docstring for the
    unreachable case this sidesteps). Targets are the current values offset
    by ``target_offset``, so they're neither trivially already-met (offset
    ``0``) nor absurdly far (mirroring ``protect_loss``'s tests' pattern of
    picking a deliberately-meaningful, not arbitrary, target).
    """
    X_casebase, y_casebase, X_default, y_default = model.casebase_and_defaults()
    nonzero = (model.A != 0).nonzero(as_tuple=False)
    assert nonzero.shape[0] >= n_edges, "fixture casebase has fewer live edges than the test needs"
    picked = nonzero[:n_edges]
    source_idx, target_idx, dim_idx = picked[:, 0], picked[:, 1], picked[:, 2]
    targets = model.A[source_idx, target_idx, dim_idx].detach() + target_offset
    return CasebaseCorrectionBatch(
        X_casebase=X_casebase, y_casebase=y_casebase, X_default=X_default, y_default=y_default,
        source_idx=source_idx, target_idx=target_idx, dim_idx=dim_idx, targets=targets,
    )


def test_casebase_correction_loss_empty_batch_returns_zero():
    model = _make_model()
    X_casebase, y_casebase, X_default, y_default = model.casebase_and_defaults()
    empty = torch.empty(0, dtype=torch.long)
    batch = CasebaseCorrectionBatch(
        X_casebase=X_casebase, y_casebase=y_casebase, X_default=X_default, y_default=y_default,
        source_idx=empty, target_idx=empty, dim_idx=empty, targets=torch.empty(0),
    )
    loss = casebase_correction_loss(model, batch)
    assert loss.item() == 0.0


def test_casebase_correction_loss_matches_manual_refit():
    """Cross-check against the documented mechanism composed by hand: since
    ``casebase_correction_loss`` re-fits ``model.A`` and reads off the
    touched entries, manually calling ``model.fit()`` with the same
    (unchanged) weights and reading the same entries must give exactly the
    same MSE -- ``fit()`` is deterministic given fixed weights.
    """
    model = _make_model()
    batch = _casebase_batch(model, n_edges=3, target_offset=0.3)

    X_casebase, y_casebase, X_default, y_default = model.casebase_and_defaults()
    with torch.no_grad():
        model.fit(X_casebase, y_casebase, X_default, y_default)
    manual_predicted = model.A[batch.source_idx, batch.target_idx, batch.dim_idx]
    expected = ((manual_predicted - batch.targets) ** 2).mean().item()

    loss = casebase_correction_loss(model, batch)
    assert loss.item() == pytest.approx(expected, abs=1e-6)


def test_casebase_correction_loss_gradient_reaches_feature_weights():
    """The load-bearing assumption of the whole casebase-correction feature:
    ``model.A``'s construction (label masking, the minimality/blocking
    product, negation, ``B``) must be genuinely differentiable w.r.t.
    ``feature_weights_1``'s parameters, not just formally callable under
    autograd. ``_compute_blocked_product`` is a continuous t-norm product
    (not a hard argmin), so this should hold -- this test is the regression
    guard for that fact, so a future change to the blocking mechanism (e.g.
    swapping in a hard top-1 selection) gets caught here, not discovered as
    a silently-inert loss during a real fine-tune run.
    """
    model = _make_model()
    freeze_all_except_trainable(model)
    extractor = model.casebase_edge_weights.feature_extractors[TRAINABLE_FEATURE_EXTRACTOR_INDEX]

    batch = _casebase_batch(model, n_edges=3, target_offset=5.0)  # far off -> definitely nonzero loss
    loss = casebase_correction_loss(model, batch)
    assert loss.item() > 0.0
    loss.backward()

    grads = [p.grad for p in extractor.parameters()]
    assert any(g is not None and g.abs().sum().item() > 0 for g in grads), (
        "no gradient reached feature_weights_1 through casebase_correction_loss's "
        "model.fit() call -- the differentiability this feature depends on broke."
    )


def test_casebase_correction_loss_reduces_with_gradient_descent():
    """End-to-end proof the gradient signal is not just nonzero but usable:
    a few Adam steps against a fixed target must actually reduce the loss."""
    model = _make_model()
    freeze_all_except_trainable(model)

    batch = _casebase_batch(model, n_edges=2, target_offset=0.3)
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=0.05)

    first_loss = None
    last_loss = None
    for _ in range(15):
        loss = casebase_correction_loss(model, batch)
        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert last_loss < first_loss


def test_compute_losses_casebase_lambda_zero_is_inert():
    """``casebase_lambda=0.0`` must make ``combined`` exactly
    ``correction + protect_lambda * protect``, not just numerically close --
    mirroring ``test_compute_losses_protect_lambda_zero_is_inert``. Also
    checks the real (nonzero) ``casebase_correction`` value is still
    reported (just excluded from ``combined``), not silently skipped.
    """
    model = _make_model()
    new_cases = model.X_train[:2]
    casebase_items = model.X_train[2:4]
    targets = torch.full((2,), 0.5)

    protect_samples = model.X_train[:2]
    with torch.no_grad():
        protect_target_classes = model(protect_samples).argmax(dim=-1).tolist()

    casebase_batch = _casebase_batch(model, n_edges=2, target_offset=5.0)

    losses = compute_losses(
        model, new_cases, casebase_items, targets,
        protect_samples=protect_samples, protect_target_classes=protect_target_classes,
        protect_margin=MARGIN, protect_lambda=1.0,
        casebase_batch=casebase_batch, casebase_lambda=0.0,
    )

    assert losses.casebase_correction.item() > 0.0  # sanity: really nonzero
    assert losses.combined.item() == pytest.approx(
        (losses.correction + 1.0 * losses.protect).item(), abs=1e-6
    )
