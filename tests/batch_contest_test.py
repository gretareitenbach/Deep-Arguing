import pytest
import torch

from deeparguing import GradualAACBR
from deeparguing.counterfactuals.grae import compute_grae
from deeparguing.counterfactuals.batch_contest import (_leaky_relu_surrogate,
                                                         batch_contest)
from deeparguing.semantics.relu_semantics import ReluSemantics
from deeparguing.semantics.sigmoid_semantics import SigmoidSemantics
from qbaf_fixtures import (
    TARGET_INDEX,
    base_score_fn as _base_score_fn,
    edge_weights_fn as _edge_weights_fn,
    irrelevance_fn as _irrelevance_fn,
    make_fitted_model as _make_qbaf_model,
)

# ---------------------------------------------------------------------------
# Same small synthetic EW-QBAF as ``tests/contest_test.py`` (see
# ``tests/qbaf_fixtures.py``), but fit with ``ReluSemantics`` instead of
# ``SigmoidSemantics`` -- the leaky-relu gradient surrogate only makes sense
# for a hard-ReLU model, and ``batch_contest`` is meant to be used with
# ``ReluSemantics``.
# ---------------------------------------------------------------------------


def _make_fitted_model(max_iters: int) -> GradualAACBR:
    return _make_qbaf_model(ReluSemantics(max_iters=max_iters, epsilon=0))


# ---------------------------------------------------------------------------
# _leaky_relu_surrogate
# ---------------------------------------------------------------------------


def test_leaky_relu_surrogate_matches_relu_on_the_positive_side():
    semantics = ReluSemantics(max_iters=5, epsilon=0)
    surrogate = _leaky_relu_surrogate(semantics, negative_slope=0.01)

    x = torch.tensor([0.3, 1.5, 4.0])
    assert torch.allclose(surrogate.infl(x), torch.relu(x))


def test_leaky_relu_surrogate_is_nonzero_where_relu_is_flat():
    semantics = ReluSemantics(max_iters=5, epsilon=0)
    surrogate = _leaky_relu_surrogate(semantics, negative_slope=0.01)

    x = torch.tensor([-2.0, -0.5, -0.001])
    # The real ReLU is exactly 0 (and its gradient exactly 0) here -- the
    # surrogate must not be.
    assert torch.all(torch.relu(x) == 0)
    assert torch.all(surrogate.infl(x) != 0)


def test_leaky_relu_surrogate_rejects_non_relu_semantics():
    semantics = SigmoidSemantics(max_iters=5, epsilon=0)
    with pytest.raises(TypeError, match="ReluSemantics"):
        _leaky_relu_surrogate(semantics, negative_slope=0.01)


def test_leaky_relu_surrogate_does_not_mutate_the_original_semantics():
    semantics = ReluSemantics(max_iters=5, epsilon=0)
    original_infl = semantics.infl
    _leaky_relu_surrogate(semantics, negative_slope=0.01)
    assert semantics.infl is original_infl


# ---------------------------------------------------------------------------
# compute_grae(..., rival_indices=...)
# ---------------------------------------------------------------------------


def test_rival_indices_differentiates_target_minus_rival():
    """With two default arguments (a real rival), the gradient with
    ``rival_indices`` set should equal target's gradient minus rival's
    gradient -- checked against two separate ``rival_indices=None`` calls."""
    semantics = ReluSemantics(max_iters=3, epsilon=0)
    model = GradualAACBR(semantics, _base_score_fn, _irrelevance_fn, _edge_weights_fn)
    model.use_symmetric_attacks = True
    model.use_supports = True
    model.use_blockers = False

    X_train = torch.tensor([[0], [1], [2], [3], [4]])
    y_train = torch.tensor([[0], [1], [0], [1], [0]])
    X_default = torch.tensor([[5], [6]], dtype=torch.float32)
    y_default = torch.tensor([[0], [1]], dtype=torch.float32)
    model.fit(X_train, y_train, X_default, y_default)

    new_case = torch.tensor([[7]], dtype=torch.float32)

    combined = compute_grae(model, new_case, target_indices=[0], rival_indices=[1])
    target_only = compute_grae(model, new_case, target_indices=[0])
    rival_only = compute_grae(model, new_case, target_indices=[1])

    assert torch.allclose(
        combined.casebase_edges,
        target_only.casebase_edges - rival_only.casebase_edges,
        atol=1e-5,
    )


def test_rival_indices_none_entry_falls_back_to_target_only():
    """A ``None`` entry in ``rival_indices`` (the single-topic-argument case,
    where the rival is a fixed constant, not a function of A) should
    contribute only the target's own gradient, matching plain
    ``rival_indices=None`` behavior for that sample."""
    model = _make_fitted_model(max_iters=3)
    new_case = torch.tensor([[6]], dtype=torch.float32)

    with_none_rival = compute_grae(
        model, new_case, target_indices=[TARGET_INDEX], rival_indices=[None]
    )
    target_only = compute_grae(model, new_case, target_indices=[TARGET_INDEX])

    assert torch.allclose(with_none_rival.casebase_edges, target_only.casebase_edges)


# ---------------------------------------------------------------------------
# batch_contest
# ---------------------------------------------------------------------------


def test_batch_contest_clears_and_persists_to_model_A():
    model = _make_fitted_model(max_iters=5)
    new_case = torch.tensor([[6]], dtype=torch.float32)

    before = model(new_case)[0, TARGET_INDEX].item()
    assert before < 0.5 + 0.01  # sanity: starts below the boundary

    result = batch_contest(model, new_case, [TARGET_INDEX], k=2, max_iters=15)

    assert result.num_total == 1
    assert bool(result.cleared[0])
    assert result.num_cleared == 1
    assert result.num_edges_changed > 0
    assert result.iterations >= 1

    # the accepted steps must persist on the model, not just a trial copy
    after = model(new_case)[0, TARGET_INDEX].item()
    assert after == pytest.approx(result.final_target_strengths[0].item())


def test_batch_contest_divergence_guard_caps_final_strength():
    """A tight ``divergence_bound`` should force the line search to back off
    to smaller steps, keeping the final target strength under the bound --
    ReluSemantics has no upper saturation, so without the guard the same
    run climbs well past it (regression-checked below)."""
    model = _make_fitted_model(max_iters=5)
    new_case = torch.tensor([[6]], dtype=torch.float32)

    result = batch_contest(
        model, new_case, [TARGET_INDEX], k=2, max_iters=15, divergence_bound=1.0
    )

    assert result.final_target_strengths.max().item() <= 1.0 + 1e-6

    # Without the guard (the default, generous bound), this same setup
    # climbs past 1.0 -- confirms the guard above is actually doing
    # something, not just trivially satisfied.
    unguarded_model = _make_fitted_model(max_iters=5)
    unguarded_result = batch_contest(
        unguarded_model, new_case, [TARGET_INDEX], k=2, max_iters=15
    )
    assert unguarded_result.final_target_strengths.max().item() > 1.0


def test_batch_contest_respects_max_iters():
    model = _make_fitted_model(max_iters=5)
    new_case = torch.tensor([[6]], dtype=torch.float32)

    result = batch_contest(model, new_case, [TARGET_INDEX], k=2, max_iters=1)

    assert result.iterations <= 1


def test_batch_contest_raises_if_model_not_fitted():
    model = GradualAACBR(
        ReluSemantics(max_iters=1, epsilon=0),
        _base_score_fn,
        _irrelevance_fn,
        _edge_weights_fn,
    )
    with pytest.raises(Exception, match="fit"):
        batch_contest(model, torch.tensor([[6.0]]), [TARGET_INDEX])


def test_batch_contest_raises_on_target_classes_length_mismatch():
    model = _make_fitted_model(max_iters=3)
    new_cases = torch.tensor([[6.0], [7.0]])
    with pytest.raises(ValueError, match="target_classes"):
        batch_contest(model, new_cases, [TARGET_INDEX])


# ---------------------------------------------------------------------------
# batch_contest(..., protect_samples=..., protect_lambda=...)
#
# Uses a dedicated two-default-argument model (same casebase/edge-weight/
# irrelevance functions as ``test_rival_indices_differentiates_target_minus_rival``
# above) so there's a real rival class to build a "protect" margin against,
# rather than the single-topic-argument threshold fallback the shared
# ``qbaf_fixtures`` model uses. Numbers below (which lambda actually moves
# the outcome, by how much) were confirmed against a live run of this exact
# fixture before being locked into assertions -- see git history.
# ---------------------------------------------------------------------------


def _make_two_class_model(max_iters: int = 5) -> GradualAACBR:
    semantics = ReluSemantics(max_iters=max_iters, epsilon=0)
    model = GradualAACBR(semantics, _base_score_fn, _irrelevance_fn, _edge_weights_fn)
    model.use_symmetric_attacks = True
    model.use_supports = True
    model.use_blockers = False

    X_train = torch.tensor([[0], [1], [2], [3], [4]])
    y_train = torch.tensor([[0], [1], [0], [1], [0]])
    X_default = torch.tensor([[5], [6]], dtype=torch.float32)
    y_default = torch.tensor([[0], [1]], dtype=torch.float32)
    model.fit(X_train, y_train, X_default, y_default)
    return model


_FLIP_CASE = torch.tensor([[6]], dtype=torch.float32)
_WATCH_CASE = torch.tensor([[7]], dtype=torch.float32)


def test_batch_contest_protect_lambda_requires_protect_samples():
    model = _make_two_class_model()
    with pytest.raises(ValueError, match="protect_lambda"):
        batch_contest(model, _FLIP_CASE, [0], protect_lambda=1.0)


def test_batch_contest_protect_target_classes_length_mismatch_raises():
    model = _make_two_class_model()
    protect_samples = torch.tensor([[7.0], [2.0]])
    with pytest.raises(ValueError, match="protect_target_classes"):
        batch_contest(
            model, _FLIP_CASE, [0],
            protect_samples=protect_samples, protect_target_classes=[1],
        )


def test_batch_contest_protect_lambda_zero_matches_baseline_behavior():
    """protect_lambda=0.0 with a nonempty protect_samples must be a
    complete no-op -- identical accepted edges/model.A/final strengths to a
    plain call with neither protect_* arg -- not just numerically
    equivalent."""
    torch.manual_seed(0)
    with_inert_protect = _make_two_class_model()
    result_with = batch_contest(
        with_inert_protect, _FLIP_CASE, [0], k=3, max_iters=20, margin=0.01,
        protect_samples=_WATCH_CASE, protect_target_classes=[1],
        protect_margin=0.05, protect_lambda=0.0,
    )

    torch.manual_seed(0)
    without_protect = _make_two_class_model()
    result_without = batch_contest(
        without_protect, _FLIP_CASE, [0], k=3, max_iters=20, margin=0.01
    )

    assert result_with.touched_edge_indices == result_without.touched_edge_indices
    assert result_with.iterations == result_without.iterations
    assert torch.allclose(with_inert_protect.A, without_protect.A)
    assert torch.allclose(
        result_with.final_target_strengths, result_without.final_target_strengths
    )
    # protect_samples was given, so the informational field is still
    # populated -- protect_lambda=0 only makes it inert, not absent.
    assert result_with.final_protect_target_strengths is not None
    assert result_without.final_protect_target_strengths is None


def test_batch_contest_result_protect_fields_populated_only_when_protect_samples_given():
    model = _make_two_class_model()
    result = batch_contest(
        model, _FLIP_CASE, [0], k=3, max_iters=20, margin=0.01,
        protect_samples=_WATCH_CASE, protect_target_classes=[1],
        protect_margin=0.05, protect_lambda=1.0,
    )
    assert result.final_protect_target_strengths is not None
    assert result.final_protect_cleared is not None
    assert result.final_protect_target_strengths.shape == (1,)


def test_batch_contest_protect_penalty_preserves_protect_margin():
    """Flipping ``_FLIP_CASE`` towards class 0 shares edges with
    ``_WATCH_CASE``'s own class-1 strength (confirmed live: an unprotected
    run swings ``_WATCH_CASE``'s class1-minus-class0 margin from 0 to
    roughly -2.46). Turning on ``protect_lambda`` for a "protect" batch
    containing ``_WATCH_CASE`` should still let the flip succeed, but leave
    ``_WATCH_CASE``'s margin far less eroded."""

    def run(protect_lambda: float) -> tuple[bool, float]:
        torch.manual_seed(0)
        model = _make_two_class_model()
        result = batch_contest(
            model, _FLIP_CASE, [0], k=3, max_iters=20, margin=0.01, divergence_bound=100.0,
            protect_samples=_WATCH_CASE, protect_target_classes=[1],
            protect_margin=0.05, protect_lambda=protect_lambda,
        )
        watch_strengths = model(_WATCH_CASE)[0]
        watch_margin = (watch_strengths[1] - watch_strengths[0]).item()
        return bool(result.cleared[0]), watch_margin

    unprotected_cleared, unprotected_margin = run(protect_lambda=0.0)
    protected_cleared, protected_margin = run(protect_lambda=1.0)

    assert unprotected_cleared
    assert unprotected_margin == pytest.approx(-2.46, abs=1e-2)

    assert protected_cleared
    assert protected_margin > unprotected_margin + 1.0  # far less eroded
    assert protected_margin == pytest.approx(-0.0071, abs=1e-2)
