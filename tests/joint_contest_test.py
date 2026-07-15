import pytest
import torch

from deeparguing import GradualAACBR
from deeparguing.counterfactuals.grae import compute_grae
from deeparguing.counterfactuals.joint_contest import (_leaky_relu_surrogate,
                                                         joint_contest)
from deeparguing.semantics.relu_semantics import ReluSemantics
from deeparguing.semantics.sigmoid_semantics import SigmoidSemantics

# ---------------------------------------------------------------------------
# Same small synthetic EW-QBAF as ``tests/contest_test.py``, but fit with
# ``ReluSemantics`` instead of ``SigmoidSemantics`` -- the leaky-relu
# gradient surrogate only makes sense for a hard-ReLU model, and
# ``joint_contest`` is meant to be used with ``ReluSemantics``.
# ---------------------------------------------------------------------------

_EDGE_WEIGHTS = torch.tensor(
    [
        # 0, 1, 2, 3, 4, 5, 6, 7
        [0, 1, 0, 0, 0, 0, 0, 0],  # 0
        [1, 0, 0, 0, 0, 0, 0, 0],  # 1
        [1, 0, 0, 1, 0, 0, 0, 0],  # 2
        [0, 1, 0, 0, 0, 0, 0, 0],  # 3
        [1, 0, 0, 1, 0, 0, 0, 0],  # 4
        [0, 0, 0, 0, 0, 0, 0, 0],  # 5
        [0, 0, 0, 0, 0, 0, 0, 0],  # 6
        [0, 0, 0, 0, 1, 0, 0, 0],  # 7
    ],
    dtype=torch.float32,
)

_BASE_SCORES = torch.tensor([0.5, 0.5, 0.7, 0.8, 0.9, 0.5, 0.5, 0.5])


def _base_score_fn(case: torch.Tensor) -> torch.Tensor:
    case = case.to(dtype=torch.int).squeeze(-1)
    return _BASE_SCORES[case].unsqueeze(-1)


def _edge_weights_fn(attacker: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    attacker = attacker.to(dtype=torch.int).squeeze(-1)
    target = target.to(dtype=torch.int).squeeze(-1)
    return _EDGE_WEIGHTS[attacker.unsqueeze(1), target.unsqueeze(0)]


def _irrelevance_fn(new_cases: torch.Tensor, casebase: torch.Tensor) -> torch.Tensor:
    new_cases = new_cases.unsqueeze(1).to(dtype=torch.int)
    casebase = casebase.unsqueeze(0).to(dtype=torch.int)
    r = _EDGE_WEIGHTS[new_cases, casebase].squeeze(-1)
    return r.unsqueeze(-1)


def _make_fitted_model(max_iters: int) -> GradualAACBR:
    semantics = ReluSemantics(max_iters=max_iters, epsilon=0)
    model = GradualAACBR(
        semantics,
        _base_score_fn,
        _irrelevance_fn,
        _edge_weights_fn,
    )
    model.use_symmetric_attacks = True
    model.use_supports = True
    model.use_blockers = False

    X_train = torch.tensor([[0], [1], [2], [3], [4]])
    y_train = torch.tensor([[0], [1], [0], [1], [0]])
    X_default = torch.tensor([[5]], dtype=torch.float32)
    y_default = torch.tensor([[5]], dtype=torch.float32)
    model.fit(X_train, y_train, X_default, y_default)
    return model


# Only one default argument exists in this casebase, so it is always the
# (only) topic/target argument -- no real rival, THRESHOLD stands in.
TARGET_INDEX = 0


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
# joint_contest
# ---------------------------------------------------------------------------


def test_joint_contest_clears_and_persists_to_model_A():
    model = _make_fitted_model(max_iters=5)
    new_case = torch.tensor([[6]], dtype=torch.float32)

    before = model(new_case)[0, TARGET_INDEX].item()
    assert before < 0.5 + 0.01  # sanity: starts below the boundary

    result = joint_contest(model, new_case, [TARGET_INDEX], k=2, max_iters=15)

    assert result.num_total == 1
    assert bool(result.cleared[0])
    assert result.num_cleared == 1
    assert result.num_edges_changed > 0
    assert result.iterations >= 1

    # the accepted steps must persist on the model, not just a trial copy
    after = model(new_case)[0, TARGET_INDEX].item()
    assert after == pytest.approx(result.final_target_strengths[0].item())


def test_joint_contest_divergence_guard_caps_final_strength():
    """A tight ``divergence_bound`` should force the line search to back off
    to smaller steps, keeping the final target strength under the bound --
    ReluSemantics has no upper saturation, so without the guard the same
    run climbs well past it (regression-checked below)."""
    model = _make_fitted_model(max_iters=5)
    new_case = torch.tensor([[6]], dtype=torch.float32)

    result = joint_contest(
        model, new_case, [TARGET_INDEX], k=2, max_iters=15, divergence_bound=1.0
    )

    assert result.final_target_strengths.max().item() <= 1.0 + 1e-6

    # Without the guard (the default, generous bound), this same setup
    # climbs past 1.0 -- confirms the guard above is actually doing
    # something, not just trivially satisfied.
    unguarded_model = _make_fitted_model(max_iters=5)
    unguarded_result = joint_contest(
        unguarded_model, new_case, [TARGET_INDEX], k=2, max_iters=15
    )
    assert unguarded_result.final_target_strengths.max().item() > 1.0


def test_joint_contest_respects_max_iters():
    model = _make_fitted_model(max_iters=5)
    new_case = torch.tensor([[6]], dtype=torch.float32)

    result = joint_contest(model, new_case, [TARGET_INDEX], k=2, max_iters=1)

    assert result.iterations <= 1


def test_joint_contest_raises_if_model_not_fitted():
    model = GradualAACBR(
        ReluSemantics(max_iters=1, epsilon=0),
        _base_score_fn,
        _irrelevance_fn,
        _edge_weights_fn,
    )
    with pytest.raises(Exception, match="fit"):
        joint_contest(model, torch.tensor([[6.0]]), [TARGET_INDEX])


def test_joint_contest_raises_on_target_classes_length_mismatch():
    model = _make_fitted_model(max_iters=3)
    new_cases = torch.tensor([[6.0], [7.0]])
    with pytest.raises(ValueError, match="target_classes"):
        joint_contest(model, new_cases, [TARGET_INDEX])
