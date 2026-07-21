import pytest
import torch

from deeparguing import GradualAACBR
from deeparguing.counterfactuals import compute_grae, finite_difference_grae
from deeparguing.semantics.sigmoid_semantics import SigmoidSemantics
from qbaf_fixtures import (
    TARGET_INDEX,
    base_score_fn as _base_score_fn,
    edge_weights_fn as _edge_weights_fn,
    irrelevance_fn as _irrelevance_fn,
    make_fitted_model as _make_qbaf_model,
)

# ---------------------------------------------------------------------------
# Shared small synthetic EW-QBAF -- see ``tests/qbaf_fixtures.py`` for the
# graph diagram and full description. This is the exact graph already
# exercised by ``tests/gradual-aacbr_test.py::test_semantics``, so the
# resulting adjacency/strengths are known-good.
# ---------------------------------------------------------------------------


def _make_fitted_model(max_iters: int) -> GradualAACBR:
    return _make_qbaf_model(SigmoidSemantics(max_iters=max_iters, epsilon=0))


@pytest.mark.parametrize("new_case_index", [6, 7])
@pytest.mark.parametrize("max_iters", [1, 3, 5])
def test_analytic_grae_matches_algorithm_1_finite_difference(new_case_index, max_iters):
    """
    Cross-checks the analytic G-RAEs computed by ``compute_grae`` (autograd)
    against ``finite_difference_grae``, which implements Algorithm 1 of the
    Contestability paper: perturb each edge weight w(r) by +/-epsilon,
    recompute the topic argument's strength, and divide the difference by
    2*epsilon.

    Lemma 5 of the paper guarantees the strength function is differentiable
    w.r.t. edge weights for acyclic EW-QBAFs under this semantics, so the
    analytic gradient (Definition 13's limit, taken via autograd) and
    Algorithm 1's perturbation estimate should agree up to floating-point
    and truncation error. A mismatch here means either the autograd wiring
    in ``compute_grae`` or the leaf/detach setup shared with
    ``finite_difference_grae`` is broken.
    """
    model = _make_fitted_model(max_iters=max_iters)
    new_case = torch.tensor([[new_case_index]], dtype=torch.float32)

    analytic = compute_grae(model, new_case, [TARGET_INDEX], per_sample=True)
    approx = finite_difference_grae(model, new_case, TARGET_INDEX, epsilon=1e-4)

    assert torch.allclose(
        analytic.casebase_edges[0], approx.casebase_edges, atol=1e-3
    ), (
        f"casebase-edge G-RAEs disagree with Algorithm 1's finite-difference "
        f"approximation:\nanalytic={analytic.casebase_edges[0]}\n"
        f"finite-diff={approx.casebase_edges}"
    )
    assert torch.allclose(
        analytic.new_case_edges[0], approx.new_case_edges, atol=1e-3
    ), (
        f"new-case-edge G-RAEs disagree with Algorithm 1's finite-difference "
        f"approximation:\nanalytic={analytic.new_case_edges[0]}\n"
        f"finite-diff={approx.new_case_edges}"
    )


@pytest.mark.parametrize("epsilon", [1e-2, 1e-3, 1e-4, 1e-5])
def test_algorithm_1_approximation_converges_as_epsilon_shrinks(epsilon):
    """
    Algorithm 1 is a perturbation-based estimate of the true gradient, so its
    error should shrink as ``epsilon`` shrinks (until floating point noise
    dominates). This pins down that ``finite_difference_grae`` really is
    behaving like a numerical-differentiation routine, independent of
    ``compute_grae`` -- i.e. it isn't accidentally hard-coded or degenerate.
    """
    model = _make_fitted_model(max_iters=5)
    new_case = torch.tensor([[6]], dtype=torch.float32)

    analytic = compute_grae(model, new_case, [TARGET_INDEX], per_sample=True)
    approx = finite_difference_grae(model, new_case, TARGET_INDEX, epsilon=epsilon)

    casebase_error = (analytic.casebase_edges[0] - approx.casebase_edges).abs().max()
    new_case_error = (analytic.new_case_edges[0] - approx.new_case_edges).abs().max()

    assert casebase_error < 1e-1
    assert new_case_error < 1e-1


def test_batched_analytic_grae_aggregates_per_sample_algorithm_1_estimates():
    """
    ``compute_grae`` without ``per_sample`` returns the *aggregate*
    casebase-edge gradient across a batch (since ``model.A`` is shared).
    Cross-check this against the sum of Algorithm 1's per-sample finite
    difference estimates, run one new case at a time -- this exercises the
    batched code path (rather than only ever calling it with batch size 1).
    """
    model = _make_fitted_model(max_iters=3)
    new_cases = torch.tensor([[6], [7]], dtype=torch.float32)

    aggregate_analytic = compute_grae(
        model, new_cases, [TARGET_INDEX, TARGET_INDEX], per_sample=False
    )

    summed_finite_difference = torch.zeros_like(model.A)
    for i in range(new_cases.shape[0]):
        approx = finite_difference_grae(
            model, new_cases[i : i + 1], TARGET_INDEX, epsilon=1e-4
        )
        summed_finite_difference += approx.casebase_edges

    assert torch.allclose(
        aggregate_analytic.casebase_edges, summed_finite_difference, atol=1e-3
    )


# ---------------------------------------------------------------------------
# Property-based tests (Propositions 4-6 of the Contestability paper).
#
# These bypass ``fit()`` and set ``model.A``/``model.X_train``/
# ``model.default_indexes`` directly, so the graph topology is exactly the
# hand-picked EW-QBAF below rather than whatever AA-CBR's attack/support
# construction would produce. ``irrelevance_edge_weights`` is stubbed to
# always return 0, so the (single, dummy) new case has no influence and the
# topic argument's strength depends purely on ``A`` and the base scores --
# i.e. a plain EW-QBAF, matching the paper's setting exactly.
#
# Important sign-convention note: unlike the paper, this codebase folds
# attack/support membership *into the sign* of the adjacency entry itself
# (``GradualAACBR.fit`` sets ``self.A = -attacks + supports``), rather than
# keeping w(r) unsigned and R^-/R^+ as separate relations. Concretely,
# entry A[j, i] = -w(r) for an attack r=(j,i) and A[j, i] = +w(r) for a
# support. Differentiating w.r.t. this *signed* entry therefore flips the
# attack-side sign relative to the paper's ∇ w.r.t. the unsigned w(r):
#   - Direct edges (Prop. 4): paper says support >= 0, attack <= 0 w.r.t.
#     w(r); since attacks are negated here, *both* come out >= 0 w.r.t. the
#     signed entry (weakening an attack or strengthening a support both
#     raise the target's strength).
#   - Indirect edges with an odd number of downstream attacks (Prop. 5,
#     cases 1 & 3): paper says support <= 0, attack >= 0 w.r.t. w(r); once
#     negated, *both* come out <= 0 here.
#   - Indirect edges with an even number of downstream attacks (Prop. 5,
#     cases 2 & 4): symmetric to the above, *both* come out >= 0 here.
# This matches the intuitive reading directly in terms of the signed
# entry: "increasing A[j, i]" always means "make j's effect on the next
# node in the chain more positive", and the sign of its net effect on the
# topic argument then only depends on how many attacks that effect crosses
# downstream -- not on whether j's own edge happened to be a support or an
# attack.
# ---------------------------------------------------------------------------


def _make_raw_model(A: torch.Tensor, base_scores: torch.Tensor, max_iters: int = 5):
    """Build a GradualAACBR with a hand-picked adjacency matrix, bypassing fit().

    ``A`` must be a (n, n) or (n, n, 1) signed adjacency matrix. Node index 0
    is always the topic argument (``model.default_indexes = [0]``). The
    single dummy new case fed through ``compute_grae``/``finite_difference_grae``
    has zero influence (see module comment above), so the topic argument's
    strength is governed purely by ``A`` and ``base_scores``.
    """
    if A.ndim == 2:
        A = A.unsqueeze(-1)
    n = A.shape[0]

    def base_score_fn(x: torch.Tensor) -> torch.Tensor:
        idx = x.to(dtype=torch.long).squeeze(-1)
        return base_scores[idx].unsqueeze(-1)

    def edge_weights_fn(attacker: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "model.A is set directly in this test; fit() is never called."
        )

    def irrelevance_fn(new_cases: torch.Tensor, casebase: torch.Tensor) -> torch.Tensor:
        return torch.zeros(new_cases.shape[0], casebase.shape[0], 1)

    semantics = SigmoidSemantics(max_iters=max_iters, epsilon=0)
    model = GradualAACBR(semantics, base_score_fn, irrelevance_fn, edge_weights_fn)
    model.A = A
    model.X_train = torch.arange(n).unsqueeze(-1).float()
    model.default_indexes = torch.tensor([0])
    return model


def _property_test_graph() -> tuple[torch.Tensor, torch.Tensor]:
    """The 9-node EW-QBAF used by the direct/indirect/independent tests.

    Node 0 is the topic argument.
      1 --attack--> 0        (direct, R-)
      2 --support--> 0       (direct, R+)
      3 --attack--> 1        (indirect via 1, R-, 1 downstream attack: odd)
      4 --support--> 1       (indirect via 1, R+, 1 downstream attack: odd)
      7 --attack--> 2        (indirect via 2, R-, 0 downstream attacks: even)
      8 --support--> 2       (indirect via 2, R+, 0 downstream attacks: even)
      5 --support--> 6       (independent: neither 5 nor 6 has any path to 0)
    """
    n = 9
    A = torch.zeros(n, n, 1)
    A[1, 0, 0] = -0.6
    A[2, 0, 0] = 0.7
    A[3, 1, 0] = -0.5
    A[4, 1, 0] = 0.4
    A[7, 2, 0] = -0.45
    A[8, 2, 0] = 0.35
    A[5, 6, 0] = 0.3
    base_scores = torch.full((n,), 0.5)
    return A, base_scores


@pytest.mark.parametrize("edge", [(1, 0), (2, 0)])
def test_direct_edges_have_nonnegative_grae(edge):
    """Direct Influence (Proposition 4): with this codebase's signed
    adjacency convention, both a direct attack and a direct support have
    non-negative G-RAE (weakening an attack or strengthening a support can
    only help the topic argument)."""
    A, base_scores = _property_test_graph()
    model = _make_raw_model(A, base_scores)
    new_case = torch.tensor([[0.0]])

    analytic = compute_grae(model, new_case, [0], per_sample=True)
    grae = analytic.casebase_edges[0].squeeze(-1)[edge]
    assert grae >= 0, f"expected non-negative G-RAE for direct edge {edge}, got {grae}"


@pytest.mark.parametrize("edge", [(3, 1), (4, 1)])
def test_indirect_edges_with_odd_downstream_attacks_have_nonpositive_grae(edge):
    """Indirect Influence (Proposition 5, cases 1 & 3): an edge whose single
    path to the topic argument crosses an odd number of further attacks has
    non-positive G-RAE here, regardless of whether the edge itself is an
    attack or a support (both encode "make node 1 -- the attacker of 0 --
    stronger", which can only hurt node 0)."""
    A, base_scores = _property_test_graph()
    model = _make_raw_model(A, base_scores)
    new_case = torch.tensor([[0.0]])

    analytic = compute_grae(model, new_case, [0], per_sample=True)
    grae = analytic.casebase_edges[0].squeeze(-1)[edge]
    assert grae <= 0, f"expected non-positive G-RAE for indirect edge {edge}, got {grae}"


@pytest.mark.parametrize("edge", [(7, 2), (8, 2)])
def test_indirect_edges_with_even_downstream_attacks_have_nonnegative_grae(edge):
    """Indirect Influence (Proposition 5, cases 2 & 4): symmetric to the odd
    case above, but the single path to the topic argument crosses an even
    (zero) number of further attacks (via node 2, the direct supporter), so
    G-RAE is non-negative here."""
    A, base_scores = _property_test_graph()
    model = _make_raw_model(A, base_scores)
    new_case = torch.tensor([[0.0]])

    analytic = compute_grae(model, new_case, [0], per_sample=True)
    grae = analytic.casebase_edges[0].squeeze(-1)[edge]
    assert grae >= 0, f"expected non-negative G-RAE for indirect edge {edge}, got {grae}"


def test_independent_edge_has_zero_grae():
    """Irrelevance (Proposition 6): an edge with no path at all to the topic
    argument has exactly zero G-RAE, both analytically and under Algorithm
    1's finite-difference approximation."""
    A, base_scores = _property_test_graph()
    model = _make_raw_model(A, base_scores)
    new_case = torch.tensor([[0.0]])

    analytic = compute_grae(model, new_case, [0], per_sample=True)
    approx = finite_difference_grae(model, new_case, 0, epsilon=1e-4)

    analytic_grae = analytic.casebase_edges[0].squeeze(-1)[5, 6]
    fd_grae = approx.casebase_edges.squeeze(-1)[5, 6]
    assert analytic_grae == pytest.approx(0.0, abs=1e-8)
    assert fd_grae == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Input validation and result-shape/detachment tests.
# ---------------------------------------------------------------------------


def test_compute_grae_raises_if_model_not_fitted():
    model = GradualAACBR(
        SigmoidSemantics(max_iters=1, epsilon=0),
        _base_score_fn,
        _irrelevance_fn,
        _edge_weights_fn,
    )
    with pytest.raises(Exception, match="fit"):
        compute_grae(model, torch.tensor([[6.0]]), [0])


def test_finite_difference_grae_raises_if_model_not_fitted():
    model = GradualAACBR(
        SigmoidSemantics(max_iters=1, epsilon=0),
        _base_score_fn,
        _irrelevance_fn,
        _edge_weights_fn,
    )
    with pytest.raises(Exception, match="fit"):
        finite_difference_grae(model, torch.tensor([[6.0]]), 0)


def test_compute_grae_raises_on_target_indices_length_mismatch():
    model = _make_fitted_model(max_iters=3)
    new_cases = torch.tensor([[6.0], [7.0]])
    with pytest.raises(ValueError, match="target_indices"):
        compute_grae(model, new_cases, [TARGET_INDEX])


def test_finite_difference_grae_raises_on_batch_size_other_than_one():
    model = _make_fitted_model(max_iters=3)
    new_cases = torch.tensor([[6.0], [7.0]])
    with pytest.raises(ValueError, match="batch size"):
        finite_difference_grae(model, new_cases, TARGET_INDEX)


def test_grae_result_tensors_are_detached_from_autograd():
    """Both entry points document that returned tensors need no further
    autograd tracking; confirm they are indeed plain (non-leaf-requiring,
    graph-free) tensors so callers can safely mutate/store them."""
    model = _make_fitted_model(max_iters=3)
    new_case = torch.tensor([[6.0]])

    analytic = compute_grae(model, new_case, [TARGET_INDEX], per_sample=True)
    assert not analytic.casebase_edges.requires_grad
    assert not analytic.new_case_edges.requires_grad
    assert analytic.casebase_edges.grad_fn is None
    assert analytic.new_case_edges.grad_fn is None

    approx = finite_difference_grae(model, new_case, TARGET_INDEX)
    assert not approx.casebase_edges.requires_grad
    assert not approx.new_case_edges.requires_grad


def test_compute_grae_shapes():
    model = _make_fitted_model(max_iters=3)
    new_cases = torch.tensor([[6.0], [7.0]])

    aggregate = compute_grae(model, new_cases, [TARGET_INDEX, TARGET_INDEX])
    assert aggregate.casebase_edges.shape == model.A.shape
    assert aggregate.new_case_edges.shape == model.new_cases_attacks_adjacency.shape

    per_sample = compute_grae(
        model, new_cases, [TARGET_INDEX, TARGET_INDEX], per_sample=True
    )
    assert per_sample.casebase_edges.shape == (new_cases.shape[0], *model.A.shape)
    assert per_sample.new_case_edges.shape == model.new_cases_attacks_adjacency.shape


def test_finite_difference_grae_shapes():
    model = _make_fitted_model(max_iters=3)
    new_case = torch.tensor([[6.0]])

    approx = finite_difference_grae(model, new_case, TARGET_INDEX)
    assert approx.casebase_edges.shape == model.A.shape
    assert approx.new_case_edges.shape == model.new_cases_attacks_adjacency.shape
