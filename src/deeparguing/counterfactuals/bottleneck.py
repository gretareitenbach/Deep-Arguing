"""
src/deeparguing/counterfactuals/bottleneck.py

Escape logic for the dead-gradient case ``contest()`` hits when
``_casebase_grae`` is uniformly ~0 for a sample's target class: not because
the sample is genuinely stuck, but because a hard-ReLU node somewhere
upstream of ``target_class``'s default argument has saturated (its own
strength is pinned at exactly 0), blocking gradient flow through it (see
``grae.py``'s module docstring and ``sweep_dead_gradients.py``, which flags
these samples ahead of time).

``GradualAACBR.forward(..., return_all_strengths=True)`` already exposes
every casebase node's own converged strength for a given sample -- not just
the target class's -- so no changes to ``gradual_aacbr.py`` are needed to
see "intermediate" strengths; this module just reads that out.

Update rule summary (mirrors ``contest.py``'s docstring):
  - Bottleneck node: first node with strength exactly 0, found by a greedy
    walk from the sample towards ``target_class``'s default argument,
    following the single strongest edge at each hop (same greedy spirit as
    ``select_top_k``'s top-k edges, not an exhaustive search).
  - Edges moved: the bottleneck node's incoming edges, ranked by
    |source node's own strength| -- since the aggregation step is linear,
    this is exactly the local partial derivative of the bottleneck's own
    aggregation w.r.t. that edge weight, read directly off the forward pass
    already computed (no backward pass needed).
  - Step size: geometric growth from a small ``alpha_init``, the mirror
    image of ``bisection_line_search``'s backtracking phase.
  - Stopping: the bottleneck node's own strength becomes nonzero ("un-stuck"
    -- a different, weaker condition than crossing the classification
    margin), or ``max_steps`` is hit.
"""

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR

from .contest import (DEFAULT_K, THRESHOLD, _forward_strengths,
                       _perturb_adjacency, _target_and_rival, select_top_k)

# ---- Config -------------------------------------------------------------

BOTTLENECK_ALPHA_INIT = 1e-3     # initial step size for the expanding search
BOTTLENECK_GROWTH_FACTOR = 2.0   # growth factor per trial
MAX_EXPANSIONS = 20              # expansion-phase retry cap


def _node_strengths(model: GradualAACBR, sample: Tensor, A: Tensor) -> Tensor:
    """Forward pass of ``sample`` with ``model.A`` temporarily swapped for
    ``A``, returning every casebase node's own converged strength -- shape
    (n, d) -- not just the target class's final strength
    (``_forward_strengths`` only reads out the default rows)."""
    original_A = model.A
    try:
        model.A = A
        with torch.no_grad():
            strengths = model(sample, return_all_strengths=True)
    finally:
        model.A = original_A
    result = strengths[0]
    return result if result.ndim == 2 else result.unsqueeze(-1)


def find_bottleneck(
    model: GradualAACBR, sample: Tensor, target_class: int
) -> tuple[int, Tensor] | None:
    """Greedily walk the influence graph from ``sample`` towards
    ``target_class``'s default argument, following the single strongest
    edge at each hop, and return the first node whose own strength is
    pinned at exactly 0 -- a saturated ReLU.

    Returns ``(bottleneck_node, node_strengths)``, where ``node_strengths``
    is the (n, d) per-node strength tensor already computed to do this walk
    (reused by ``select_bottleneck_edges`` rather than recomputed). Returns
    ``None`` if the walk reaches ``target_class``'s default argument (or a
    dead end) without ever finding a saturated node -- i.e. there is no ReLU
    bottleneck here, and a dead gradient (if any) has some other cause.
    """
    assert model.A is not None
    A = model.post_process_func(model.A)
    node_strengths = _node_strengths(model, sample, model.A)
    E = model.new_cases_attacks_adjacency[0]  # (n, d): sample's own edges into the casebase
    target_idx = int(model.default_indexes[target_class].item())

    def is_saturated(node: int) -> bool:
        return bool((node_strengths[node] == 0).all())

    current = int(E.abs().sum(dim=-1).argmax().item())
    visited = {current}
    while True:
        if is_saturated(current):
            return current, node_strengths
        if current == target_idx:
            return None  # reached the target with nothing saturated along the way

        outgoing = A[current].abs().sum(dim=-1).clone()
        outgoing[list(visited)] = -1.0
        nxt = int(outgoing.argmax().item())
        if outgoing[nxt] <= 0:
            return None  # dead end before reaching the target or a saturated node
        visited.add(nxt)
        current = nxt


def _bottleneck_leverage_vector(
    node_strengths: Tensor, A: Tensor, bottleneck_node: int
) -> Tensor:
    """Flat (n*n*d) vector matching ``_casebase_grae``'s layout: zero
    everywhere except ``bottleneck_node``'s incoming edges
    (``A[:, bottleneck_node, :]``), where the local leverage of source node
    j's edge on ``bottleneck_node``'s own aggregation is exactly
    ``node_strengths[j]`` -- the aggregation step is linear, so this *is*
    the partial derivative, no backward pass needed."""
    n, m, d = A.shape
    leverage = torch.zeros(n, m, d, dtype=node_strengths.dtype, device=A.device)
    leverage[:, bottleneck_node, :] = node_strengths
    return leverage.reshape(-1)


def select_bottleneck_edges(
    node_strengths: Tensor, A: Tensor, bottleneck_node: int, k: int
) -> Tensor:
    """Indices (into the flattened ``model.A``, same convention as
    ``select_top_k``) of the k edges feeding into ``bottleneck_node`` with
    the largest ``|node_strengths[source]|`` -- see
    ``_bottleneck_leverage_vector``."""
    return select_top_k(_bottleneck_leverage_vector(node_strengths, A, bottleneck_node), k)


def expanding_step_search(
    model: GradualAACBR,
    sample: Tensor,
    edge_indices: Tensor,
    direction: Tensor,
    bottleneck_node: int,
    alpha_init: float = BOTTLENECK_ALPHA_INIT,
    growth_factor: float = BOTTLENECK_GROWTH_FACTOR,
    max_steps: int = MAX_EXPANSIONS,
) -> tuple[float, Tensor] | None:
    """Mirror image of ``bisection_line_search``'s bracketing phase: starts
    small and grows ``alpha`` (instead of shrinking from ``alpha_max``)
    until ``bottleneck_node``'s own strength is no longer pinned at exactly
    0, or ``max_steps`` is hit.

    Terminates on "un-stuck", a different (and weaker) condition than
    crossing the classification margin ``bisection_line_search`` checks for
    -- escaping the bottleneck doesn't by itself mean ``target_class`` wins
    the argmax, only that the gradient is live again so the ordinary search
    can take over next iteration. Hence a separate function rather than
    reusing ``bisection_line_search``.

    Returns ``(alpha, new_A)`` for the first un-stuck trial, or ``None`` if
    it never un-sticks within budget.
    """
    assert model.A is not None
    alpha = alpha_init
    for _ in range(max_steps):
        trial_A = _perturb_adjacency(model.A, edge_indices, direction, alpha)
        node_strengths = _node_strengths(model, sample, trial_A)
        if not bool((node_strengths[bottleneck_node] == 0).all()):
            return alpha, trial_A
        alpha *= growth_factor
    return None


def find_and_escape_bottleneck(
    model: GradualAACBR,
    sample: Tensor,
    target_class: int,
    grae_vector: Tensor,
    k: int = DEFAULT_K,
    threshold: float = THRESHOLD,
) -> tuple[Tensor, float, Tensor, float, int | None, float] | None:
    """Handle the dead-gradient case: find the saturated bottleneck node,
    rank its incoming edges by local leverage, and grow a step along the
    highest-leverage one until it un-sticks.

    Returns the same 6-tuple shape ``contest()``'s ordinary path builds
    (``edge_indices``, prefixed onto ``bisection_line_search``'s 5-tuple)
    -- ``(edge_indices, alpha, new_A, new_target_strength, new_rival_class,
    new_rival_strength)`` -- so both branches feed one shared trace-recording
    path, or ``None`` if no bottleneck exists or it can't be escaped within
    budget -- either way, ``contest()`` treats the sample as structurally
    hopeless.
    """
    assert model.A is not None
    bottleneck = find_bottleneck(model, sample, target_class)
    if bottleneck is None:
        return None
    bottleneck_node, node_strengths = bottleneck

    edge_indices = select_bottleneck_edges(node_strengths, model.A, bottleneck_node, k)
    leverage_vector = _bottleneck_leverage_vector(node_strengths, model.A, bottleneck_node)
    direction = leverage_vector[edge_indices]
    if not bool(direction.any()):
        # Every candidate edge's source is itself pinned at 0, so the local
        # (linear, one-hop) leverage vanishes at all of them -- but a
        # multi-hop path could still carry a nonzero true gradient through
        # these same positions, so fall back to it before giving up.
        direction = grae_vector[edge_indices]
        if not bool(direction.any()):
            return None  # no directional signal at all -- genuinely stuck

    step = expanding_step_search(model, sample, edge_indices, direction, bottleneck_node)
    if step is None:
        return None
    alpha, new_A = step

    new_strengths = _forward_strengths(model, sample, new_A)
    new_target_strength, new_rival_class, new_rival_strength = _target_and_rival(
        new_strengths, target_class, threshold
    )
    return edge_indices, alpha, new_A, new_target_strength, new_rival_class, new_rival_strength
