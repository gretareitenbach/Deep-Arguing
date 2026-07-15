"""Gradient-based Relation Attribution Explanations (G-RAEs).

Implements Definition 13 of the Contestability paper (Yin, Potyka, Rago,
Kampik & Toni), computing the gradient of an argument's final strength
with respect to individual edge weights, using PyTorch autograd instead
of the paper's perturbation-based approximation (their Algorithm 1).

G-RAEs are computed with respect to both edge tensors that feed a
new case's classification:

    - ``model.A``: the casebase-internal adjacency. Shared across every
      prediction the model makes (built once in ``fit()``), so edits
      here are the only ones with a persistent, global effect. This is
      what Week 3's heuristic algorithm will eventually adjust, and
      what Week 4 measures the ripple effect of.
    - ``model.new_cases_attacks_adjacency``: the new case's own edges
      into the casebase. Recomputed fresh per prediction, so edits here
      are local to a single sample.

Neither tensor is a true PyTorch leaf when it comes out of a normal
forward pass -- each is the output of a learned network (
``casebase_edge_weights`` / ``irrelevance_edge_weights`` respectively).
To get the gradient with respect to the *edge weight values themselves*
(matching the paper's definition, which perturbs w(r) directly), we
detach both tensors into fresh leaves before re-running the downstream
semantics computation. Gradients then stop at the leaves rather than
continuing back through the generating networks or the raw input
pixels.
"""

import itertools
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR
from deeparguing.semantics.gradual_semantics import GradualSemantics


@dataclass
class GRAEResult:
    """Container for the two edge-weight gradients that make up a G-RAE.

    Attributes
    ----------
    casebase_edges : Tensor
        Gradient of the (summed, batched) target strength with respect
        to ``model.A``. Shape matches ``model.A`` (n, n, d).

        Note: since ``model.A`` is shared across the batch, a single
        backward pass over multiple new cases gives the *aggregate*
        sensitivity across all of them, not a per-sample breakdown. If
        a per-sample casebase-edge G-RAE is needed (e.g. for the
        Direct Influence sign check), backward passes must be run
        one sample at a time -- see ``per_sample`` on
        ``compute_grae``.
    new_case_edges : Tensor
        Gradient of each sample's own target strength with respect to
        its row of ``model.new_cases_attacks_adjacency``. Shape (B, n,
        d) -- naturally per-sample already, since this tensor is
        batched to begin with.
    target_indices : Sequence[int]
        The default-argument index used as the topic argument for each
        sample in the batch (i.e. which class's strength we
        differentiated).
    """

    casebase_edges: Tensor
    new_case_edges: Tensor
    target_indices: Sequence[int]


def _batched_casebase_base_scores(model: GradualAACBR, batch_size: int) -> Tensor:
    """Recompute and batch-tile the casebase's own base scores.

    Mirrors ``GradualAACBR._GradualAACBR__batch_base_scores``. Does not
    depend on either G-RAE leaf, so it is run under ``no_grad`` -- we
    only want gradients flowing to ``A_leaf``/``E_leaf``, not back into
    the base-score network.
    """
    with torch.no_grad():
        scores = model.compute_base_scores(model.X_train)
    return torch.tile(scores.unsqueeze(0), (batch_size, 1, 1))  # B x n x d


def _replay_default_strengths(
    model: GradualAACBR,
    A: Tensor,
    E: Tensor,
    casebase_base_scores: Tensor,
    new_cases_base_scores: Tensor,
    semantics: GradualSemantics | None = None,
) -> Tensor:
    """Replay ``__new_case_influence`` + ``gradual_semantics`` with ``A``/``E`` swapped in.

    Mirrors ``GradualAACBR.forward``/``__new_case_influence`` exactly,
    except ``model.A`` and ``model.new_cases_attacks_adjacency`` are
    replaced by ``A`` and ``E`` (the detached leaves in ``compute_grae``,
    or perturbed copies in ``finite_difference_grae``).

    ``semantics``, if given, replaces ``model.gradual_semantics`` for this
    replay only -- used by ``joint_contest`` to get a gradient through a
    differentiable surrogate (e.g. leaky-ReLU standing in for a hard ReLU
    that's saturated exactly at 0) without changing what the model actually
    computes anywhere else.
    """
    semantics = semantics or model.gradual_semantics
    aggregations = semantics.aggregation_func(
        E.unsqueeze(1), new_cases_base_scores  # B x 1 x n x d
    )
    influenced_base_scores = semantics.influence_func(
        casebase_base_scores, aggregations
    )
    strengths = semantics(model.post_process_func(A), influenced_base_scores)

    if model.dimensions > 1:
        final_strengths = torch.matmul(strengths, model.W)  # (B, n, d) -> (B, n)
    else:
        final_strengths = strengths.squeeze(-1)  # (B, n, 1) -> (B, n)

    return final_strengths[:, model.default_indexes]


def compute_grae(
    model: GradualAACBR,
    new_cases: Tensor,
    target_indices: Sequence[int],
    per_sample: bool = False,
    rival_indices: Sequence[int | None] | None = None,
    semantics_override: GradualSemantics | None = None,
) -> GRAEResult:
    """Compute G-RAEs for a batch of new cases against a chosen target argument each.

    Parameters
    ----------
    model : GradualAACBR
        A model that has already been ``fit()``. ``model.A`` must be
        populated.
    new_cases : Tensor
        Batch of new case characterisations, shape (B, x1, ..., xn) --
        same shape ``model.forward`` expects. Typically the
        misclassified samples pulled from ``misclassified_qbaf.json``.
    target_indices : Sequence[int]
        Length-B sequence giving, for each sample, which entry of
        ``model.default_indexes`` to treat as the topic argument (i.e.
        the desired/correct class to differentiate the strength of --
        not necessarily the class the model actually predicted).
    per_sample : bool, default True
        If True, also loop over the batch one sample at a time to
        recover a per-sample ``casebase_edges`` gradient (see
        ``GRAEResult.casebase_edges``), at the cost of B extra backward
        passes. If False, ``casebase_edges`` is the aggregate gradient
        across the whole batch.
    rival_indices : Sequence[int | None] | None, default None
        If given, length-B, one entry per sample: differentiate
        ``target_strength - rival_strength`` instead of just
        ``target_strength`` for that sample (``None`` for a sample means
        "no rival, differentiate target only" -- e.g. the
        single-topic-argument case where the rival is a fixed constant,
        not a function of ``A``). Used by ``joint_contest`` so the shared
        gradient reflects the full margin ``m_i(A) = target_i(A) -
        rival_i(A)``, not just the target side of it.
    semantics_override : GradualSemantics | None, default None
        If given, replaces ``model.gradual_semantics`` for this replay
        only (see ``_replay_default_strengths``) -- e.g. a leaky-ReLU
        surrogate so saturated (exactly-0) nodes still carry a gradient.

    Returns
    -------
    GRAEResult
        See above. All returned tensors are detached (no further
        autograd tracking needed downstream).

    Notes
    -----
    Implementation sketch:

    1. Run (or reuse) a forward pass so ``model.A`` and
       ``model.new_cases_attacks_adjacency`` are populated for
       ``new_cases``.
    2. Detach both into fresh leaves, ``A_leaf`` and ``E_leaf``, each
       with ``requires_grad_(True)``.
    3. Replay only the downstream computation
       (``__new_case_influence``'s aggregation/influence step, then
       ``gradual_semantics(A_leaf, ...)``) using the leaves in place of
       the originals -- do not recompute
       ``casebase_edge_weights``/``irrelevance_edge_weights`` from
       scratch, or the leaves are pointless.
    4. Gather ``final_strengths[range(B), target_indices]``, sum, and
       call ``.backward()`` once.
    5. ``E_leaf.grad`` is already per-sample. For ``A_leaf.grad``,
       either use the aggregate (default) or loop per-sample with
       ``retain_graph=True`` if ``per_sample`` is requested.
    """
    if model.A is None:
        raise Exception("Ensure the model has been fit first.")

    batch_size = new_cases.shape[0]
    target_indices = list(target_indices)
    if len(target_indices) != batch_size:
        raise ValueError(
            "target_indices must have exactly one entry per new case: "
            f"got {len(target_indices)} for a batch of {batch_size}."
        )

    with torch.no_grad():
        # Populates model.new_cases_attacks_adjacency / new_cases_base_scores
        # for this batch; the resulting graph itself is discarded.
        model(new_cases)

    A_leaf = model.A.detach().clone().requires_grad_(True)
    E_leaf = model.new_cases_attacks_adjacency.detach().clone().requires_grad_(True)

    casebase_base_scores = _batched_casebase_base_scores(model, batch_size)
    new_cases_base_scores = model.new_cases_base_scores.detach()

    default_strengths = _replay_default_strengths(
        model, A_leaf, E_leaf, casebase_base_scores, new_cases_base_scores,
        semantics=semantics_override,
    )

    target_indices_t = torch.as_tensor(target_indices, dtype=torch.long)
    target_strengths = default_strengths[torch.arange(batch_size), target_indices_t]

    if rival_indices is None:
        objective = target_strengths
    else:
        if len(rival_indices) != batch_size:
            raise ValueError(
                "rival_indices must have exactly one entry per new case: "
                f"got {len(rival_indices)} for a batch of {batch_size}."
            )
        terms = [
            target_strengths[i] if rival is None
            else target_strengths[i] - default_strengths[i, rival]
            for i, rival in enumerate(rival_indices)
        ]
        objective = torch.stack(terms)

    objective.sum().backward(retain_graph=per_sample)
    assert A_leaf.grad is not None and E_leaf.grad is not None

    casebase_edges = A_leaf.grad.detach().clone()
    new_case_edges = E_leaf.grad.detach().clone()

    if per_sample:
        casebase_edges = torch.zeros(
            (batch_size, *model.A.shape), dtype=A_leaf.dtype, device=A_leaf.device
        )
        for i in range(batch_size):
            A_leaf.grad = None
            objective[i].backward(retain_graph=True)
            assert A_leaf.grad is not None
            casebase_edges[i] = A_leaf.grad.detach().clone()

    return GRAEResult(
        casebase_edges=casebase_edges,
        new_case_edges=new_case_edges,
        target_indices=target_indices,
    )


def finite_difference_grae(
    model: GradualAACBR,
    new_case: Tensor,
    target_index: int,
    epsilon: float = 1e-4,
) -> GRAEResult:
    """Approximate G-RAEs via perturbation (Algorithm 1 in the paper).

    A brute-force cross-check for ``compute_grae``'s analytic
    gradients: perturb one edge weight at a time by ``epsilon``,
    recompute the target strength, and divide the difference by
    ``epsilon``. Intended for a single sample on a small casebase --
    this is O(n) forward passes and will not scale to the full
    misclassified-sample batch.

    Parameters
    ----------
    model : GradualAACBR
        A model that has already been ``fit()``.
    new_case : Tensor
        A single new case characterisation, shape (1, x1, ..., xn).
    target_index : int
        Which entry of ``model.default_indexes`` to treat as the topic
        argument.
    epsilon : float, default 1e-4
        Perturbation size. Too large biases the estimate; too small
        risks floating point noise dominating the difference.

    Returns
    -------
    GRAEResult
        Same shape/structure as ``compute_grae``'s output, so the two
        can be compared directly (e.g. ``torch.allclose``).

    Notes
    -----
    Tuesday's cross-check: run this and ``compute_grae`` on the same
    small synthetic casebase (3-4 nodes) and confirm they agree within
    a reasonable tolerance before trusting the analytic version on
    real CIFAR-10 samples.
    """
    if model.A is None:
        raise Exception("Ensure the model has been fit first.")
    if new_case.shape[0] != 1:
        raise ValueError(
            "finite_difference_grae expects a single new case (batch size 1), "
            f"got batch of {new_case.shape[0]}."
        )

    with torch.no_grad():
        model(new_case)

    A0 = model.A.detach().clone()
    E0 = model.new_cases_attacks_adjacency.detach().clone()  # (1, n, d)

    casebase_base_scores = _batched_casebase_base_scores(model, batch_size=1)
    new_cases_base_scores = model.new_cases_base_scores.detach()

    def target_strength(A: Tensor, E: Tensor) -> float:
        with torch.no_grad():
            default_strengths = _replay_default_strengths(
                model, A, E, casebase_base_scores, new_cases_base_scores
            )
        return default_strengths[0, target_index].item()

    casebase_edges = torch.zeros_like(A0)
    for idx in itertools.product(*(range(size) for size in A0.shape)):
        A_plus, A_minus = A0.clone(), A0.clone()
        A_plus[idx] += epsilon
        A_minus[idx] -= epsilon
        casebase_edges[idx] = (
            target_strength(A_plus, E0) - target_strength(A_minus, E0)
        ) / (2 * epsilon)

    new_case_edges = torch.zeros_like(E0)
    for idx in itertools.product(*(range(size) for size in E0.shape)):
        E_plus, E_minus = E0.clone(), E0.clone()
        E_plus[idx] += epsilon
        E_minus[idx] -= epsilon
        new_case_edges[idx] = (
            target_strength(A0, E_plus) - target_strength(A0, E_minus)
        ) / (2 * epsilon)

    return GRAEResult(
        casebase_edges=casebase_edges,
        new_case_edges=new_case_edges,
        target_indices=[target_index],
    )