"""
src/deeparguing/contest/batch_contest.py

Joint contestability algorithm: optimizes one shared adjacency edit
(``model.A``) against every sample's hinge loss at once, instead of
``contest()``'s per-sample sequential loop. Each outer iteration forwards
the batch, computes a shared gradient (via a leaky-ReLU surrogate so
saturated nodes still carry a gradient), sparsifies it with top-k edge
selection, and takes a shared backtracking-line-search step.
"""

import copy
import functools
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR
from deeparguing.semantics.gradual_semantics import GradualSemantics
from deeparguing.semantics.relu_semantics import ReluSemantics

from .contest import (DEFAULT_K, MARGIN, MAX_ITERS, THRESHOLD,
                       _forward_strengths_batch, _mask_default_sources,
                       _perturb_adjacency, _target_and_rival_batch,
                       select_top_k)
from .grae import compute_grae

# ---- Config -------------------------------------------------------------

TOL = 1e-4                   # stop once the active-sample hinge sum is at or below this
LEAKY_NEGATIVE_SLOPE = 0.01   # slope of the gradient-only ReLU surrogate in its "dead" region
DIVERGENCE_BOUND = 100.0      # reject a trial step if it pushes any active target strength above this
ALPHA_INIT = 1.0              # initial line-search step size
BACKTRACK_FACTOR = 0.5        # shrink factor per failed line-search trial
MAX_BACKTRACKS = 10           # line-search retry cap per outer iteration
PROTECT_MARGIN = MARGIN       # default margin a "protect" sample must keep above its rival


@dataclass
class BatchContestResult:
    """Outcome of ``batch_contest()``."""

    cleared: Tensor  # bool (B,) -- whether target beat rival by >= margin at the end
    num_cleared: int
    num_total: int
    num_edges_changed: int
    touched_edge_indices: list[int]  # flat indices into model.A that were edited
    iterations: int
    final_target_strengths: Tensor
    final_rival_classes: list[int | None]
    final_rival_strengths: Tensor
    final_protect_target_strengths: Tensor | None = None  # only set if protect_samples was given
    final_protect_cleared: Tensor | None = None


def _leaky_relu_surrogate(semantics: GradualSemantics, negative_slope: float) -> GradualSemantics:
    """Copy of ``semantics`` with its ReLU swapped for leaky-ReLU, used only
    to compute ``grad_A L`` in ``batch_contest`` so saturated nodes still
    carry a gradient. Real forward passes elsewhere keep using ``semantics``
    (the true hard ReLU) unchanged.
    """
    if not isinstance(semantics, ReluSemantics):
        raise TypeError(
            "the leaky-relu gradient surrogate only makes sense for "
            f"ReluSemantics, got {type(semantics).__name__}"
        )
    surrogate = copy.copy(semantics)
    surrogate.infl = functools.partial(F.leaky_relu, negative_slope=negative_slope)
    return surrogate


def _joint_backtracking_step(
    model: GradualAACBR,
    active_samples: Tensor,
    active_targets: list[int],
    edge_indices: Tensor,
    direction: Tensor,
    threshold: float,
    margin: float,
    old_loss: float,
    alpha_init: float,
    factor: float,
    max_backtracks: int,
    divergence_bound: float,
    protect_active_samples: Tensor | None = None,
    protect_active_targets: list[int] | None = None,
    protect_margin: float = PROTECT_MARGIN,
    protect_lambda: float = 0.0,
) -> Tensor | None:
    """Shrink alpha from ``alpha_init`` until a trial stays under
    ``divergence_bound`` (on the flip batch, and the protect batch too if
    given) and decreases the combined loss below ``old_loss``.

    Parameters
    ----------
    model, active_samples, active_targets, edge_indices, direction
        Current model and the active flip batch's samples/targets/edit.
    threshold, margin, old_loss, alpha_init, factor, max_backtracks, divergence_bound
        Search parameters; see ``batch_contest``.
    protect_active_samples, protect_active_targets, protect_margin, protect_lambda
        Optional protect batch contributing a second hinge term to the loss.

    Returns
    -------
    Tensor | None
        The accepted ``new_A``, or ``None`` if nothing within budget qualifies.
    """
    assert model.A is not None
    has_protect = (
        protect_lambda > 0.0
        and protect_active_samples is not None
        and protect_active_samples.shape[0] > 0
    )
    alpha = alpha_init
    for _ in range(max_backtracks):
        trial_A = _perturb_adjacency(model.A, edge_indices, direction, alpha)
        trial_strengths = _forward_strengths_batch(model, active_samples, trial_A)
        trial_target, _, trial_rival = _target_and_rival_batch(
            trial_strengths, active_targets, threshold
        )
        if trial_target.max().item() <= divergence_bound:
            trial_loss = torch.clamp(margin - (trial_target - trial_rival), min=0.0).sum().item()

            if has_protect:
                protect_strengths = _forward_strengths_batch(model, protect_active_samples, trial_A)
                protect_target, _, protect_rival = _target_and_rival_batch(
                    protect_strengths, protect_active_targets, threshold
                )
                if protect_target.max().item() > divergence_bound:
                    alpha *= factor
                    continue
                trial_loss += protect_lambda * torch.clamp(
                    protect_margin - (protect_target - protect_rival), min=0.0
                ).sum().item()

            if trial_loss < old_loss:
                return trial_A
        alpha *= factor
    return None


def batch_contest(
    model: GradualAACBR,
    samples: Tensor,
    target_classes: Sequence[int],
    k: int = DEFAULT_K,
    threshold: float = THRESHOLD,
    margin: float = MARGIN,
    max_iters: int = MAX_ITERS,
    tol: float = TOL,
    max_edits: int | None = None,
    batch_size: int | None = None,
    leaky_negative_slope: float = LEAKY_NEGATIVE_SLOPE,
    divergence_bound: float = DIVERGENCE_BOUND,
    alpha_init: float = ALPHA_INIT,
    backtrack_factor: float = BACKTRACK_FACTOR,
    max_backtracks: int = MAX_BACKTRACKS,
    protect_samples: Tensor | None = None,
    protect_target_classes: Sequence[int] | None = None,
    protect_margin: float = PROTECT_MARGIN,
    protect_lambda: float = 0.0,
) -> BatchContestResult:
    """Jointly contest a batch of samples against one shared ``model.A``.

    Parameters
    ----------
    model : GradualAACBR
        A fitted model (``model.A`` populated).
    samples : Tensor
        Batch of new cases to contest, shape (B, x1, ..., xn).
    target_classes : Sequence[int]
        Length-B, the desired class for each sample.
    k, threshold, margin, max_iters
        See ``contest.py``'s module-level defaults.
    tol : float
        Stop once the active-sample hinge sum is at or below this.
    max_edits : int | None
        Stop once this many distinct edges (flattened indices into
        ``model.A``) have been touched. ``None`` (default) is unbounded.
    batch_size : int | None
        If given, splits each outer iteration into shuffled mini-batches
        (one step per mini-batch) instead of the full-batch default.
    leaky_negative_slope, divergence_bound, alpha_init, backtrack_factor, max_backtracks
        See module-level defaults above.
    protect_samples, protect_target_classes : Tensor | None, Sequence[int] | None
        Optional second batch (e.g. currently correctly-classified
        validation examples) to protect from margin erosion. Each protect
        sample contributes ``clamp(protect_margin - (own_target -
        own_rival), min=0)`` to the shared loss, weighted by
        ``protect_lambda``. Left at ``None``/``protect_lambda=0.0`` by
        default, in which case this has no effect.
    protect_margin, protect_lambda
        See above.

    Returns
    -------
    BatchContestResult
    """
    if model.A is None:
        raise Exception("Ensure the model has been fit first.")

    batch_total = samples.shape[0]
    target_classes = list(target_classes)
    if len(target_classes) != batch_total:
        raise ValueError(
            "target_classes must have exactly one entry per sample: "
            f"got {len(target_classes)} for a batch of {batch_total}."
        )

    if protect_lambda > 0.0 and protect_samples is None:
        raise ValueError("protect_lambda > 0 requires protect_samples to be given.")
    if protect_samples is not None:
        protect_target_classes = list(protect_target_classes) if protect_target_classes is not None else []
        if len(protect_target_classes) != protect_samples.shape[0]:
            raise ValueError(
                "protect_target_classes must have exactly one entry per protect sample: "
                f"got {len(protect_target_classes)} for {protect_samples.shape[0]} protect samples."
            )
    has_protect = protect_samples is not None and protect_samples.shape[0] > 0

    surrogate = _leaky_relu_surrogate(model.gradual_semantics, leaky_negative_slope)
    touched_edges: set[int] = set()
    iters_run = 0

    for iters_run in range(1, max_iters + 1):
        if batch_size is None:
            chunks = [torch.arange(batch_total, device=samples.device)]
        else:
            perm = torch.randperm(batch_total, device=samples.device)
            chunks = [perm[i : i + batch_size] for i in range(0, batch_total, batch_size)]

        pass_max_hinge = 0.0
        pass_took_step = False
        budget_hit = False

        for chunk in chunks:
            chunk_samples = samples[chunk]
            chunk_targets = [target_classes[i] for i in chunk.tolist()]

            strengths = _forward_strengths_batch(model, chunk_samples, model.A)
            target_strengths, rival_classes, rival_strengths = _target_and_rival_batch(
                strengths, chunk_targets, threshold
            )
            hinge = torch.clamp(margin - (target_strengths - rival_strengths), min=0.0)
            pass_max_hinge = max(pass_max_hinge, hinge.max().item())

            active_local = (hinge > 0).nonzero(as_tuple=True)[0]
            if active_local.numel() == 0:
                continue

            active_global = chunk[active_local]
            active_samples = samples[active_global]
            active_targets = [target_classes[i] for i in active_global.tolist()]
            active_rivals = [rival_classes[i] for i in active_local.tolist()]
            old_loss = hinge[active_local].sum().item()

            grae_result = compute_grae(
                model,
                active_samples,
                target_indices=active_targets,
                rival_indices=active_rivals,
                semantics_override=surrogate,
            )
            g = grae_result.casebase_edges.reshape(-1)

            protect_active_samples: Tensor | None = None
            protect_active_targets: list[int] | None = None
            protect_old_loss = 0.0
            if has_protect and protect_lambda > 0.0:
                protect_strengths = _forward_strengths_batch(model, protect_samples, model.A)
                protect_target, protect_rival_classes, protect_rival = _target_and_rival_batch(
                    protect_strengths, protect_target_classes, threshold
                )
                protect_hinge = torch.clamp(protect_margin - (protect_target - protect_rival), min=0.0)
                protect_active_local = (protect_hinge > 0).nonzero(as_tuple=True)[0]
                if protect_active_local.numel() > 0:
                    protect_active_samples = protect_samples[protect_active_local]
                    protect_active_targets = [
                        protect_target_classes[i] for i in protect_active_local.tolist()
                    ]
                    protect_active_rivals = [
                        protect_rival_classes[i] for i in protect_active_local.tolist()
                    ]
                    protect_old_loss = protect_hinge[protect_active_local].sum().item()

                    if protect_lambda > 0.0:
                        protect_grae_result = compute_grae(
                            model,
                            protect_active_samples,
                            target_indices=protect_active_targets,
                            rival_indices=protect_active_rivals,
                            semantics_override=surrogate,
                        )
                        g = g + protect_lambda * protect_grae_result.casebase_edges.reshape(-1)

            g = _mask_default_sources(model, g)
            if g.abs().max().item() == 0.0:
                continue

            edge_indices = select_top_k(g, k)
            direction = g[edge_indices]

            combined_old_loss = old_loss + protect_lambda * protect_old_loss

            new_A = _joint_backtracking_step(
                model, active_samples, active_targets, edge_indices, direction,
                threshold, margin, combined_old_loss,
                alpha_init=alpha_init, factor=backtrack_factor,
                max_backtracks=max_backtracks, divergence_bound=divergence_bound,
                protect_active_samples=protect_active_samples,
                protect_active_targets=protect_active_targets,
                protect_margin=protect_margin,
                protect_lambda=protect_lambda,
            )
            if new_A is None:
                continue

            touched_edges.update(edge_indices.tolist())
            model.A = new_A
            pass_took_step = True

            if max_edits is not None and len(touched_edges) >= max_edits:
                budget_hit = True
                break

        if pass_max_hinge <= tol or not pass_took_step or budget_hit:
            break

    final_strengths = _forward_strengths_batch(model, samples, model.A)
    final_target, final_rival_classes, final_rival = _target_and_rival_batch(
        final_strengths, target_classes, threshold
    )
    cleared = (final_target - final_rival) >= margin

    final_protect_target_strengths: Tensor | None = None
    final_protect_cleared: Tensor | None = None
    if protect_samples is not None:
        protect_final_strengths = _forward_strengths_batch(model, protect_samples, model.A)
        protect_final_target, _, protect_final_rival = _target_and_rival_batch(
            protect_final_strengths, protect_target_classes, threshold
        )
        final_protect_target_strengths = protect_final_target
        final_protect_cleared = (protect_final_target - protect_final_rival) >= protect_margin

    return BatchContestResult(
        cleared=cleared,
        num_cleared=int(cleared.sum().item()),
        num_total=batch_total,
        num_edges_changed=len(touched_edges),
        touched_edge_indices=sorted(touched_edges),
        iterations=iters_run,
        final_target_strengths=final_target,
        final_rival_classes=final_rival_classes,
        final_rival_strengths=final_rival,
        final_protect_target_strengths=final_protect_target_strengths,
        final_protect_cleared=final_protect_cleared,
    )
