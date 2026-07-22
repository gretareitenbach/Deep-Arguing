"""
src/deeparguing/counterfactuals/batch_contest.py

Joint contestability algorithm: instead of ``contest()``'s per-sample
sequential loop (each misclassified sample solved independently against a
live, shared ``model.A``, so one sample's fix can partially undo another's),
this optimizes one shared adjacency edit against every sample's hinge loss
at once.

Loss:  L(A) = sum_i max(0, margin - m_i(A))
where  m_i(A) = target_strength_i(A) - rival_strength_i(A).

Outer loop, until every hinge term is 0, ``max_iters`` is hit, or the edit
budget is exhausted:
  1. Forward the whole batch (or a mini-batch, see ``batch_size``), computing
     every sample's target/rival strength. The rival is recomputed fresh
     every pass (``_target_and_rival_batch``'s argmax), since it can switch
     as ``A`` moves.
  2. hinge = relu(margin - m); active = hinge > 0. This mask replaces
     ``contest()``'s per-sample ``result.success`` bookkeeping. If nothing
     is active, stop.
  3. One backward pass of the summed hinge over the active samples
     (``compute_grae`` with ``rival_indices`` set, so the gradient reflects
     the full margin, not just the target side of it) gives the shared
     ``grad_A L``.
  4. Sparsify with ``select_top_k`` on the flattened gradient.
  5. Backtracking line search for a shared step size that decreases the
     active-sample hinge sum and doesn't diverge (see divergence guard
     below).
  6. Apply once to the shared ``A``, magnitude-clamped exactly like
     ``contest.py``'s ``_perturb_adjacency``.

Dead gradients: ``contest()`` handles a sample whose gradient is uniformly
~0 (a saturated ReLU pinning some ancestor's strength at exactly 0) by
routing to ``bottleneck.find_and_escape_bottleneck``, a per-sample graph
walk -- which doesn't fold into a single batched step, since it needs each
sample's own node-strength graph. Instead, this module computes the shared
gradient through a leaky-ReLU surrogate (``_leaky_relu_surrogate``): a copy
of ``model.gradual_semantics`` with its ReLU's exact-zero region replaced by
a small nonzero slope, used only for this backward pass. Every active
sample -- saturated or not -- then contributes a smooth, nonzero direction
to the same shared gradient, so "dead" never needs to be a special case.
The real forward passes that decide the hinge loss, the stopping condition,
and the line search all keep using the model's actual (hard-ReLU) semantics
unchanged.

Divergence guard: ``ReluSemantics`` has no upper saturation (unlike e.g.
``QuadraticEnergySemantics``), so ``forward_till_convergence``'s repeated
``relu(relu(base) + aggregation)`` can in principle diverge for a large
enough step. The line search rejects any trial whose active-sample target
strength exceeds ``divergence_bound`` before even checking whether it
decreased the loss, so a step that blows up strengths can never be
accepted just because the (also blown-up) hinge sum happened to look small.

Protecting global accuracy: nothing above knows or cares about samples
outside the flip batch, so a step that clears every targeted sample can
still erode the margin of samples elsewhere that were already correct.
``protect_samples``/``protect_target_classes``/``protect_margin``/
``protect_lambda`` (all optional, default off) add a second hinge term for
a caller-supplied "protect" batch -- e.g. a sample of currently
correctly-classified validation examples -- to the same shared loss and
gradient, so a step is only accepted if it decreases the flip batch's
hinge sum *plus* ``protect_lambda`` times the protect batch's. This is a
soft, per-step penalty computed on a fixed sample; it does not by itself
guarantee held-out accuracy doesn't drop -- see
``counterfactuals/global_optimize.py`` for a hard, periodic full-split
accuracy check with rollback layered on top of this.
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
                       _forward_strengths_batch, _perturb_adjacency,
                       _target_and_rival_batch, select_top_k)
from .grae import compute_grae

# ---- Config -------------------------------------------------------------

TOL = 1e-4                  # stop once the active-sample hinge sum is at or below this
LEAKY_NEGATIVE_SLOPE = 0.01  # slope of the gradient-only ReLU surrogate in its "dead" region
DIVERGENCE_BOUND = 100.0    # reject a trial step if it pushes any active target strength above this
ALPHA_INIT = 1.0            # initial line-search step size
BACKTRACK_FACTOR = 0.5      # shrink factor per failed line-search trial
MAX_BACKTRACKS = 10         # line-search retry cap per outer iteration
PROTECT_MARGIN = MARGIN     # default margin a "protect" sample must keep above its rival


@dataclass
class BatchContestResult:
    cleared: Tensor  # bool (B,) -- whether target beat rival by >= margin at the end
    num_cleared: int
    num_total: int
    num_edges_changed: int
    touched_edge_indices: list[int]  # flat indices into model.A that were edited, for correlating specific edges with any later global-accuracy shift
    iterations: int
    final_target_strengths: Tensor
    final_rival_classes: list[int | None]
    final_rival_strengths: Tensor
    # Populated only when batch_contest was called with protect_samples; None otherwise.
    final_protect_target_strengths: Tensor | None = None
    final_protect_cleared: Tensor | None = None


def _leaky_relu_surrogate(semantics: GradualSemantics, negative_slope: float) -> GradualSemantics:
    """A differentiable relaxation of ``semantics``, used only to compute
    ``grad_A L`` in ``batch_contest``. A copy of ``semantics`` with its ReLU
    swapped for leaky-ReLU: identical for positive input, but a small
    nonzero slope where the real ReLU would be exactly flat, so a node
    saturated at 0 still carries a usable gradient direction. The real
    forward passes elsewhere keep using ``semantics`` (the true hard ReLU)
    untouched."""
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
    """Shrink alpha from ``alpha_init`` until a trial both stays under the
    divergence guard -- on the flip batch, and on the protect batch too if
    one is given -- and decreases the combined loss (flip hinge sum, plus
    ``protect_lambda`` times the protect-batch hinge sum when
    ``protect_active_samples`` is given) below ``old_loss``. Returns the
    accepted ``new_A``, or ``None`` if nothing within budget qualifies -- the
    caller treats that as no usable step this iteration for this
    (mini-)batch."""
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
    """Main loop. See module docstring for the algorithm.

    ``batch_size``, if given, splits each outer iteration into shuffled
    mini-batches (one step per mini-batch, reshuffled every pass) instead of
    the full-batch default (one step per outer iteration, over every active
    sample at once).

    ``max_edits``, if given, stops the loop once that many distinct edges
    (flattened indices into ``model.A``) have been touched -- the edit
    budget. Left as ``None`` (unbounded) by default, since no such cap
    exists elsewhere in this codebase to inherit.

    ``protect_samples``/``protect_target_classes`` (optional): a second
    batch -- e.g. currently correctly-classified validation examples -- to
    protect from margin erosion while the flip batch above is optimized.
    Each protect sample contributes ``clamp(protect_margin -
    (own_target - own_rival), min=0)`` to the shared loss, weighted by
    ``protect_lambda``, and its gradient is combined into the same shared
    top-k edge selection and line search used for the flip batch -- it's
    just a second term in the same objective, not a separate mechanism.
    Recomputed fresh every chunk (never mini-batched itself) since
    ``model.A`` may have moved since the last chunk's step. A chunk with no
    active flip sample takes no step at all (unchanged from before this
    param existed), so protect violations are only ever corrected as a
    side effect of a flip-driven step, never on their own initiative. Left
    at ``protect_lambda=0.0``/``protect_samples=None`` by default, in which
    case every new code path here is skipped entirely and behavior is
    unchanged from before this parameter existed.
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

            # Protect-set hinge against the current model.A -- see
            # batch_contest's docstring for why this is recomputed fresh
            # every chunk and only contributes alongside an active flip step.
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

            if g.abs().max().item() == 0.0:
                continue  # even the leaky surrogate found no directional signal

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
