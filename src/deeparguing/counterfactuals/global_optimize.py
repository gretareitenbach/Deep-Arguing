"""Global optimization: maximize misclassified-sample flips via
``batch_contest`` while protecting the model's global (held-out-split)
accuracy.

Two mechanisms working together (see ``batch_contest.py``'s module
docstring for the first, this module's docstring for the second):

  1. A soft per-step penalty: a "protect" batch of currently
     correctly-classified held-out examples, sampled once up front (see
     ``_build_protect_set``), rides along in every ``batch_contest`` call's
     shared loss/gradient/line search via its ``protect_samples``/
     ``protect_lambda`` params.
  2. A hard, periodic guardrail: every ``eval_every`` outer iterations,
     this module calls out to ``evals/global_contest_eval.py`` -- a real,
     full-held-out-split accuracy check, exactly what ``batch_contest``'s
     soft penalty above only ever approximates on a fixed sample -- and
     rolls ``model.A`` back to the last snapshot that stayed within
     ``max_acc_drop`` of baseline if the check fails, then stops.

Both mechanisms are evaluated against the same held-out split
(``--eval-split``, default ``val``), so the hard check is verifying the
same thing the soft penalty nudges toward. This module deliberately leaves
``test`` as an honest, unpeeked split -- run ``run_global_contest_eval.py
--split test`` against the checkpoint this script saves for a final,
post-hoc confirmation.

This module is the only place in ``counterfactuals/`` that imports from
``evals/`` (mirroring ``sweep_global_contest_eval.py``, the existing
precedent) -- ``batch_contest.py`` itself stays a pure numerical primitive
with no dataset/eval-split dependency, only ever seeing whatever
``protect_samples`` tensor this orchestrator hands it.

All hyperparameters and dataset/checkpoint paths live in a YAML config file
(default ``tuning/contest/global_optimize.yaml``), following the same
CLI-flag-overrides-YAML convention as ``contest_all.py``.

Usage::

    python -m deeparguing.counterfactuals.global_optimize
    python -m deeparguing.counterfactuals.global_optimize \\
        --config tuning/contest/global_optimize.yaml \\
        --num-samples 20 --max-iters 20 --eval-every 5 --max-acc-drop 0.01
"""

import argparse
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

from deeparguing.counterfactuals.batch_contest import (ALPHA_INIT,
                                                         DIVERGENCE_BOUND,
                                                         MAX_BACKTRACKS,
                                                         PROTECT_MARGIN, TOL,
                                                         BatchContestResult,
                                                         batch_contest)
from deeparguing.counterfactuals.contest import (DEFAULT_K, MARGIN,
                                                   MAX_ITERS, THRESHOLD)
from deeparguing.counterfactuals.contest_all import (_load_config, _required,
                                                       _resolved)
from deeparguing.counterfactuals.run_contest import (load_all_samples,
                                                       load_fitted_model_and_data)
from deeparguing.evals.global_contest_eval import (GlobalEvalMetrics,
                                                     compute_baseline_metrics,
                                                     evaluate_contested_model)
from deeparguing.gradual_aacbr import GradualAACBR

DEFAULT_CONFIG_PATH = "tuning/contest/global_optimize.yaml"
DEFAULT_PROTECT_SAMPLE_SIZE = 200
DEFAULT_PROTECT_LAMBDA = 1.0
DEFAULT_MAX_ACC_DROP = 0.01
DEFAULT_EVAL_EVERY = 10
DEFAULT_EVAL_SPLIT = "val"


@dataclasses.dataclass
class GlobalOptimizeResult:
    # cumulative touched-edge bookkeeping across every accepted round;
    # cleared/final_target_strengths/etc. reflect the final model.A.
    batch_result: BatchContestResult
    baseline: GlobalEvalMetrics
    final_metrics: GlobalEvalMetrics
    acc_drop: float
    rolled_back: bool
    stopped_reason: str  # "converged" | "acc_drop" | "max_iters" | "max_edits"
    rounds: list[dict[str, Any]]
    protect_sample_size: int


def _build_protect_set(
    model: GradualAACBR, X_eval: Tensor, y_eval: Tensor, sample_size: int
) -> tuple[Tensor, list[int]]:
    """Predict on ``X_eval`` with the model's current (baseline) ``model.A``,
    keep only currently correctly-classified rows, and sample up to
    ``sample_size`` of them once.

    Sampled once per run, not resampled per round: both ``batch_contest``'s
    line search (old vs. trial loss within one call) and this module's
    round-over-round ``acc_drop`` comparison are only meaningful if the
    protect subset staying fixed isn't itself a confound. Broader coverage,
    if wanted, comes from a larger ``sample_size``, not resampling.
    """
    with torch.no_grad():
        predicted = model(X_eval).argmax(dim=-1)
    true = y_eval.argmax(dim=-1)
    correct_indices = (predicted == true).nonzero(as_tuple=True)[0]
    if correct_indices.numel() == 0:
        return X_eval[:0], []
    n = min(sample_size, correct_indices.numel())
    perm = torch.randperm(correct_indices.numel(), device=correct_indices.device)[:n]
    chosen = correct_indices[perm]
    return X_eval[chosen], true[chosen].tolist()


def global_optimize(
    model: GradualAACBR,
    samples: Tensor,
    true_classes: Sequence[int],
    X_eval: Tensor,
    y_eval: Tensor,
    *,
    k: int = DEFAULT_K,
    threshold: float = THRESHOLD,
    margin: float = MARGIN,
    max_iters: int = MAX_ITERS,
    tol: float = TOL,
    max_edits: int | None = None,
    batch_size: int | None = None,
    divergence_bound: float = DIVERGENCE_BOUND,
    alpha_init: float = ALPHA_INIT,
    max_backtracks: int = MAX_BACKTRACKS,
    protect_margin: float = PROTECT_MARGIN,
    protect_lambda: float = DEFAULT_PROTECT_LAMBDA,
    protect_sample_size: int = DEFAULT_PROTECT_SAMPLE_SIZE,
    max_acc_drop: float = DEFAULT_MAX_ACC_DROP,
    eval_every: int = DEFAULT_EVAL_EVERY,
    eval_batch_size: int | None = None,
) -> GlobalOptimizeResult:
    """Repeatedly call ``batch_contest`` in ``eval_every``-sized rounds,
    checking real held-out accuracy between rounds and rolling back to the
    last passing snapshot if ``max_acc_drop`` is exceeded. See module
    docstring for the two-mechanism design."""
    if model.A is None:
        raise Exception("Ensure the model has been fit first.")

    baseline = compute_baseline_metrics(model, X_eval, y_eval, batch_size=eval_batch_size)
    protect_samples, protect_target_classes = _build_protect_set(
        model, X_eval, y_eval, protect_sample_size
    )
    has_protect = protect_samples.shape[0] > 0

    # Zero-effect snapshot of the current (pre-optimization) state -- always
    # a valid BatchContestResult to fall back to if no round ever runs or
    # passes the guardrail, so the rest of this function never has to
    # special-case "no result yet".
    result = batch_contest(
        model, samples, true_classes,
        k=k, threshold=threshold, margin=margin, max_iters=0,
        tol=tol, batch_size=batch_size, divergence_bound=divergence_bound,
        alpha_init=alpha_init, max_backtracks=max_backtracks,
        protect_samples=protect_samples if has_protect else None,
        protect_target_classes=protect_target_classes if has_protect else None,
        protect_margin=protect_margin, protect_lambda=protect_lambda,
    )
    last_good_A = model.A.detach().clone()
    total_touched: set[int] = set()
    rounds: list[dict[str, Any]] = []
    rolled_back = False
    stopped_reason = "max_iters"
    iters_done = 0

    while iters_done < max_iters:
        round_budget = min(eval_every, max_iters - iters_done)
        remaining_edits = None if max_edits is None else max_edits - len(total_touched)
        if remaining_edits is not None and remaining_edits <= 0:
            stopped_reason = "max_edits"
            break

        round_result = batch_contest(
            model, samples, true_classes,
            k=k, threshold=threshold, margin=margin, max_iters=round_budget,
            tol=tol, max_edits=remaining_edits, batch_size=batch_size,
            divergence_bound=divergence_bound, alpha_init=alpha_init,
            max_backtracks=max_backtracks,
            protect_samples=protect_samples if has_protect else None,
            protect_target_classes=protect_target_classes if has_protect else None,
            protect_margin=protect_margin, protect_lambda=protect_lambda,
        )
        iters_done += round_result.iterations

        eval_result = evaluate_contested_model(
            model, model.A, X_eval, y_eval, baseline, batch_size=eval_batch_size
        )
        acc_drop = baseline.accuracy - eval_result.metrics.accuracy
        round_rolled_back = acc_drop > max_acc_drop

        rounds.append(
            {
                "iterations": round_result.iterations,
                "num_cleared": round_result.num_cleared,
                "global_acc": eval_result.metrics.accuracy,
                "acc_drop": acc_drop,
                "rolled_back": round_rolled_back,
            }
        )

        if round_rolled_back:
            model.A = last_good_A.clone()
            rolled_back = True
            stopped_reason = "acc_drop"
            break

        total_touched.update(round_result.touched_edge_indices)
        result = round_result
        last_good_A = model.A.detach().clone()

        if round_result.iterations < round_budget:
            # batch_contest already decided to stop internally (tol reached,
            # no usable step, or its own edit budget hit) -- no point
            # running another round.
            stopped_reason = "converged"
            break

    # `result`'s own touched-edge bookkeeping only covers its own (last
    # accepted) round -- override with the cumulative footprint across every
    # accepted round so a caller sees the true total, not just the last one.
    result = dataclasses.replace(
        result, num_edges_changed=len(total_touched), touched_edge_indices=sorted(total_touched)
    )

    final_eval = evaluate_contested_model(
        model, model.A, X_eval, y_eval, baseline, batch_size=eval_batch_size
    )

    return GlobalOptimizeResult(
        batch_result=result,
        baseline=baseline,
        final_metrics=final_eval.metrics,
        acc_drop=baseline.accuracy - final_eval.metrics.accuracy,
        rolled_back=rolled_back,
        stopped_reason=stopped_reason,
        rounds=rounds,
        protect_sample_size=protect_samples.shape[0],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="YAML file holding hyperparameters and paths (see tuning/contest/global_optimize.yaml). "
        "Any other flag passed here overrides the corresponding value in it.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--qbaf", default=None)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit the run to the first N misclassified samples (default: all).",
    )
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument(
        "--max-edits",
        type=int,
        default=None,
        help="Stop once this many distinct edges have been touched, cumulative "
        "across rounds (default: unbounded).",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--divergence-bound", type=float, default=None)
    parser.add_argument("--alpha-init", type=float, default=None)
    parser.add_argument("--max-backtracks", type=int, default=None)
    parser.add_argument(
        "--protect-margin",
        type=float,
        default=None,
        help="Margin a protect-set sample must keep above its rival before it's "
        "considered eroded (see batch_contest.py's protect_margin).",
    )
    parser.add_argument(
        "--protect-lambda",
        type=float,
        default=None,
        help="Weight of the protect-set hinge penalty in the shared loss. 0 disables it.",
    )
    parser.add_argument(
        "--protect-sample-size",
        type=int,
        default=None,
        help="How many currently-correctly-classified eval-split examples to sample "
        "once, up front, as the protect set.",
    )
    parser.add_argument(
        "--max-acc-drop",
        type=float,
        default=None,
        help="Hard guardrail: roll back and stop once eval-split accuracy has "
        "dropped this much from baseline.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Outer iterations per round before the hard guardrail re-checks "
        "real eval-split accuracy.",
    )
    parser.add_argument(
        "--eval-split",
        default=None,
        help="Held-out split ('val' or 'test') used for both the protect-set "
        "pool and the guardrail check. Default: val -- keep 'test' unpeeked "
        "and use run_global_contest_eval.py --split test for a final check.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory to write the run's JSON log (and default checkpoint "
        "location) to. Created if it doesn't exist.",
    )
    parser.add_argument(
        "--save-checkpoint",
        default=None,
        help="Where to save the model (with its optimized model.A) after the "
        "run. Defaults to '<log-dir>/global_optimize_checkpoint.pt'; pass an "
        "empty string to skip saving a checkpoint entirely.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)

    checkpoint = _required(args.checkpoint, config, "checkpoint", args.config)
    qbaf = _required(args.qbaf, config, "qbaf", args.config)
    num_samples = _resolved(args.num_samples, config, "num_samples", None)
    k = _resolved(args.k, config, "k", DEFAULT_K)
    threshold = _resolved(args.threshold, config, "threshold", THRESHOLD)
    margin = _resolved(args.margin, config, "margin", MARGIN)
    max_iters = _resolved(args.max_iters, config, "max_iters", MAX_ITERS)
    tol = _resolved(args.tol, config, "tol", TOL)
    max_edits = _resolved(args.max_edits, config, "max_edits", None)
    batch_size = _resolved(args.batch_size, config, "batch_size", None)
    divergence_bound = _resolved(args.divergence_bound, config, "divergence_bound", DIVERGENCE_BOUND)
    alpha_init = _resolved(args.alpha_init, config, "alpha_init", ALPHA_INIT)
    max_backtracks = _resolved(args.max_backtracks, config, "max_backtracks", MAX_BACKTRACKS)
    protect_margin = _resolved(args.protect_margin, config, "protect_margin", PROTECT_MARGIN)
    protect_lambda = _resolved(args.protect_lambda, config, "protect_lambda", DEFAULT_PROTECT_LAMBDA)
    protect_sample_size = _resolved(
        args.protect_sample_size, config, "protect_sample_size", DEFAULT_PROTECT_SAMPLE_SIZE
    )
    max_acc_drop = _resolved(args.max_acc_drop, config, "max_acc_drop", DEFAULT_MAX_ACC_DROP)
    eval_every = _resolved(args.eval_every, config, "eval_every", DEFAULT_EVAL_EVERY)
    eval_split = _resolved(args.eval_split, config, "eval_split", DEFAULT_EVAL_SPLIT)
    if eval_split not in ("val", "test"):
        raise ValueError(f"eval_split must be 'val' or 'test', got {eval_split!r}.")
    eval_batch_size = _resolved(args.eval_batch_size, config, "eval_batch_size", None)
    device = _resolved(args.device, config, "device", "cuda" if torch.cuda.is_available() else "cpu")
    log_dir_str = _resolved(args.log_dir, config, "log_dir", "outputs/contestation")
    # save_checkpoint's tri-state (unset -> default path, "" -> skip, path ->
    # explicit) mirrors contest_all.py -- must not fall through _resolved's
    # "treat null/absent as unset" rule.
    save_checkpoint = args.save_checkpoint if args.save_checkpoint is not None else config.get("save_checkpoint")

    with open(qbaf, "r", encoding="utf-8") as f:
        qbaf_data = json.load(f)

    print(f"Loading baseline checkpoint from {checkpoint} ...")
    model, data_dict = load_fitted_model_and_data(checkpoint, device)
    assert model.A is not None, "checkpoint's model was never fit()"
    original_A = model.A.detach().clone()
    samples, true_classes = load_all_samples(qbaf_data, device, num_samples)

    X_eval = data_dict[f"X_{eval_split}"]
    y_eval = data_dict[f"y_{eval_split}"]

    print(
        f"Running global optimization over {samples.shape[0]} misclassified samples, "
        f"protecting {eval_split}-split accuracy (max_acc_drop={max_acc_drop}, eval_every={eval_every})..."
    )
    result = global_optimize(
        model, samples, true_classes, X_eval, y_eval,
        k=k, threshold=threshold, margin=margin, max_iters=max_iters, tol=tol,
        max_edits=max_edits, batch_size=batch_size, divergence_bound=divergence_bound,
        alpha_init=alpha_init, max_backtracks=max_backtracks,
        protect_margin=protect_margin, protect_lambda=protect_lambda,
        protect_sample_size=protect_sample_size, max_acc_drop=max_acc_drop,
        eval_every=eval_every, eval_batch_size=eval_batch_size,
    )

    print(
        f"\nCleared {result.batch_result.num_cleared}/{result.batch_result.num_total} samples "
        f"({result.batch_result.num_cleared / max(1, result.batch_result.num_total):.1%}), "
        f"{result.batch_result.num_edges_changed} edges changed. "
        f"Baseline {eval_split} accuracy={result.baseline.accuracy:.4f}, "
        f"final={result.final_metrics.accuracy:.4f} (drop={result.acc_drop:+.4f}), "
        f"rolled_back={result.rolled_back}, stopped_reason={result.stopped_reason}, "
        f"protect_sample_size={result.protect_sample_size}"
    )

    log_dir = Path(log_dir_str)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Unravel each touched flat index into (source, target, dim) plus its
    # before/after weight, same convention as contest_all.py's log.
    n1, n2, d = model.A.shape
    original_flat = original_A.reshape(-1)
    new_flat = model.A.reshape(-1)
    touched_edges = [
        {
            "edge_id": idx,
            "source": (idx // d) // n2,
            "target": (idx // d) % n2,
            "dim": idx % d,
            "old_weight": original_flat[idx].item(),
            "new_weight": new_flat[idx].item(),
        }
        for idx in result.batch_result.touched_edge_indices
    ]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"global_optimize_{timestamp}.json"
    log = {
        "config": {
            "config_file": args.config,
            "checkpoint": checkpoint,
            "qbaf": qbaf,
            "num_samples": num_samples,
            "k": k,
            "threshold": threshold,
            "margin": margin,
            "max_iters": max_iters,
            "tol": tol,
            "max_edits": max_edits,
            "batch_size": batch_size,
            "divergence_bound": divergence_bound,
            "alpha_init": alpha_init,
            "max_backtracks": max_backtracks,
            "protect_margin": protect_margin,
            "protect_lambda": protect_lambda,
            "protect_sample_size": protect_sample_size,
            "max_acc_drop": max_acc_drop,
            "eval_every": eval_every,
            "eval_split": eval_split,
        },
        "summary": {
            "num_total": result.batch_result.num_total,
            "num_cleared": result.batch_result.num_cleared,
            "success_rate": result.batch_result.num_cleared / max(1, result.batch_result.num_total),
            "num_edges_changed": result.batch_result.num_edges_changed,
        },
        "guardrail": {
            "baseline_accuracy": result.baseline.accuracy,
            "final_accuracy": result.final_metrics.accuracy,
            "acc_drop": result.acc_drop,
            "rolled_back": result.rolled_back,
            "stopped_reason": result.stopped_reason,
            "protect_sample_size": result.protect_sample_size,
        },
        "rounds": result.rounds,
        "samples": [
            {
                "index": i,
                "true_class": true_classes[i],
                "cleared": bool(result.batch_result.cleared[i]),
                "final_target_strength": result.batch_result.final_target_strengths[i].item(),
                "final_rival_class": result.batch_result.final_rival_classes[i],
                "final_rival_strength": result.batch_result.final_rival_strengths[i].item(),
            }
            for i in range(result.batch_result.num_total)
        ],
        "touched_edges": touched_edges,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"Saved run log to {log_path}")

    if save_checkpoint is None:
        save_checkpoint = str(log_dir / "global_optimize_checkpoint.pt")

    if save_checkpoint:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config_paths": torch.load(checkpoint, map_location=device)["config_paths"],
                "A": model.A,
                "X_train": model.X_train,
                "y_train": model.y_train,
                "default_indexes": model.default_indexes,
            },
            save_checkpoint,
        )
        print(f"Saved globally-optimized checkpoint (new adjacency matrix in 'A') to {save_checkpoint}")


if __name__ == "__main__":
    main()
