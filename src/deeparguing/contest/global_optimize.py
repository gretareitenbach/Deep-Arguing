"""Global optimization: maximize misclassified-sample flips via
``batch_contest`` while protecting the model's global (held-out-split)
accuracy, using a soft per-step protect-set penalty plus a hard periodic
accuracy guardrail with rollback.

Usage::

    python -m deeparguing.contest.global_optimize
    python -m deeparguing.contest.global_optimize \\
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

from deeparguing.contest.core.batch_contest import (ALPHA_INIT,
                                                     DIVERGENCE_BOUND,
                                                     MAX_BACKTRACKS,
                                                     PROTECT_MARGIN, TOL,
                                                     BatchContestResult,
                                                     batch_contest)
from deeparguing.contest.core.contest import (DEFAULT_K, MARGIN,
                                               MAX_ITERS, THRESHOLD)
from deeparguing.contest.scripts.config_cli import (load_config, required,
                                                     resolved)
from deeparguing.contest.scripts.run_contest import (load_all_samples,
                                                      load_fitted_model_and_data)
from deeparguing.evals.global_contest_eval import (GlobalEvalMetrics,
                                                     compute_baseline_metrics,
                                                     evaluate_contested_model)
from deeparguing.gradual_aacbr import GradualAACBR
from deeparguing.md_log import write_markdown_log
from deeparguing.output_paths import (output_path, resolve_read_path,
                                       resolve_write_path, today_output_dir)

DEFAULT_CONFIG_PATH = "tuning/contest/global_optimize.yaml"
DEFAULT_PROTECT_SAMPLE_SIZE = 200
DEFAULT_PROTECT_LAMBDA = 1.0
DEFAULT_MAX_ACC_DROP = 0.01
DEFAULT_EVAL_EVERY = 10
DEFAULT_EVAL_SPLIT = "val"
DEFAULT_MD_LOG_FILENAME = "global_optimize.md"
DEFAULT_SEED = 0


@dataclasses.dataclass
class GlobalOptimizeResult:
    """Outcome of ``global_optimize()``. ``batch_result``'s touched-edge
    bookkeeping is cumulative across every accepted round; its cleared/
    final_target_strengths/etc. reflect the final ``model.A``."""

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

    Returns
    -------
    tuple[Tensor, list[int]]
        The sampled inputs and their true class labels.
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


def _rounds_table(rounds: list[dict[str, Any]]) -> str:
    """Markdown table of each round's outcome (iterations, cleared count,
    accuracy, drop, rollback)."""
    if not rounds:
        return "No rounds ran."
    lines = [
        "| Round | Iterations | Cleared | Global Acc | Acc Drop | Rolled Back |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rounds, start=1):
        lines.append(
            f"| {i} | {r['iterations']} | {r['num_cleared']} | {r['global_acc']:.4f} | "
            f"{r['acc_drop']:+.4f} | {'yes' if r['rolled_back'] else 'no'} |"
        )
    return "\n".join(lines)


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
    last passing snapshot if ``max_acc_drop`` is exceeded.

    Parameters
    ----------
    model : GradualAACBR
        A fitted model (``model.A`` populated).
    samples : Tensor
        Misclassified samples to contest, shape (B, x1, ..., xn).
    true_classes : Sequence[int]
        Length-B, the desired class for each sample.
    X_eval, y_eval : Tensor
        Held-out split used for both the protect set and the guardrail check.
    k, threshold, margin, max_iters, tol, max_edits, batch_size,
    divergence_bound, alpha_init, max_backtracks
        Passed through to ``batch_contest``.
    protect_margin, protect_lambda, protect_sample_size
        Protect-set penalty configuration; see ``batch_contest``'s
        ``protect_*`` parameters and ``_build_protect_set``.
    max_acc_drop : float
        Guardrail threshold: roll back and stop once eval-split accuracy
        has dropped this much from baseline.
    eval_every : int
        Outer iterations per round before the guardrail re-checks accuracy.
    eval_batch_size : int | None
        Batch size for the guardrail's forward pass.

    Returns
    -------
    GlobalOptimizeResult
    """
    if model.A is None:
        raise Exception("Ensure the model has been fit first.")

    baseline = compute_baseline_metrics(model, X_eval, y_eval, batch_size=eval_batch_size)
    protect_samples, protect_target_classes = _build_protect_set(
        model, X_eval, y_eval, protect_sample_size
    )
    has_protect = protect_samples.shape[0] > 0

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
            stopped_reason = "converged"
            break

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
        help="YAML file holding hyperparameters and paths. Any other flag "
        "passed here overrides the corresponding value in it.",
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
        help="Margin a protect-set sample must keep above its rival.",
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
        help="How many currently-correctly-classified eval-split examples to "
        "sample once, up front, as the protect set.",
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
        help="Held-out split ('val' or 'test') used for the protect-set pool "
        "and the guardrail check. Default: val.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Seeds torch before the protect set is drawn. Default: {DEFAULT_SEED}.",
    )
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
    parser.add_argument(
        "--md-log-path",
        default=None,
        help="Markdown file to append a human-readable run summary to "
        f"(created if missing). Defaults to today's outputs/<date>/"
        f"{DEFAULT_MD_LOG_FILENAME}; pass an empty string to skip.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    checkpoint = resolve_read_path(required(args.checkpoint, config, "checkpoint", args.config))
    qbaf = resolve_read_path(required(args.qbaf, config, "qbaf", args.config))
    num_samples = resolved(args.num_samples, config, "num_samples", None)
    k = resolved(args.k, config, "k", DEFAULT_K)
    threshold = resolved(args.threshold, config, "threshold", THRESHOLD)
    margin = resolved(args.margin, config, "margin", MARGIN)
    max_iters = resolved(args.max_iters, config, "max_iters", MAX_ITERS)
    tol = resolved(args.tol, config, "tol", TOL)
    max_edits = resolved(args.max_edits, config, "max_edits", None)
    batch_size = resolved(args.batch_size, config, "batch_size", None)
    divergence_bound = resolved(args.divergence_bound, config, "divergence_bound", DIVERGENCE_BOUND)
    alpha_init = resolved(args.alpha_init, config, "alpha_init", ALPHA_INIT)
    max_backtracks = resolved(args.max_backtracks, config, "max_backtracks", MAX_BACKTRACKS)
    protect_margin = resolved(args.protect_margin, config, "protect_margin", PROTECT_MARGIN)
    protect_lambda = resolved(args.protect_lambda, config, "protect_lambda", DEFAULT_PROTECT_LAMBDA)
    protect_sample_size = resolved(
        args.protect_sample_size, config, "protect_sample_size", DEFAULT_PROTECT_SAMPLE_SIZE
    )
    max_acc_drop = resolved(args.max_acc_drop, config, "max_acc_drop", DEFAULT_MAX_ACC_DROP)
    eval_every = resolved(args.eval_every, config, "eval_every", DEFAULT_EVAL_EVERY)
    eval_split = resolved(args.eval_split, config, "eval_split", DEFAULT_EVAL_SPLIT)
    if eval_split not in ("val", "test"):
        raise ValueError(f"eval_split must be 'val' or 'test', got {eval_split!r}.")
    eval_batch_size = resolved(args.eval_batch_size, config, "eval_batch_size", None)
    seed = resolved(args.seed, config, "seed", DEFAULT_SEED)
    device = resolved(args.device, config, "device", "cuda" if torch.cuda.is_available() else "cpu")
    log_dir_str = resolved(args.log_dir, config, "log_dir", str(today_output_dir()))
    save_checkpoint = args.save_checkpoint if args.save_checkpoint is not None else config.get("save_checkpoint")
    md_log_path = args.md_log_path if args.md_log_path is not None else config.get("md_log_path")
    if md_log_path is None:
        md_log_path = output_path(DEFAULT_MD_LOG_FILENAME)

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
        f"protecting {eval_split}-split accuracy (max_acc_drop={max_acc_drop}, eval_every={eval_every}, seed={seed})..."
    )
    torch.manual_seed(seed)
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
            "seed": seed,
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

    if md_log_path:
        write_markdown_log(
            [
                "--- GLOBAL OPTIMIZE ---",
                f"Run: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
                f"Checkpoint: {checkpoint}",
                f"QBAF: {qbaf} (num_samples={num_samples if num_samples is not None else 'all'})",
                f"Eval split: {eval_split}",
                f"Config: k={k} margin={margin} protect_margin={protect_margin} "
                f"protect_lambda={protect_lambda} protect_sample_size={protect_sample_size} "
                f"max_acc_drop={max_acc_drop} eval_every={eval_every} max_iters={max_iters} seed={seed}",
                f"Cleared {result.batch_result.num_cleared}/{result.batch_result.num_total} samples "
                f"({result.batch_result.num_cleared / max(1, result.batch_result.num_total):.1%}), "
                f"{result.batch_result.num_edges_changed} edges changed",
                f"Baseline {eval_split} accuracy: {result.baseline.accuracy:.4f}; "
                f"Final accuracy: {result.final_metrics.accuracy:.4f}; "
                f"Drop: {result.acc_drop:+.4f} (budget: {max_acc_drop})",
                f"Rolled back: {result.rolled_back}; Stopped reason: {result.stopped_reason}; "
                f"Protect sample size: {result.protect_sample_size}",
                _rounds_table(result.rounds),
                f"Run log: {log_path}",
            ],
            md_log_path,
        )
        print(f"Appended run summary to {md_log_path}")

    if save_checkpoint is None:
        save_checkpoint = str(log_dir / "global_optimize_checkpoint.pt")

    if save_checkpoint:
        save_checkpoint = resolve_write_path(save_checkpoint)
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
