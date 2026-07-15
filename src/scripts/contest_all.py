"""Run a single joint optimization over every misclassified sample against a
shared model.A, instead of contesting each sample sequentially.

The old sequential loop (see git history) ran ``contest()`` once per sample
against the same live ``model.A``, so edits accumulated one sample at a
time and one sample's fix could partially undo another's. This script
instead calls ``joint_contest()``, which optimizes one shared adjacency
edit against every sample's hinge loss at once -- see
``counterfactuals/joint_contest.py``'s module docstring for the full
algorithm (shared gradient, shared top-k edge selection, shared step).

Usage::

    python src/scripts/contest_all.py \\
        --checkpoint outputs/model_checkpoint.pt \\
        --qbaf outputs/misclassified_qbaf.json

By default this writes two things to ``--log-dir`` (``outputs/contestation``):
a timestamped JSON log (run config, summary counts, and a per-sample
cleared/failed breakdown) and a checkpoint holding the new, contested
``model.A`` -- pass ``--save-checkpoint ""`` to skip the latter.

This runs one joint optimization over every misclassified sample (each
outer iteration involves a batched forward pass, a batched backward pass,
and a line search) -- run it on whatever machine has the compute for that,
not necessarily this one.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from deeparguing.counterfactuals.contest import DEFAULT_K, MARGIN, MAX_ITERS, THRESHOLD
from deeparguing.counterfactuals.joint_contest import (DIVERGENCE_BOUND, TOL,
                                                         joint_contest)
from deeparguing.counterfactuals.run_contest import load_model
from misclassified_grae_by_class import load_all_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/model_checkpoint.pt")
    parser.add_argument("--qbaf", default="outputs/misclassified_qbaf.json")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit the run to the first N misclassified samples (default: all).",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--margin", type=float, default=MARGIN)
    parser.add_argument("--max-iters", type=int, default=MAX_ITERS)
    parser.add_argument("--tol", type=float, default=TOL)
    parser.add_argument(
        "--max-edits",
        type=int,
        default=None,
        help="Stop once this many distinct edges have been touched (default: unbounded).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="If given, use shuffled mini-batches of this size (one step per "
        "batch, reshuffled every pass) instead of the full-batch default.",
    )
    parser.add_argument(
        "--divergence-bound",
        type=float,
        default=DIVERGENCE_BOUND,
        help="Reject a line-search trial if it pushes any active sample's "
        "target strength above this (ReluSemantics has no upper saturation).",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--log-dir",
        default="outputs/contestation",
        help="Directory to write the run's JSON log (and default checkpoint "
        "location) to. Created if it doesn't exist.",
    )
    parser.add_argument(
        "--save-checkpoint",
        default=None,
        help="Where to save the model (with its perturbed model.A) after the "
        "run. Defaults to '<log-dir>/contested_checkpoint.pt'; pass an empty "
        "string to skip saving a checkpoint entirely.",
    )
    args = parser.parse_args()

    with open(args.qbaf, "r", encoding="utf-8") as f:
        qbaf = json.load(f)

    model = load_model(args.checkpoint, args.device)
    samples, true_classes = load_all_samples(qbaf, args.device, args.num_samples)

    print(f"Running joint contest over {samples.shape[0]} misclassified samples...")
    result = joint_contest(
        model,
        samples,
        true_classes,
        k=args.k,
        threshold=args.threshold,
        margin=args.margin,
        max_iters=args.max_iters,
        tol=args.tol,
        max_edits=args.max_edits,
        batch_size=args.batch_size,
        divergence_bound=args.divergence_bound,
    )

    print(
        f"\nCleared {result.num_cleared}/{result.num_total} samples "
        f"({result.num_cleared / max(1, result.num_total):.1%}), "
        f"{result.num_edges_changed} edges changed, "
        f"{result.iterations} iterations used"
    )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"contestation_{timestamp}.json"
    log = {
        "config": {
            "checkpoint": args.checkpoint,
            "qbaf": args.qbaf,
            "num_samples": args.num_samples,
            "k": args.k,
            "threshold": args.threshold,
            "margin": args.margin,
            "max_iters": args.max_iters,
            "tol": args.tol,
            "max_edits": args.max_edits,
            "batch_size": args.batch_size,
            "divergence_bound": args.divergence_bound,
        },
        "summary": {
            "num_total": result.num_total,
            "num_cleared": result.num_cleared,
            "success_rate": result.num_cleared / max(1, result.num_total),
            "num_edges_changed": result.num_edges_changed,
            "iterations": result.iterations,
        },
        "samples": [
            {
                "index": i,
                "true_class": true_classes[i],
                "cleared": bool(result.cleared[i]),
                "final_target_strength": result.final_target_strengths[i].item(),
                "final_rival_class": result.final_rival_classes[i],
                "final_rival_strength": result.final_rival_strengths[i].item(),
            }
            for i in range(result.num_total)
        ],
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"Saved run log to {log_path}")

    save_checkpoint_path = args.save_checkpoint
    if save_checkpoint_path is None:
        save_checkpoint_path = str(log_dir / "contested_checkpoint.pt")

    if save_checkpoint_path:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config_paths": torch.load(args.checkpoint, map_location=args.device)["config_paths"],
                "A": model.A,
                "X_train": model.X_train,
                "y_train": model.y_train,
                "default_indexes": model.default_indexes,
            },
            save_checkpoint_path,
        )
        print(f"Saved contested checkpoint (new adjacency matrix in 'A') to {save_checkpoint_path}")


if __name__ == "__main__":
    main()
