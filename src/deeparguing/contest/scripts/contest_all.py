"""Run a single joint optimization over every misclassified sample against a
shared ``model.A``, via ``batch_contest()``, instead of contesting each
sample sequentially. Hyperparameters and dataset/checkpoint paths come from
a YAML config file (default ``tuning/contest/contest.yaml``); any CLI flag
overrides the corresponding config value.

Usage::

    python -m deeparguing.contest.scripts.contest_all
    python -m deeparguing.contest.scripts.contest_all --config tuning/contest/contest.yaml
    python -m deeparguing.contest.scripts.contest_all --k 10 --margin 0.005
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from deeparguing.contest.core.contest import DEFAULT_K, MARGIN, MAX_ITERS, THRESHOLD
from deeparguing.contest.core.batch_contest import (ALPHA_INIT,
                                                     DIVERGENCE_BOUND,
                                                     MAX_BACKTRACKS, TOL,
                                                     batch_contest)
from deeparguing.contest.scripts.config_cli import load_config, required, resolved
from deeparguing.contest.scripts.run_contest import load_all_samples, load_model
from deeparguing.output_paths import resolve_read_path, today_output_dir

DEFAULT_CONFIG_PATH = "tuning/contest/contest.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="YAML file holding hyperparameters and paths (see tuning/contest/contest.yaml). "
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
        default=None,
        help="Reject a line-search trial if it pushes any active sample's "
        "target strength above this (ReluSemantics has no upper saturation).",
    )
    parser.add_argument(
        "--alpha-init",
        type=float,
        default=None,
        help="Initial (largest) step size the line search backtracks from.",
    )
    parser.add_argument(
        "--max-backtracks",
        type=int,
        default=None,
        help="Line-search retry cap per outer iteration.",
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
        help="Where to save the model (with its perturbed model.A) after the "
        "run. Defaults to '<log-dir>/<checkpoint-filename>'; pass an empty "
        "string to skip saving a checkpoint entirely.",
    )
    parser.add_argument(
        "--checkpoint-filename",
        default=None,
        help="Filename (under log-dir) for the saved checkpoint. Default: "
        "'contested_checkpoint.pt'. Override this in a variant config (e.g. "
        "'pruned_contested_checkpoint.pt') so it doesn't collide with "
        "another variant's output landing in the same date folder.",
    )
    parser.add_argument(
        "--log-prefix",
        default=None,
        help="Filename prefix (under log-dir) for the timestamped JSON log. "
        "Default: 'contestation'. Same collision-avoidance purpose as "
        "--checkpoint-filename.",
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
    device = resolved(args.device, config, "device", "cuda" if torch.cuda.is_available() else "cpu")
    log_dir_str = resolved(args.log_dir, config, "log_dir", str(today_output_dir()))
    checkpoint_filename = resolved(args.checkpoint_filename, config, "checkpoint_filename", "contested_checkpoint.pt")
    log_prefix = resolved(args.log_prefix, config, "log_prefix", "contestation")
    save_checkpoint = args.save_checkpoint if args.save_checkpoint is not None else config.get("save_checkpoint")

    with open(qbaf, "r", encoding="utf-8") as f:
        qbaf_data = json.load(f)

    model = load_model(checkpoint, device)
    assert model.A is not None, "checkpoint's model was never fit()"
    original_A = model.A.detach().clone()
    samples, true_classes = load_all_samples(qbaf_data, device, num_samples)

    print(f"Running joint contest over {samples.shape[0]} misclassified samples...")
    result = batch_contest(
        model,
        samples,
        true_classes,
        k=k,
        threshold=threshold,
        margin=margin,
        max_iters=max_iters,
        tol=tol,
        max_edits=max_edits,
        batch_size=batch_size,
        divergence_bound=divergence_bound,
        alpha_init=alpha_init,
        max_backtracks=max_backtracks,
    )

    print(
        f"\nCleared {result.num_cleared}/{result.num_total} samples "
        f"({result.num_cleared / max(1, result.num_total):.1%}), "
        f"{result.num_edges_changed} edges changed, "
        f"{result.iterations} iterations used"
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
        for idx in result.touched_edge_indices
    ]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"{log_prefix}_{timestamp}.json"
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
        "touched_edges": touched_edges,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"Saved run log to {log_path}")

    if save_checkpoint is None:
        save_checkpoint = str(log_dir / checkpoint_filename)

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
        print(f"Saved contested checkpoint (new adjacency matrix in 'A') to {save_checkpoint}")


if __name__ == "__main__":
    main()
