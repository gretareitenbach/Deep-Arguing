"""Run a single joint optimization over every misclassified sample against a
shared model.A, instead of contesting each sample sequentially.

The old sequential loop (see git history) ran ``contest()`` once per sample
against the same live ``model.A``, so edits accumulated one sample at a
time and one sample's fix could partially undo another's. This script
instead calls ``batch_contest()``, which optimizes one shared adjacency
edit against every sample's hinge loss at once -- see
``counterfactuals/batch_contest.py``'s module docstring for the full
algorithm (shared gradient, shared top-k edge selection, shared step).

All hyperparameters and dataset/checkpoint paths live in a YAML config file
(default ``tuning/contest/contest.yaml``, which holds the winning config
found by an earlier grid search -- see git history, the sweep script itself
has since been removed) rather than being hardcoded here. Any CLI flag, if
given, overrides the corresponding value from that file for a one-off run
without editing it.

Usage::

    python -m deeparguing.counterfactuals.contest_all
    python -m deeparguing.counterfactuals.contest_all --config tuning/contest/contest.yaml
    python -m deeparguing.counterfactuals.contest_all --k 10 --margin 0.005   # one-off override

By default this writes two things to ``log_dir`` (``outputs/contestation``):
a timestamped JSON log (run config, summary counts, and a per-sample
cleared/failed breakdown) and a checkpoint holding the new, contested
``model.A`` -- set ``save_checkpoint: ""`` (or ``--save-checkpoint ""``) to
skip the latter.

This runs one joint optimization over every misclassified sample (each
outer iteration involves a batched forward pass, a batched backward pass,
and a line search) -- run it on whatever machine has the compute for that,
not necessarily this one.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from deeparguing.counterfactuals.contest import DEFAULT_K, MARGIN, MAX_ITERS, THRESHOLD
from deeparguing.counterfactuals.batch_contest import (ALPHA_INIT,
                                                         DIVERGENCE_BOUND,
                                                         MAX_BACKTRACKS, TOL,
                                                         batch_contest)
from deeparguing.counterfactuals.run_contest import load_all_samples, load_model

DEFAULT_CONFIG_PATH = "tuning/contest/contest.yaml"


def _load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        print(
            f"Warning: config file {config_path} not found -- proceeding with "
            "CLI flags and library defaults only."
        )
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolved(cli_value: Any, config: dict[str, Any], key: str, fallback: Any) -> Any:
    """CLI flag (if given) overrides the config file's value, which
    overrides ``fallback``. A ``null`` in the config file is treated the
    same as the key being absent (falls through to ``fallback``), since
    both mean "nothing was actually specified" for every setting here."""
    if cli_value is not None:
        return cli_value
    if config.get(key) is not None:
        return config[key]
    return fallback


def _required(cli_value: Any, config: dict[str, Any], key: str, config_path: str) -> Any:
    value = cli_value if cli_value is not None else config.get(key)
    if value is None:
        raise ValueError(
            f"'{key}' was not given on the command line and is not set in "
            f"{config_path} -- add it there or pass --{key.replace('_', '-')}."
        )
    return value


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
        "run. Defaults to '<log-dir>/contested_checkpoint.pt'; pass an empty "
        "string to skip saving a checkpoint entirely.",
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
    device = _resolved(args.device, config, "device", "cuda" if torch.cuda.is_available() else "cpu")
    log_dir_str = _resolved(args.log_dir, config, "log_dir", "outputs/contestation")
    # save_checkpoint's tri-state (unset -> default path, "" -> skip, path ->
    # explicit) means an empty string from the config must NOT fall through
    # to _resolved's "treat null/absent as unset" rule, so it's handled
    # directly rather than via _resolved.
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

    # Unravel each touched flat index into (source, target, dim) plus its
    # before/after weight -- lets a later global-accuracy investigation
    # correlate a specific edge with whatever samples depend on it, instead
    # of only knowing how many edges moved in total.
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
    log_path = log_dir / f"contestation_{timestamp}.json"
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
        save_checkpoint = str(log_dir / "contested_checkpoint.pt")

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
