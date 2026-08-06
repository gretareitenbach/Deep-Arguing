"""Standalone driver for the single-sample contestability loop in
``contest.py``. Rebuilds a fitted model from a checkpoint and runs
``contest()`` against one real misclassified sample end to end.

Usage::

    python -m deeparguing.contest.scripts.run_contest \\
        --checkpoint model_checkpoint.pt \\
        --qbaf misclassified_qbaf.json \\
        --sample-index 0

Requires a checkpoint produced by ``cli/run.py --run_test --misclassified_log``.
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from deeparguing.cli.parse_yaml import parse_model_config, read_config_files
from deeparguing.contest.core.contest import (DEFAULT_K, MARGIN,
                                               MAX_ITERS, THRESHOLD,
                                               contest)
from deeparguing.gradual_aacbr import GradualAACBR
from deeparguing.output_paths import resolve_read_path, today_output_dir


def load_fitted_model_and_data(
    checkpoint_path: str, device: str
) -> tuple[GradualAACBR, dict]:
    """Rebuild the model architecture from the checkpoint's config and
    reload its weights and fit()-produced state.

    Parameters
    ----------
    checkpoint_path : str
    device : str

    Returns
    -------
    tuple[GradualAACBR, dict]
        The reloaded model, and the config's ``data_dict`` (e.g. for
        callers that need a held-out ``X_<split>``/``y_<split>`` pair).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = read_config_files(checkpoint["config_paths"])
    data_dict, instances = parse_model_config(model_config, trial=None, device=device)
    model = instances["model"]
    if hasattr(model, "to"):
        model = model.to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.A = checkpoint["A"].to(device)
    model.X_train = checkpoint["X_train"].to(device)
    model.y_train = checkpoint["y_train"].to(device)
    model.default_indexes = checkpoint["default_indexes"].to(device)
    model.eval()
    return model, data_dict


def load_model(checkpoint_path: str, device: str) -> GradualAACBR:
    """Rebuild and reload the model from a checkpoint, discarding its config's
    data split."""
    model, _ = load_fitted_model_and_data(checkpoint_path, device)
    return model


def load_sample(
    qbaf_path: str, sample_index: int, device: str
) -> tuple[torch.Tensor, int]:
    """Pull one misclassified sample out of the QBAF export.

    Returns
    -------
    tuple[torch.Tensor, int]
        The model-input-shaped sample (shape (1, ...)) and its true class.
    """
    with open(qbaf_path, "r", encoding="utf-8") as f:
        qbaf = json.load(f)

    if "new_cases" not in qbaf:
        raise ValueError(
            f"{qbaf_path} has no 'new_cases' entry -- re-run the CLI with "
            "--misclassified_log to produce one."
        )
    if not (0 <= sample_index < len(qbaf["new_cases"])):
        raise IndexError(
            f"--sample-index {sample_index} out of range: {qbaf_path} has "
            f"{len(qbaf['new_cases'])} exported samples."
        )

    sample = torch.tensor(
        qbaf["new_cases"][sample_index], dtype=torch.float32, device=device
    ).unsqueeze(0)
    true_class = int(qbaf["new_cases_labels"][sample_index])
    return sample, true_class


def load_all_samples(
    qbaf: dict, device: str, num_samples: int | None = None
) -> tuple[torch.Tensor, list[int]]:
    """Pull misclassified samples + true labels out of a loaded QBAF export.

    Parameters
    ----------
    qbaf : dict
        Loaded QBAF export (must have a ``new_cases`` entry).
    device : str
    num_samples : int | None
        If given, caps the number of samples returned to the first N.

    Returns
    -------
    tuple[torch.Tensor, list[int]]
        Samples (shape (n, ...)) and their true class labels.
    """
    if "new_cases" not in qbaf:
        raise ValueError(
            "qbaf has no 'new_cases' entry -- re-run the CLI with "
            "--misclassified_log to produce one."
        )

    n = len(qbaf["new_cases"])
    if num_samples is not None:
        n = min(n, num_samples)

    samples = torch.tensor(
        qbaf["new_cases"][:n], dtype=torch.float32, device=device
    )
    true_classes = [int(c) for c in qbaf["new_cases_labels"][:n]]
    return samples, true_classes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="model_checkpoint.pt")
    parser.add_argument("--qbaf", default="misclassified_qbaf.json")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Class to push the sample's strength towards. Defaults to the "
        "sample's own ground-truth label.",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--margin",
        type=float,
        default=MARGIN,
        help="How much target_class's strength must exceed the strongest "
        "rival class's strength before the search declares victory.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help="Fallback virtual rival strength, used only if the model has a "
        "single default/topic argument.",
    )
    parser.add_argument("--max-iters", type=int, default=MAX_ITERS)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--log", default="info", choices=["debug", "info", "warning", "error"]
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory to additionally write a per-run contest log file to. "
        "Defaults to today's outputs/<date>/ folder.",
    )
    args = parser.parse_args()

    args.checkpoint = resolve_read_path(args.checkpoint)
    args.qbaf = resolve_read_path(args.qbaf)

    log_dir = Path(args.log_dir) if args.log_dir else today_output_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"contest_sample{args.sample_index}.log"

    logging.basicConfig(
        level=args.log.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
    )

    logging.info(f"Writing run log to {log_path}")
    logging.info(f"Loading model checkpoint from {args.checkpoint} ...")
    model = load_model(args.checkpoint, args.device)

    sample, true_class = load_sample(args.qbaf, args.sample_index, args.device)
    target_class = args.target_class if args.target_class is not None else true_class
    # default row for class c sits at index c directly (no offset)
    target_index = target_class

    logging.info(
        f"Sample {args.sample_index}: true class {true_class}, contesting "
        f"towards class {target_class} (default row {target_index})"
    )

    result = contest(
        model,
        sample,
        target_class=target_index,
        k=args.k,
        threshold=args.threshold,
        margin=args.margin,
        max_iters=args.max_iters,
    )

    def _rival_label(rival_class: int | None) -> str:
        return f"class{rival_class}" if rival_class is not None else "threshold(no rival)"

    logging.info(
        f"success={result.success} iterations={result.iterations} "
        f"final_target_strength={result.final_target_strength:.4f} "
        f"final_rival={_rival_label(result.final_rival_class)}:{result.final_rival_strength:.4f} "
        f"margin_needed={args.margin:.4f} "
        f"max_weight_delta={result.max_weight_delta:.6f}"
    )
    for i, step in enumerate(result.edge_trace, start=1):
        weight_deltas = [
            round(new - old, 6)
            for old, new in zip(step.old_weights, step.new_weights)
        ]
        strength_gain = step.new_target_strength - step.old_target_strength
        old_margin = step.old_target_strength - step.old_rival_strength
        new_margin = step.new_target_strength - step.new_rival_strength
        rival_note = (
            f"rival stayed {_rival_label(step.old_rival_class)}"
            if step.old_rival_class == step.new_rival_class
            else f"rival changed {_rival_label(step.old_rival_class)}->{_rival_label(step.new_rival_class)}"
        )
        logging.info(
            f"  step {i}: edges={step.edge_ids} alpha={step.alpha:.4g} "
            f"weights {step.old_weights} -> {step.new_weights} "
            f"(delta={weight_deltas}) "
            f"target_strength {step.old_target_strength:.4f} -> {step.new_target_strength:.4f} "
            f"(gain {strength_gain:+.4f}); {rival_note} "
            f"({step.old_rival_strength:.4f} -> {step.new_rival_strength:.4f}); "
            f"margin {old_margin:+.4f} -> {new_margin:+.4f} (need >= {args.margin:.4f})"
        )


if __name__ == "__main__":
    main()
