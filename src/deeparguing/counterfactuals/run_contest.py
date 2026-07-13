"""Standalone driver for the single-sample contestability loop in ``contest.py``.

``cli/run.py`` trains/evaluates/exports; it is not the place to interactively
contest one prediction. This script only *consumes* what a
``--misclassified_log`` run already produced -- ``outputs/model_checkpoint.pt``
(weights + fitted casebase, see ``run.py``) and ``outputs/misclassified_qbaf.json``
(the real misclassified samples, already in model-input form) -- rebuilds the
live model, and runs ``contest()`` against one real sample end to end.

Usage::

    python -m deeparguing.counterfactuals.run_contest \\
        --checkpoint outputs/model_checkpoint.pt \\
        --qbaf outputs/misclassified_qbaf.json \\
        --sample-index 0

Requires a checkpoint produced by ``cli/run.py --run_test --misclassified_log``
(re-run the CLI with that flag if ``outputs/model_checkpoint.pt`` doesn't
exist yet).
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from deeparguing.cli.parse_yaml import parse_model_config, read_config_files
from deeparguing.counterfactuals.contest import (DEFAULT_K, MARGIN,
                                                  MAX_ITERS, contest)
from deeparguing.gradual_aacbr import GradualAACBR


def load_model(checkpoint_path: str, device: str) -> GradualAACBR:
    """Rebuild the model architecture from the checkpoint's config and reload it.

    ``state_dict()`` only covers registered parameters/buffers, so the
    fit()-produced attributes (``A``/``X_train``/``default_indexes``) are
    restored separately from the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = read_config_files(checkpoint["config_paths"])
    _, instances = parse_model_config(model_config, trial=None, device=device)
    model = instances["model"]
    if hasattr(model, "to"):
        model = model.to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.A = checkpoint["A"].to(device)
    model.X_train = checkpoint["X_train"].to(device)
    model.y_train = checkpoint["y_train"].to(device)
    model.default_indexes = checkpoint["default_indexes"].to(device)
    model.eval()
    return model


def load_sample(
    qbaf_path: str, sample_index: int, device: str
) -> tuple[torch.Tensor, int]:
    """Pull one misclassified sample (already model-input shaped) out of the QBAF export."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/model_checkpoint.pt")
    parser.add_argument("--qbaf", default="outputs/misclassified_qbaf.json")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help=(
            "Class to push the sample's strength towards. Defaults to the "
            "sample's own ground-truth label (i.e. contest the model's "
            "wrong prediction back towards the correct class)."
        ),
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--margin",
        type=float,
        default=MARGIN,
        help=(
            "How much target_class's strength must exceed the strongest "
            "other class's strength before the search declares victory "
            "(an argmax-relative win margin, not an absolute threshold -- "
            "class strengths are not normalized/mutually exclusive)."
        ),
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
        default="outputs",
        help=(
            "Directory to additionally write a per-run contest log file to "
            "(one line per iteration: edges perturbed, step size, weight "
            "and strength deltas). Console logging is unaffected."
        ),
    )
    args = parser.parse_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_dir) / f"contest_sample{args.sample_index}.log"

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
    # default_indexes rows are ordered the same as X_defaults/y_defaults
    # ("labels.flip([0])"), but "labels" itself (torch.unique(y, dim=0)) is
    # already in reverse-class order for one-hot rows, so the two reversals
    # cancel out: default row for class c sits at c directly (verified
    # against a real checkpoint's y_train[default_indexes] -- no offset).
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
        margin=args.margin,
        max_iters=args.max_iters,
    )

    logging.info(
        f"success={result.success} iterations={result.iterations} "
        f"final_target_strength={result.final_target_strength:.4f} "
        f"final_rival=class{result.final_rival_class}:{result.final_rival_strength:.4f} "
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
            f"rival stayed class{step.old_rival_class}"
            if step.old_rival_class == step.new_rival_class
            else f"rival changed class{step.old_rival_class}->class{step.new_rival_class}"
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
