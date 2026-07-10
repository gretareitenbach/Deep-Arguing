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

import torch

from deeparguing.cli.parse_yaml import parse_model_config, read_config_files
from deeparguing.counterfactuals.contest import (DEFAULT_K, MARGIN,
                                                  MAX_ITERS, THRESHOLD,
                                                  contest)
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
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--margin", type=float, default=MARGIN)
    parser.add_argument("--max-iters", type=int, default=MAX_ITERS)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--log", default="info", choices=["debug", "info", "warning", "error"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log.upper(), format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info(f"Loading model checkpoint from {args.checkpoint} ...")
    model = load_model(args.checkpoint, args.device)

    sample, true_class = load_sample(args.qbaf, args.sample_index, args.device)
    no_classes = len(model.default_indexes)
    target_class = args.target_class if args.target_class is not None else true_class
    # default_indexes rows are ordered labels.flip([0]) at fit time, so the
    # default row for class c sits at len(labels) - 1 - c -- mirrors
    # cli/run.py's target_indices computation for misclassified_grae.pt.
    target_index = no_classes - 1 - target_class

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

    logging.info(
        f"success={result.success} iterations={result.iterations} "
        f"final_strength={result.final_strength:.4f} "
        f"max_weight_delta={result.max_weight_delta:.6f}"
    )
    for i, (edge_ids, alpha, new_weights) in enumerate(result.edge_trace, start=1):
        logging.info(
            f"  step {i}: alpha={alpha:.4g} edges={edge_ids} new_weights={new_weights}"
        )


if __name__ == "__main__":
    main()
