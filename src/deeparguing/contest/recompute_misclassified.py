"""Recompute the misclassified-sample QBAF export for an already-fitted
checkpoint, without going through ``cli/run.py``'s full train loop.

``cli/run.py --misclassified_log`` builds ``outputs/qbaf/misclassified_qbaf.json``
as one step of a training run, against whatever ``model.A`` training just
produced. That set goes stale the moment ``model.A`` changes by some other
means -- e.g. ``contest/prune_edges.py`` hard-thresholding weak edges out of
a checkpoint. This script closes that gap: given any checkpoint (pruned,
contested, or a fresh training checkpoint), it rebuilds the live model +
its data split the same way ``run_contest.py``/``run_global_contest_eval.py``
do, runs inference on ``--split`` to find which samples the *checkpoint as
given* gets wrong, and exports exactly those samples as a QBAF file in the
same shape ``contest_all.py``/``run_contest.py`` already consume (a
``new_cases``/``new_cases_labels`` pair -- see ``run_contest.load_all_samples``).

GRAE attribution is intentionally not computed here (unlike
``cli/run.py --grae_log``) since neither ``contest_all.py`` nor ``contest.py``
read it; add ``--grae`` support later if some other consumer needs it.

Usage::

    python -m deeparguing.contest.recompute_misclassified \\
        --checkpoint outputs/checkpoints/pruned_model_checkpoint.pt \\
        --output outputs/qbaf/pruned_misclassified_qbaf.json
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from deeparguing.contest.run_contest import load_fitted_model_and_data

DEFAULT_CHECKPOINT = "outputs/checkpoints/pruned_model_checkpoint.pt"
DEFAULT_OUTPUT = "outputs/qbaf/pruned_misclassified_qbaf.json"


def find_misclassified(
    model, X: torch.Tensor, y: torch.Tensor, batch_size: int | None = None
) -> np.ndarray:
    """Run batched inference and return the indices where the model's argmax
    prediction disagrees with the argmax ground-truth label (same comparison
    ``cli/run.py``'s ``--misclassified_log`` block uses)."""
    model.eval()
    current_batch_size = batch_size if batch_size is not None else len(X)

    all_preds = []
    for i in range(0, len(X), current_batch_size):
        batch_preds = model(X[i : i + current_batch_size]).cpu().detach().numpy()
        all_preds.append(batch_preds)

    y_predicted_classes = np.argmax(np.concatenate(all_preds, axis=0), axis=1)
    y_true_classes = np.argmax(y.cpu().detach().numpy(), axis=1)
    return np.where(y_predicted_classes != y_true_classes)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "val"],
        help="Which held-out split (from the checkpoint's own data config) to "
        "run inference on.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-misclassified",
        type=int,
        default=None,
        help="Cap the number of misclassified samples exported (default: all).",
    )
    parser.add_argument("--batch-size", type=int, default=None)
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

    logging.info(f"Loading checkpoint from {args.checkpoint} ...")
    model, data_dict = load_fitted_model_and_data(args.checkpoint, args.device)
    X = data_dict[f"X_{args.split}"]
    y = data_dict[f"y_{args.split}"]

    logging.info(f"Running inference on the {args.split} split ({X.shape[0]} samples)...")
    misclassified_indices = find_misclassified(model, X, y, batch_size=args.batch_size)

    num_to_extract = (
        len(misclassified_indices)
        if args.max_misclassified is None
        else min(args.max_misclassified, len(misclassified_indices))
    )
    selected_indices = misclassified_indices[:num_to_extract]
    logging.info(
        f"{len(misclassified_indices)}/{X.shape[0]} samples misclassified "
        f"({len(misclassified_indices) / X.shape[0]:.1%}); exporting {num_to_extract}"
    )

    X_misc = X[selected_indices]
    y_misc = y[selected_indices]

    image_mean = data_dict.get("image_mean", None)
    image_std = data_dict.get("image_std", None)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Triggers a forward pass internally to populate new_cases_base_scores
    # and new_cases_attacks_adjacency before exporting (see
    # GradualAACBR.export_to_json).
    model.export_to_json(
        str(output_path),
        image_mean=image_mean,
        image_std=image_std,
        new_cases=X_misc,
        new_cases_labels=y_misc,
        batch_size=args.batch_size,
    )
    logging.info(
        f"Exported {num_to_extract} misclassified samples and their QBAF "
        f"tensors to {output_path}"
    )


if __name__ == "__main__":
    main()
