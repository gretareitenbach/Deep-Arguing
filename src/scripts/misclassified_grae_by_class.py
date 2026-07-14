"""Average casebase-edge G-RAEs for misclassified samples, grouped by true class.

For every misclassified sample exported by ``cli/run.py --misclassified_log``,
computes the per-sample casebase G-RAE (gradient of that sample's *true*
class strength w.r.t. every edge in ``model.A``, see
``counterfactuals/grae.py``), then averages those per-sample edge gradients
within each true class.

The result is a ``(num_classes, n, n, d)`` tensor: ``result[c, i, j]`` is the
average gradient of edge ``i -> j`` over all misclassified samples whose true
label is ``c``. E.g. to check edge 56 -> 72 for true class 7::

    result[7, 56, 72]

Usage::

    python src/scripts/misclassified_grae_by_class.py \\
        --checkpoint outputs/model_checkpoint.pt \\
        --qbaf outputs/misclassified_qbaf.json \\
        --output outputs/misclassified_grae_by_class.pt

This is a full sweep (one backward pass per misclassified sample) -- run it
on whatever machine has the compute for that, not necessarily this one.
"""

import argparse
import json

import torch

from deeparguing.counterfactuals.grae import compute_grae
from deeparguing.counterfactuals.run_contest import load_model


def load_all_samples(
    qbaf_path: str, device: str, num_samples: int | None = None
) -> tuple[torch.Tensor, list[int]]:
    """Pull every misclassified sample + true label out of the QBAF export."""
    with open(qbaf_path, "r", encoding="utf-8") as f:
        qbaf = json.load(f)

    if "new_cases" not in qbaf:
        raise ValueError(
            f"{qbaf_path} has no 'new_cases' entry -- re-run the CLI with "
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


def average_grae_by_true_class(
    casebase_edges: torch.Tensor, true_classes: list[int], num_classes: int
) -> torch.Tensor:
    """Average per-sample casebase edge gradients within each true class.

    Parameters
    ----------
    casebase_edges : Tensor
        Per-sample casebase G-RAEs, shape (B, n, n, d) -- i.e.
        ``GRAEResult.casebase_edges`` from ``compute_grae(..., per_sample=True)``.
    true_classes : list[int]
        Length-B true label for each sample.
    num_classes : int
        Total number of classes, used to size the output's leading dim (so
        classes with zero misclassified samples still get an all-NaN slot
        rather than shifting indices).

    Returns
    -------
    Tensor
        Shape (num_classes, n, n, d). Entry [c] is the mean over all samples
        with true_classes[i] == c; classes with no samples are filled with
        NaN.
    """
    n1, n2, d = casebase_edges.shape[1:]
    out = torch.full(
        (num_classes, n1, n2, d), float("nan"), dtype=casebase_edges.dtype
    )
    true_classes_t = torch.tensor(true_classes, dtype=torch.long)
    for c in range(num_classes):
        mask = true_classes_t == c
        if mask.any():
            out[c] = casebase_edges[mask].mean(dim=0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/model_checkpoint.pt")
    parser.add_argument("--qbaf", default="outputs/misclassified_qbaf.json")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit the sweep to the first N misclassified samples (default: all).",
    )
    parser.add_argument("--output", default="outputs/misclassified_grae_by_class.pt")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--edge",
        type=int,
        nargs=2,
        default=None,
        metavar=("SOURCE", "TARGET"),
        help="If given, print the per-class average gradient for this one edge.",
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    num_classes = len(model.default_indexes)

    samples, true_classes = load_all_samples(args.qbaf, args.device, args.num_samples)
    print(f"Loaded {samples.shape[0]} misclassified samples from {args.qbaf}")

    # target_indices = true_classes: differentiate each sample's *own* true
    # class strength w.r.t. every casebase edge, not the (wrong) predicted one.
    result = compute_grae(
        model, samples, target_indices=true_classes, per_sample=True
    )

    grae_by_class = average_grae_by_true_class(
        result.casebase_edges, true_classes, num_classes
    )

    torch.save(
        {"grae_by_class": grae_by_class, "num_classes": num_classes},
        args.output,
    )
    print(f"Saved (num_classes, n, n, d) averaged G-RAEs to {args.output}")

    if args.edge is not None:
        src, tgt = args.edge
        print(f"\nEdge {src} -> {tgt}, average G-RAE by true class:")
        for c in range(num_classes):
            value = grae_by_class[c, src, tgt]
            value_str = "no samples" if torch.isnan(value).any() else f"{value.tolist()}"
            print(f"  true class {c}: {value_str}")


if __name__ == "__main__":
    main()
