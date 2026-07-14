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
        --output outputs/misclassified_grae_by_class.pt \\
        --viz-output outputs/misclassified_grae_by_class_viz.json

The ``--viz-output`` file is the original QBAF export (same
adjacency_matrix/X_train/y_train/etc. the visualizer already reads) plus a
``grae_by_class`` field, shape (num_classes, n, n, d) -- upload it directly
to ``visualizer/index.html`` and toggle "Avg G-RAE by True Class" to render
edge thickness/color by the selected class's average gradient.

Also prints, per class, the top ``--top-k`` edges (default 10) ranked by
|average G-RAE| -- pass ``--top-k 0`` to skip this.

This is a full sweep (one backward pass per misclassified sample) -- run it
on whatever machine has the compute for that, not necessarily this one.
"""

import argparse
import json

import torch

from deeparguing.counterfactuals.grae import compute_grae
from deeparguing.counterfactuals.run_contest import load_model


def load_all_samples(
    qbaf: dict, device: str, num_samples: int | None = None
) -> tuple[torch.Tensor, list[int]]:
    """Pull every misclassified sample + true label out of a loaded QBAF export."""
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


def print_top_edges_by_class(grae_by_class: torch.Tensor, top_k: int) -> None:
    """Print the top-k edges by |average G-RAE|, one ranked list per class."""
    num_classes, n1, n2, d = grae_by_class.shape
    for c in range(num_classes):
        class_grid = grae_by_class[c]
        if torch.isnan(class_grid).all():
            print(f"\nClass {c}: no misclassified samples")
            continue

        flat = torch.nan_to_num(class_grid, nan=0.0).reshape(-1)
        if torch.all(flat == 0):
            print(
                f"\nClass {c}: all gradients are zero (dead) -- likely a "
                "saturated node on the path to this class's default argument"
            )
            continue

        k = min(top_k, flat.numel())
        _, top_flat_idx = flat.abs().topk(k)

        print(f"\nClass {c}: top {k} edges by |avg G-RAE|")
        for rank, idx in enumerate(top_flat_idx.tolist(), start=1):
            # flat is (n1, n2, d) reshaped row-major, so unravel in that order.
            dd = idx % d
            i, j = divmod(idx // d, n2)
            print(f"  {rank:>2}. edge {i} -> {j} (dim {dd}): {flat[idx].item():+.6g}")


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
        "--viz-output",
        default="outputs/misclassified_grae_by_class_viz.json",
        help=(
            "Where to write a visualizer-ready copy of the QBAF export with "
            "an added 'grae_by_class' field. Pass '' to skip this output."
        ),
    )
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
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top edges (by |avg G-RAE|) to print per class. 0 disables this.",
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    num_classes = len(model.default_indexes)

    with open(args.qbaf, "r", encoding="utf-8") as f:
        qbaf = json.load(f)

    samples, true_classes = load_all_samples(qbaf, args.device, args.num_samples)
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

    if args.viz_output:
        # NaN (classes with zero misclassified samples) isn't valid JSON --
        # the visualizer just wants "no signal" for those, so zero them out.
        qbaf["grae_by_class"] = torch.nan_to_num(grae_by_class, nan=0.0).tolist()
        with open(args.viz_output, "w", encoding="utf-8") as f:
            json.dump(qbaf, f)
        print(f"Saved visualizer-ready QBAF + grae_by_class to {args.viz_output}")

    if args.edge is not None:
        src, tgt = args.edge
        print(f"\nEdge {src} -> {tgt}, average G-RAE by true class:")
        for c in range(num_classes):
            value = grae_by_class[c, src, tgt]
            value_str = "no samples" if torch.isnan(value).any() else f"{value.tolist()}"
            print(f"  true class {c}: {value_str}")

    if args.top_k > 0:
        print_top_edges_by_class(grae_by_class, args.top_k)


if __name__ == "__main__":
    main()
