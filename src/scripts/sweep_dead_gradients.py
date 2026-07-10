"""Check which misclassified samples have a usable (non-zero) casebase
gradient before running the contest() search on them.

contest() is a local, gradient-based search over model.A: if the analytic
G-RAE (see counterfactuals/grae.py) is uniformly zero for a sample's target
class -- e.g. because a hard-ReLU node on the path to that class's default
argument is saturated -- no amount of iterating will move it. This script
scans outputs/misclassified_qbaf.json and reports, per sample, the current
target-class strength and the max |gradient| magnitude, so "dead" samples
can be skipped in favour of ones contest() can actually make progress on.

Usage::

    python src/scripts/sweep_dead_gradients.py \\
        --checkpoint outputs/model_checkpoint.pt \\
        --qbaf outputs/misclassified_qbaf.json
"""

import argparse
import json

import torch

from deeparguing.counterfactuals.grae import compute_grae
from deeparguing.counterfactuals.run_contest import load_model, load_sample

LIVE_THRESHOLD = 1e-9  # max|grad| above this counts as "live"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/model_checkpoint.pt")
    parser.add_argument("--qbaf", default="outputs/misclassified_qbaf.json")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit the scan to the first N samples (default: all in the file).",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    no_classes = len(model.default_indexes)

    with open(args.qbaf, "r", encoding="utf-8") as f:
        qbaf = json.load(f)
    n = len(qbaf["new_cases"])
    if args.num_samples is not None:
        n = min(n, args.num_samples)

    print(f"Scanning {n} misclassified samples from {args.qbaf} ...\n")
    print(f"{'idx':>4}  {'true_cls':>8}  {'strength':>10}  {'max|grad|':>12}  live?")
    print("-" * 55)

    live_indices = []
    for i in range(n):
        sample, true_class = load_sample(args.qbaf, i, args.device)
        target_index = no_classes - 1 - true_class

        with torch.no_grad():
            strength = model(sample)[0, target_index].item()

        result = compute_grae(model, sample, target_indices=[target_index])
        max_abs_grad = result.casebase_edges.reshape(-1).abs().max().item()
        is_live = max_abs_grad > LIVE_THRESHOLD
        if is_live:
            live_indices.append(i)

        print(
            f"{i:>4}  {true_class:>8}  {strength:>10.4f}  {max_abs_grad:>12.6g}  "
            f"{'yes' if is_live else 'no'}"
        )

    print("\n" + "-" * 55)
    print(f"{len(live_indices)}/{n} samples have a live casebase gradient.")
    print(f"Live sample indices: {live_indices}")


if __name__ == "__main__":
    main()
