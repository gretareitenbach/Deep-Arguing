"""Run contest() over every misclassified sample against a shared model.A,
optionally using conflict-aware top-k edge selection.

Sequentially contesting many samples against the same casebase adjacency
(``model.A``) means edits accumulate: sample 2's fix can partially undo
sample 1's, if they land on the same edge wanting opposite signs. This
script runs that sequential loop end to end (rather than one sample at a
time via ``run_contest.py``) and, with ``--conflict-lambda > 0``, routes
edge selection through ``contest.select_top_k_conflict_aware`` instead of
plain top-|G-RAE|, which discounts edges that *other* classes rely on in
the opposite direction (see the module docstring on that function, and
``misclassified_grae_by_class.py`` for the per-class gradient data it's
built from).

The conflict map (``grae_by_class``) is computed once up front from the
*original* checkpoint and held fixed for the whole run -- it does not get
recomputed as edges move, so it's a static approximation of "who else wants
this edge," not a live one.

Usage::

    python src/scripts/contest_all.py \\
        --checkpoint outputs/model_checkpoint.pt \\
        --qbaf outputs/misclassified_qbaf.json \\
        --conflict-lambda 0.5

    # Side-by-side comparison against plain (conflict-unaware) contestation:
    python src/scripts/contest_all.py --compare-baseline --conflict-lambda 0.5

This runs contest() once per misclassified sample (each involving several
forward/backward passes) -- run it on whatever machine has the compute for
that, not necessarily this one.
"""

import argparse
import json
from dataclasses import dataclass

import torch
from torch import Tensor

from deeparguing.counterfactuals.contest import (DEFAULT_K, MARGIN,
                                                   MAX_ITERS, THRESHOLD,
                                                   contest)
from deeparguing.counterfactuals.grae import compute_grae
from deeparguing.counterfactuals.run_contest import load_model
from deeparguing.gradual_aacbr import GradualAACBR
from misclassified_grae_by_class import (average_grae_by_true_class,
                                          load_all_samples)


@dataclass
class RunStats:
    successes: int = 0
    total: int = 0
    total_iterations: int = 0
    # edge_id -> ordered list of (sample_index, sign of applied delta)
    edge_history: dict = None

    def __post_init__(self):
        if self.edge_history is None:
            self.edge_history = {}

    @property
    def success_rate(self) -> float:
        return self.successes / self.total if self.total else 0.0

    def record_edge_reversals(self) -> int:
        """Count how many times, across the whole run, an edge that was
        moved one way by an earlier sample got moved the opposite way by a
        later one -- a direct measure of the "everything conflicting"
        symptom this script exists to reduce."""
        reversals = 0
        for signs in self.edge_history.values():
            nonzero = [s for _, s in signs if s != 0]
            for prev, cur in zip(nonzero, nonzero[1:]):
                if prev != cur:
                    reversals += 1
        return reversals

    def top_conflicted_edges(self, top_k: int = 10) -> list[tuple[int, int]]:
        """(edge_id, number of distinct samples that touched it), sorted
        descending -- the edges "being used a ton" the conflict-aware
        selection is meant to spread load away from."""
        counts = [(edge_id, len(signs)) for edge_id, signs in self.edge_history.items()]
        counts.sort(key=lambda x: x[1], reverse=True)
        return counts[:top_k]


def run_all(
    model: GradualAACBR,
    samples: Tensor,
    true_classes: list[int],
    k: int,
    threshold: float,
    margin: float,
    max_iters: int,
    grae_by_class: Tensor | None,
    conflict_lambda: float,
) -> RunStats:
    """Contest every sample in order against the same live model.A, letting
    edits accumulate exactly as they would running contest() in a loop by
    hand."""
    stats = RunStats()
    for i in range(samples.shape[0]):
        sample = samples[i : i + 1]
        true_class = true_classes[i]

        result = contest(
            model,
            sample,
            target_class=true_class,
            k=k,
            threshold=threshold,
            margin=margin,
            max_iters=max_iters,
            grae_by_class=grae_by_class,
            conflict_lambda=conflict_lambda,
        )

        stats.total += 1
        stats.total_iterations += result.iterations
        if result.success:
            stats.successes += 1

        for step in result.edge_trace:
            for edge_id, old_w, new_w in zip(step.edge_ids, step.old_weights, step.new_weights):
                sign = 0
                if new_w > old_w:
                    sign = 1
                elif new_w < old_w:
                    sign = -1
                stats.edge_history.setdefault(edge_id, []).append((i, sign))

        print(
            f"  sample {i:>4} (true class {true_class:>2}): "
            f"{'OK  ' if result.success else 'FAIL'} in {result.iterations:>2} iters, "
            f"final target={result.final_target_strength:.4f}"
        )

    return stats


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
    parser.add_argument(
        "--conflict-lambda",
        type=float,
        default=0.5,
        help="Weight on the conflict penalty during edge selection. 0 = plain top-|G-RAE| "
        "(the original conflict-unaware behavior).",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also run a plain (conflict-lambda=0) pass on a separately loaded model, "
        "and print both results side by side.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--save-checkpoint",
        default=None,
        help="If given, save the model (with its perturbed model.A) here after the "
        "conflict-aware run, e.g. outputs/contested_checkpoint.pt.",
    )
    args = parser.parse_args()

    with open(args.qbaf, "r", encoding="utf-8") as f:
        qbaf = json.load(f)

    def load_and_prep():
        model = load_model(args.checkpoint, args.device)
        samples, true_classes = load_all_samples(qbaf, args.device, args.num_samples)
        return model, samples, true_classes

    print("Computing per-class G-RAE conflict map from the original checkpoint...")
    conflict_model, samples, true_classes = load_and_prep()
    num_classes = len(conflict_model.default_indexes)
    grae_result = compute_grae(
        conflict_model, samples, target_indices=true_classes, per_sample=True
    )
    grae_by_class = average_grae_by_true_class(
        grae_result.casebase_edges, true_classes, num_classes
    )
    grae_by_class = torch.nan_to_num(grae_by_class, nan=0.0)
    del conflict_model  # only needed to build the conflict map

    if args.compare_baseline:
        print(
            f"\n=== Baseline: plain top-|G-RAE| (conflict_lambda=0), "
            f"{samples.shape[0]} samples ==="
        )
        baseline_model, samples, true_classes = load_and_prep()
        baseline_stats = run_all(
            baseline_model, samples, true_classes, args.k, args.threshold,
            args.margin, args.max_iters, grae_by_class=None, conflict_lambda=0.0,
        )
        print(
            f"\nBaseline: {baseline_stats.successes}/{baseline_stats.total} succeeded "
            f"({baseline_stats.success_rate:.1%}), "
            f"avg iters={baseline_stats.total_iterations / max(1, baseline_stats.total):.1f}, "
            f"edge reversals={baseline_stats.record_edge_reversals()}"
        )

    print(
        f"\n=== Conflict-aware (conflict_lambda={args.conflict_lambda}), "
        f"{samples.shape[0]} samples ==="
    )
    model, samples, true_classes = load_and_prep()
    stats = run_all(
        model, samples, true_classes, args.k, args.threshold, args.margin,
        args.max_iters, grae_by_class=grae_by_class, conflict_lambda=args.conflict_lambda,
    )

    print(
        f"\nConflict-aware: {stats.successes}/{stats.total} succeeded "
        f"({stats.success_rate:.1%}), "
        f"avg iters={stats.total_iterations / max(1, stats.total):.1f}, "
        f"edge reversals={stats.record_edge_reversals()}"
    )

    print("\nMost-touched edges (edge_id, # distinct samples that moved it):")
    for edge_id, count in stats.top_conflicted_edges():
        n = model.A.shape[1]
        i, j = divmod(edge_id // model.A.shape[2], n)
        print(f"  edge {i} -> {j}: touched by {count} samples")

    if args.save_checkpoint:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config_paths": torch.load(args.checkpoint, map_location=args.device)["config_paths"],
                "A": model.A,
                "X_train": model.X_train,
                "y_train": model.y_train,
                "default_indexes": model.default_indexes,
            },
            args.save_checkpoint,
        )
        print(f"\nSaved contested checkpoint to {args.save_checkpoint}")


if __name__ == "__main__":
    main()
