"""Prune weak edges out of a fitted model's casebase adjacency (model.A)
by hard-thresholding on magnitude as a first pass before contesting.

Usage::

    python -m deeparguing.contest.scripts.prune_edges \\
        --checkpoint model_checkpoint.pt \\
        --threshold 0.1 \\
        --output pruned_model_checkpoint.pt
"""

import argparse
from dataclasses import dataclass

import torch
from torch import Tensor

from deeparguing.output_paths import resolve_read_path, resolve_write_path

DEFAULT_THRESHOLD = 0.1
DEFAULT_CHECKPOINT = "model_checkpoint.pt"
DEFAULT_OUTPUT = "pruned_model_checkpoint.pt"


@dataclass(frozen=True)
class PruneResult:
    pruned_A: Tensor
    num_edges_before: int
    num_edges_after: int
    threshold: float

    @property
    def num_pruned(self) -> int:
        return self.num_edges_before - self.num_edges_after


def prune_edges(A: Tensor, threshold: float = DEFAULT_THRESHOLD) -> PruneResult:
    """Zero every entry of A with abs(weight) < threshold.

    Parameters
    ----------
    A : Tensor
        Casebase adjacency.
    threshold : float
        Magnitude cutoff; entries below this are zeroed regardless of sign.

    Returns
    -------
    PruneResult
    """
    num_edges_before = int(A.count_nonzero().item())
    pruned_A = A.clone()
    pruned_A[pruned_A.abs() < threshold] = 0.0
    num_edges_after = int(pruned_A.count_nonzero().item())
    return PruneResult(
        pruned_A=pruned_A,
        num_edges_before=num_edges_before,
        num_edges_after=num_edges_after,
        threshold=threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Prune edges with abs(weight) below this value.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    checkpoint_path = resolve_read_path(args.checkpoint)
    output_path = resolve_write_path(args.output)

    print(f"Loading checkpoint from {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)

    result = prune_edges(checkpoint["A"], threshold=args.threshold)
    print(
        f"Pruned {result.num_pruned}/{result.num_edges_before} edges "
        f"(threshold={result.threshold:g}), {result.num_edges_after} remain"
    )

    checkpoint["A"] = result.pruned_A
    torch.save(checkpoint, output_path)
    print(f"Saved pruned checkpoint to {output_path}")


if __name__ == "__main__":
    main()
