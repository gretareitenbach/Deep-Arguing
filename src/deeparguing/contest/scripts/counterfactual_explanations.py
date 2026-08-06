"""Generate human-readable counterfactual explanations for misclassified
samples: "if these edges were this much higher/lower, this would have been
predicted correctly."

For each sample, runs the single-sample contestability search (``contest()``
in ``core/contest.py``) to find the minimal set of ``model.A`` edge edits
that flips the prediction, records what changed, then undoes the edit --
this is a probe, not a real edit, so ``model.A`` is left exactly as it was
found. Appends one markdown section per sample to a report.

Usage::

    python -m deeparguing.contest.scripts.counterfactual_explanations
    python -m deeparguing.contest.scripts.counterfactual_explanations \\
        --checkpoint model_checkpoint.pt \\
        --qbaf misclassified_qbaf.json \\
        --sample-index 0
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone

import torch
from torch import Tensor
from tqdm import tqdm

from deeparguing.contest.core.contest import (DEFAULT_K, MARGIN, MAX_ITERS,
                                               THRESHOLD, contest)
from deeparguing.contest.scripts.run_contest import (load_all_samples,
                                                      load_model)
from deeparguing.gradual_aacbr import GradualAACBR
from deeparguing.md_log import write_markdown_log
from deeparguing.output_paths import resolve_read_path, resolve_write_path

DEFAULT_OUTPUT_FILENAME = "counterfactual_explanations.md"


@dataclass(frozen=True)
class EdgeChange:
    """One edge ``contest()`` touched: its flat ``model.A`` index, decoded
    (source_case, target_case, head), and its weight before/after."""

    edge_id: int
    source_case: int
    target_case: int
    head: int
    old_weight: float
    new_weight: float

    @property
    def delta(self) -> float:
        return self.new_weight - self.old_weight


@dataclass
class CounterfactualExplanation:
    """Outcome of ``explain_sample()``: whether a counterfactual was found
    for ``sample_index``, and if so (or even if not -- partial progress is
    still informative), every edge that was tried."""

    sample_index: int
    true_class: int
    target_class: int
    success: bool
    iterations: int
    final_target_strength: float | None
    final_rival_class: int | None
    final_rival_strength: float | None
    edges: list[EdgeChange]


def _decode_edge(edge_id: int, n2: int, d: int) -> tuple[int, int, int]:
    """Flat ``model.A`` index -> (source_case, target_case, head); same
    (n, n, d) unravel convention ``contest_all.py``'s touched-edge log uses."""
    return (edge_id // d) // n2, (edge_id // d) % n2, edge_id % d


def _case_label(model: GradualAACBR, default_index_set: set[int], case_index: int) -> str:
    """Human-readable tag for a casebase row: its class label plus whether
    it's one of the model's default/topic arguments.

    ``y_train`` is one-hot (shape (N, Y>1)) on real fitted models, but a
    single scalar column (Y==1) in some synthetic test fixtures -- handle
    both instead of assuming ``argmax`` is always right.
    """
    y_row = model.y_train[case_index]
    label = int(y_row.item()) if y_row.numel() == 1 else int(y_row.argmax().item())
    tag = ", default" if case_index in default_index_set else ""
    return f"case #{case_index} (label {label}{tag})"


def explain_sample(
    model: GradualAACBR,
    sample: Tensor,
    sample_index: int,
    true_class: int,
    target_class: int,
    k: int = DEFAULT_K,
    threshold: float = THRESHOLD,
    margin: float = MARGIN,
    max_iters: int = MAX_ITERS,
    max_edits: int | None = None,
) -> CounterfactualExplanation:
    """Run ``contest()`` against one sample to discover its minimal edge
    edit, then undo it -- ``model.A`` is restored to exactly what it was
    before this call, regardless of whether a counterfactual was found.

    ``max_edits`` (see ``contest()``) caps how many distinct edges the
    search may touch, at the cost of possibly not finding a counterfactual
    within that budget -- useful for keeping the explanation small enough
    to read, since an unrestricted search can revisit/introduce dozens of
    edges across iterations.

    Returns a ``CounterfactualExplanation`` describing which edges would
    need to change, and by how much, for ``model`` to predict
    ``target_class`` for ``sample``.
    """
    assert model.A is not None, "model was never fit()"
    original_A = model.A.detach().clone()
    _, n2, d = original_A.shape

    result = contest(
        model, sample, target_class=target_class,
        k=k, threshold=threshold, margin=margin, max_iters=max_iters, max_edits=max_edits,
    )

    final_A = model.A
    touched = sorted({edge_id for step in result.edge_trace for edge_id in step.edge_ids})
    original_flat = original_A.reshape(-1)
    final_flat = final_A.reshape(-1)
    edges = [
        EdgeChange(
            edge_id,
            *_decode_edge(edge_id, n2, d),
            original_flat[edge_id].item(),
            final_flat[edge_id].item(),
        )
        for edge_id in touched
    ]

    model.A = original_A  # undo -- this is a probe, not a real edit

    return CounterfactualExplanation(
        sample_index, true_class, target_class,
        result.success, result.iterations,
        result.final_target_strength, result.final_rival_class, result.final_rival_strength,
        edges,
    )


def _edges_table(
    model: GradualAACBR, default_index_set: set[int], edges: list[EdgeChange], d: int
) -> str:
    lines = [
        "| Edge | Old weight | New weight | Delta |",
        "|---|---|---|---|",
    ]
    for e in edges:
        source = _case_label(model, default_index_set, e.source_case)
        target = _case_label(model, default_index_set, e.target_case)
        head_note = f" (head {e.head})" if d > 1 else ""
        lines.append(
            f"| {source} -> {target}{head_note} "
            f"| {e.old_weight:+.4f} | {e.new_weight:+.4f} | {e.delta:+.4f} |"
        )
    return "\n".join(lines)


def _rival_label(rival_class: int | None) -> str:
    return f"class {rival_class}" if rival_class is not None else "threshold (no rival)"


def render_sample(
    model: GradualAACBR,
    default_index_set: set[int],
    explanation: CounterfactualExplanation,
    d: int,
    max_iters: int,
    margin: float,
) -> list[str]:
    """Markdown lines for one sample's section, in the format
    ``write_markdown_log`` expects (see ``md_log.py``)."""
    lines = [
        f"--- Sample {explanation.sample_index} (true class {explanation.true_class}) ---",
        f"Contested towards class {explanation.target_class}",
    ]

    if explanation.success:
        lines.append(
            f"Counterfactual found in {explanation.iterations} iteration(s): "
            f"{len(explanation.edges)} edge(s) changed. Final target strength "
            f"{explanation.final_target_strength:.4f} vs "
            f"{_rival_label(explanation.final_rival_class)} "
            f"{explanation.final_rival_strength:.4f}."
        )
    else:
        gap = explanation.final_target_strength - explanation.final_rival_strength
        lines.append(
            f"No counterfactual found within {max_iters} iterations "
            f"(need >= {margin:.4f} margin, still {gap:+.4f}). "
            f"{len(explanation.edges)} edge(s) were tried."
        )

    if explanation.edges:
        lines.append(_edges_table(model, default_index_set, explanation.edges, d))
    else:
        lines.append("No edges were touched.")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="model_checkpoint.pt")
    parser.add_argument("--qbaf", default="misclassified_qbaf.json")
    parser.add_argument(
        "--sample-index", type=int, default=None,
        help="Explain only this one misclassified sample (default: all of them).",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Limit to the first N misclassified samples (default: all). "
        "Ignored if --sample-index is given.",
    )
    parser.add_argument(
        "--target-class", type=int, default=None,
        help="Class to contest every sample towards. Defaults to each "
        "sample's own ground-truth label.",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--margin", type=float, default=MARGIN)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--max-iters", type=int, default=MAX_ITERS)
    parser.add_argument(
        "--max-edits", type=int, default=None,
        help="Cap each sample's explanation to at most this many distinct "
        "edges (default: unbounded). A search that hits the cap before "
        "reaching the margin is reported as 'no counterfactual found' with "
        "whatever partial edit it got to -- use this to keep explanations "
        "small enough to read, at the cost of some samples no longer "
        "finding a counterfactual at all.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_FILENAME,
        help="Markdown file to append explanations to (today's outputs/ folder "
        "unless given a path with a directory component).",
    )
    args = parser.parse_args()

    checkpoint = resolve_read_path(args.checkpoint)
    qbaf_path = resolve_read_path(args.qbaf)
    output = resolve_write_path(args.output)

    with open(qbaf_path, "r", encoding="utf-8") as f:
        qbaf_data = json.load(f)

    model = load_model(checkpoint, args.device)
    assert model.A is not None, "checkpoint's model was never fit()"
    default_index_set = set(model.default_indexes.tolist())
    _, _, d = model.A.shape

    num_samples_to_load = (
        args.sample_index + 1 if args.sample_index is not None else args.num_samples
    )
    samples, true_classes = load_all_samples(qbaf_data, args.device, num_samples_to_load)

    if args.sample_index is not None:
        if not (0 <= args.sample_index < samples.shape[0]):
            raise IndexError(
                f"--sample-index {args.sample_index} out of range: {qbaf_path} "
                f"has {samples.shape[0]} misclassified samples."
            )
        sample_indices = [args.sample_index]
    else:
        sample_indices = range(samples.shape[0])
    total = len(sample_indices)

    write_markdown_log(
        [
            "--- COUNTERFACTUAL EXPLANATIONS ---",
            f"Run: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            f"Checkpoint: {checkpoint}",
            f"QBAF: {qbaf_path}",
            f"Config: k={args.k} margin={args.margin} threshold={args.threshold} "
            f"max_iters={args.max_iters} max_edits={args.max_edits}",
            f"Samples: {total} of {samples.shape[0]} misclassified",
        ],
        output,
    )

    num_found = 0
    processed = 0
    progress = tqdm(sample_indices, desc="Explaining samples", unit="sample")
    for i in progress:
        sample = samples[i : i + 1]
        true_class = true_classes[i]
        target_class = args.target_class if args.target_class is not None else true_class

        explanation = explain_sample(
            model, sample, i, true_class, target_class,
            k=args.k, threshold=args.threshold, margin=args.margin, max_iters=args.max_iters,
            max_edits=args.max_edits,
        )
        num_found += explanation.success
        processed += 1
        progress.set_postfix(found=f"{num_found}/{processed}")

        write_markdown_log(
            render_sample(model, default_index_set, explanation, d, args.max_iters, args.margin),
            output,
        )

    print(f"\n{num_found}/{total} samples had a counterfactual; appended to {output}")


if __name__ == "__main__":
    main()
