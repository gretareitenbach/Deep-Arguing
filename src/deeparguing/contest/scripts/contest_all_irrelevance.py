"""Run new_case_contest independently over every misclassified sample in a
QBAF export, editing only each sample's own irrelevance row rather than the
shared model adjacency that contest_all.py/batch_contest edits.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from deeparguing.contest.core.contest import DEFAULT_K, MARGIN, MAX_ITERS, THRESHOLD
from deeparguing.contest.core.new_case_contest import new_case_contest
from deeparguing.contest.scripts.config_cli import load_config, required, resolved
from deeparguing.contest.scripts.run_contest import load_all_samples, load_model
from deeparguing.output_paths import (resolve_read_path, resolve_write_path,
                                       today_output_dir)

DEFAULT_CONFIG_PATH = "tuning/contest/contest_all_irrelevance.yaml"


def _touched_edge_triples(
    initial_E: torch.Tensor | None, final_E: torch.Tensor | None
) -> list[dict[str, Any]]:
    """Every (casebase_item, dim, old_E, corrected_E) entry where the sample's
    irrelevance row changed between initial_E and final_E.
    """
    if initial_E is None or final_E is None:
        return []
    changed = torch.nonzero(final_E != initial_E, as_tuple=False)
    return [
        {
            "casebase_item": int(casebase_item),
            "dim": int(dim),
            "old_E": initial_E[casebase_item, dim].item(),
            "corrected_E": final_E[casebase_item, dim].item(),
        }
        for casebase_item, dim in changed.tolist()
    ]


def _save(output_path: Path, config: dict[str, Any], n: int, results: list[dict | None]) -> None:
    completed = [r for r in results if r is not None]
    num_flipped = sum(1 for r in completed if r.get("flipped"))
    payload = {
        "config": config,
        "summary": {
            "num_total": n,
            "num_completed": len(completed),
            "num_flipped": num_flipped,
            "flip_rate": num_flipped / len(completed) if completed else 0.0,
        },
        "samples": completed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="YAML file holding hyperparameters and paths (see "
        "tuning/contest/contest_all_irrelevance.yaml). Any other flag passed "
        "here overrides the corresponding value in it.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--qbaf", default=None)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit the run to the first N misclassified samples (default: all).",
    )
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument(
        "--max-edits",
        type=int,
        default=None,
        help="Stop once this many distinct entries of a sample's E have been "
        "touched (default: unbounded).",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory to write the run's JSON output to. Created if it "
        "doesn't exist. Defaults to today's outputs/<date>/ folder.",
    )
    parser.add_argument("--output-filename", default=None)
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Write partial results to the output file every N newly "
        "processed samples.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If the output file already exists with a matching config, skip "
        "indices it already has and continue from there instead of starting "
        "over.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    checkpoint = resolve_read_path(required(args.checkpoint, config, "checkpoint", args.config))
    qbaf = resolve_read_path(required(args.qbaf, config, "qbaf", args.config))
    num_samples = resolved(args.num_samples, config, "num_samples", None)
    k = resolved(args.k, config, "k", DEFAULT_K)
    threshold = resolved(args.threshold, config, "threshold", THRESHOLD)
    margin = resolved(args.margin, config, "margin", MARGIN)
    max_iters = resolved(args.max_iters, config, "max_iters", MAX_ITERS)
    max_edits = resolved(args.max_edits, config, "max_edits", None)
    device = resolved(args.device, config, "device", "cuda" if torch.cuda.is_available() else "cpu")
    log_dir_str = resolved(args.log_dir, config, "log_dir", str(today_output_dir()))
    output_filename = resolved(args.output_filename, config, "output_filename", "contest_all_irrelevance.json")
    save_every = resolved(args.save_every, config, "save_every", 25)

    run_config = {
        "checkpoint": checkpoint,
        "qbaf": qbaf,
        "num_samples": num_samples,
        "k": k,
        "threshold": threshold,
        "margin": margin,
        "max_iters": max_iters,
        "max_edits": max_edits,
        "device": device,
        "schema_version": 2,  # bump on result-schema changes so --resume detects a mismatch
    }

    log_dir = Path(log_dir_str)
    output_path = Path(resolve_write_path(str(log_dir / output_filename))) if log_dir_str else Path(resolve_write_path(output_filename))

    done: dict[int, dict] = {}
    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("config") == run_config:
            done = {s["index"]: s for s in prev["samples"]}
            print(f"Resuming: {len(done)} samples already done in {output_path}")
        else:
            print(
                f"WARNING: {output_path} exists but its config doesn't match "
                "this run's config -- ignoring it and starting fresh."
            )

    print(f"Loading model from {checkpoint} ...")
    model = load_model(checkpoint, device)

    print(f"Loading qbaf from {qbaf} ...")
    with open(qbaf, "r", encoding="utf-8") as f:
        qbaf_data = json.load(f)

    samples, true_classes = load_all_samples(qbaf_data, device, num_samples)
    n = samples.shape[0]
    print(
        f"Running new_case_contest independently over {n} misclassified samples "
        f"(device={device}, k={k}, threshold={threshold}, margin={margin}, "
        f"max_iters={max_iters}, max_edits={max_edits})"
    )

    results: list[dict | None] = [done.get(i) for i in range(n)]
    num_processed_this_run = 0
    num_flipped_so_far = sum(1 for r in results if r is not None and r.get("flipped"))
    num_completed_so_far = sum(1 for r in results if r is not None)

    pending = [i for i in range(n) if results[i] is None]
    progress = tqdm(pending, desc="Contesting irrelevance edges", unit="sample", initial=0, total=len(pending))
    if num_completed_so_far:
        progress.set_postfix(flip_rate=f"{num_flipped_so_far}/{num_completed_so_far}")

    for i in progress:
        sample = samples[i : i + 1]
        target_class = true_classes[i]
        try:
            result = new_case_contest(
                model, sample, target_class=target_class,
                k=k, threshold=threshold, margin=margin,
                max_iters=max_iters, max_edits=max_edits,
            )
            touched_edges = _touched_edge_triples(result.initial_E, result.final_E)
            results[i] = {
                "index": i,
                "true_class": target_class,
                "flipped": result.success,
                "iterations": result.iterations,
                "final_target_strength": result.final_target_strength,
                "final_rival_class": result.final_rival_class,
                "final_rival_strength": result.final_rival_strength,
                "max_E_delta": result.max_weight_delta,
                "num_edges_touched": len(touched_edges),
                "touched_edges": touched_edges,
            }
        except Exception as e:
            progress.write(f"  sample {i}: ERROR {e!r}")
            results[i] = {
                "index": i,
                "true_class": target_class,
                "flipped": False,
                "error": repr(e),
            }

        num_processed_this_run += 1
        num_completed_so_far += 1
        num_flipped_so_far += bool(results[i].get("flipped"))
        progress.set_postfix(flip_rate=f"{num_flipped_so_far}/{num_completed_so_far}")

        if num_processed_this_run % save_every == 0:
            _save(output_path, run_config, n, results)

    _save(output_path, run_config, n, results)
    print(f"\nDone. Saved to {output_path}")
    completed = [r for r in results if r is not None]
    num_flipped = sum(1 for r in completed if r.get("flipped"))
    print(f"{num_flipped}/{len(completed)} flipped ({num_flipped / max(1, len(completed)):.1%})")


if __name__ == "__main__":
    main()
