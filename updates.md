# Updates

Everything committed by Greta Reitenbach since forking the repo from Adam Gould's
`Deep-Arguing` (last upstream commit: `8eedb57`, "Final experiments completed",
2026-05-07). Organized by date; commit hashes refer to `git log`.

## 2026-07-02

- **Fixed bugs causing tests to fail** (`69328d1`)
  - `GradualAACBR.forward`: `return_all_strengths` path squeezes the trailing
    dimension when `dimensions == 1`, matching the shape the non-`return_all_strengths`
    path already returned.
  - `CurriculumTrainer`: made `validation_log_strategy` optional (defaults to a new
    `CurriculumValidationLog()`), fixed several methods that were missing an
    explicit `self` parameter, extracted the epochs-vs-curriculum-length check
    into `_validate_epochs`, added a `_reset_optimizer_state` helper, and made
    the calls to `criterion`/`regulariser` tolerant of both `(model, preds, y)`
    and `(preds, y)` signatures via a `try`/`except TypeError` fallback.
  - `train/strategies.py`: same criterion/regulariser signature fallback applied
    in `CurriculumValidationLog`.

- **Edited evals to be compatible with older numpy** (`f326272`)
  - `evals.py`: replaced `np.concat` (newer numpy only) with `np.concatenate`,
    and swapped the `assert type(x) == float` checks for explicit `float(x)`
    casts so the metrics don't depend on the exact numpy scalar type returned.

- **Added misclassification download to CLI run** (`1236cae`)
  - `cli/run.py`: after test-set evaluation, isolate misclassified test samples
    and export them (with their QBAF tensors) to `graphs/misclassified_qbaf.json`
    via `model.export_to_json`. (Later gated behind a flag — see 2026-07-06.)

## 2026-07-03

- **Fixed visualizer cytoscape extension** (`802c4a3`)
  - `visualizer/index.html`: one-line fix to the Cytoscape extension setup.

- **Merge branch 'main' into counterfactual** (`60cea82`)
  - Brought the `counterfactual` branch up to date with the fixes above.

## 2026-07-06

- **Gate misclassified QBAF export behind `--misclassified_log` flag** (`3b595aa`)
  - `cli/parse_command_line.py` / `cli/run.py`: the misclassified-sample export
    added on 07-02 was running unconditionally whenever `--run_test` was set;
    added `--misclassified_log`/`-ml` so it's opt-in instead.

- **Debugging** (`26017f1`)
  - Vendored `torchviz`'s `make_dot` into `src/deeparguing/evals/compute_graph.py`,
    since `torchviz` is unmaintained and fails to import on Python >= 3.12
    (depends on the removed `distutils` module).
  - `requirements.txt`: dropped `torchviz`, pinned previously-implicit
    dependencies (`torchvision`, `graphviz`, `matplotlib`, `optuna`, `PyYAML`,
    `wandb`).
  - `evals.py` / `parse_yaml.py`: small follow-on fixes for the above.

- **Created tests for evals.py** (`894e126`)
  - Added `tests/evals_test.py` (146 lines) covering `evaluate_model`.
  - `evals.py`: renamed a variable (`predictions` -> `final_strengths`/
    `batch_predictions`) for clarity while batching predictions.
  - `compute_graph.py`: added a `pyright` ignore-comment for the vendored,
    untyped `torchviz` code.

## 2026-07-07

- **Created grae module** (`ef43211`)
  - New `src/deeparguing/counterfactuals/grae.py` module: implements
    Gradient-based Relation Attribution Explanations (G-RAEs, Definition 13
    of the Contestability paper). `compute_grae` detaches `model.A` and
    `model.new_cases_attacks_adjacency` into fresh leaf tensors, replays the
    aggregation -> influence -> `gradual_semantics` computation on those
    leaves, and reads `.grad` off both after a single `.backward()` (with an
    optional per-sample mode for the casebase-edge gradient).
    `finite_difference_grae` provides a perturbation-based cross-check of
    the analytic gradients (Algorithm 1 of the paper).
  - CLI wiring: added `--grae_log`/`-gl` to `parse_command_line.py`; `run.py`
    now computes G-RAEs for the exported misclassified samples (when
    `--misclassified_log` is also set) and saves them to
    `graphs/misclassified_grae.pt`.
  - Added `tests/grae_test.py` (25 tests) covering the new module:
    - Cross-checks `compute_grae`'s analytic gradients against
      `finite_difference_grae` (Algorithm 1) across new cases/iteration
      counts, checks the finite-difference error shrinks with `epsilon`, and
      checks the batched/aggregate code path sums the per-sample Algorithm 1
      estimates correctly.
    - Property-based tests for Propositions 4-6 (Direct/Indirect Influence,
      Irrelevance) on a hand-built EW-QBAF that bypasses `fit()` by setting
      `model.A`/`X_train`/`default_indexes` directly, confirming G-RAEs have
      the expected sign for direct, indirect (odd/even downstream-attack
      parity), and independent edges.
    - Input-validation tests (unfitted model, `target_indices` length
      mismatch, batch size > 1 for the finite-difference path) and
      result-shape/detachment checks.

- **Updated visualizer for edge perturbation** (`6b7f8ea`)
  - `gradual_aacbr.py`: `export_to_json` gained optional `grae_casebase_edges`/
    `grae_new_case_edges`/`grae_target_indices` params, serialized under a
    `"grae"` key, so a G-RAE result can ride along in the same JSON the
    visualizer already reads instead of a separate file.
  - `cli/run.py`: reordered the `--grae_log` block to run before the JSON
    export (it previously only wrote `misclassified_grae.pt`), and now passes
    the computed G-RAEs into `export_to_json` too. The `.pt` export is
    unchanged/still written, since it's more compact for large casebases and
    carries `selected_indices`, which the JSON export doesn't.
  - `visualizer/app.js` / `index.html`: added a "G-RAE Perturbation" sidebar
    panel. Clicking an edge (while a new case is selected) selects it for
    perturbation; a Δ slider live-computes `predicted = original_strength +
    gradient * Δ` (a linear/first-order approximation) and pulses the
    differentiated default node's opacity/border accordingly.
  - Extended the panel to handle multi-target exports (`target_indices` as a
    list-of-lists instead of one index per sample, with a matching extra
    axis on `casebase_edges`/`new_case_edges`): renders a live "competition"
    table of every candidate class's predicted strength, bolds the current
    winner, and computes the exact crossing Δ (closed-form, since predicted
    strength is linear in Δ) at which another class would overtake it.
    `run.py`'s real export still only produces the single-target shape;
    multi-target is currently only exercised by hand-built demo fixtures.
  - Fixed a latent bug in the upload handler: the actual precompute/render
    work ran inside a `setTimeout` nested outside the surrounding
    `try`/`catch`, so any exception there (regardless of cause) was silently
    swallowed — no alert, and `hideLoading()` never ran, leaving the loading
    overlay stuck indefinitely. Wrapped that block (and the `initCytoscape`
    step) in their own `try`/`catch` so failures now surface an alert and
    clear the overlay.

## 2026-07-09

- **Added summary markdown for CLI** (`d4b0028`)
  - `cli/run.py`: new `write_markdown_summary` helper renders the CLI's
    per-run summary lines (validation/train/test results, plus the new
    GRAE breakdown below) to `graphs/summary.md` -- `--- X ---` lines become
    `## X` headings, everything else becomes a bullet. Hyperparameter-tuning
    runs append a `--- BEST TRIAL ---` block after the final trial's summary.
  - Added a G-RAE magnitude breakdown by edge type: buckets each sample's
    `compute_grae` output by the sign of the underlying adjacency entry
    (`model.A` for casebase edges, `model.new_cases_attacks_adjacency` for
    new-case edges) into attack (negative) vs. support (positive), and
    averages `|gradient|` within each bucket via a new `_mean_abs_grae`
    helper. Logged per-seed and averaged into the summary's new
    `--- GRAE RESULTS ---` block. Since `new_cases_attacks_adjacency` is
    always `<= 0` by construction (`RegularIrrelevance` returns a value in
    `[0, 1]`, then negated -- new cases can only attack, never support), the
    new-case support bucket is expected to come out `nan` (empty mask, no
    positive entries to average).
