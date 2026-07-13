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

## 2026-07-10

- **Created contest.py and test suite** (`a8cd256`)
  - New `src/deeparguing/counterfactuals/contest.py` module: implements the
    Week 3 heuristic contestability algorithm sketched in `grae.py`'s module
    docstring. Each outer iteration computes G-RAEs for the current sample
    against `target_class` (via `grae.compute_grae`), picks the top-`k`
    entries of `model.A` by `|G-RAE|` (`select_top_k`), then runs a
    backtracking line search (`backtracking_line_search`) that shrinks the
    step size from `ALPHA_MAX` until the trial perturbation crosses
    `threshold + margin` (or backtracks are exhausted, in which case the
    smallest-alpha trial is taken as a fallback partial step). Accepted
    steps are committed directly onto `model.A` -- the casebase-internal
    adjacency, so a successful contest has a persistent, global effect on
    the model, not just the queried sample. Returns a `ContestResult`
    (`success`, `iterations`, `max_weight_delta`, `edge_trace`,
    `final_strength`).
  - The file started as pseudocode (`sample.edge_weights`,
    `model.semantics_forward`, `NotImplementedError` stubs) that didn't
    match this repo's actual API; rewired it to the real one -- gradients
    come from the already-tested `grae.compute_grae`, and trial strengths
    come from real forward passes (`model(sample)`) with `model.A`
    temporarily swapped for a trial tensor. Also fixed a latent bug in the
    original sketch where hitting `max_iters` always reported the constant
    `MAX_ITERS` as the iteration count rather than how many iterations
    actually ran.
  - Exported `ContestResult`/`contest` from `counterfactuals/__init__.py`,
    mirroring how `grae.py`'s API is already exposed there.
  - Added `tests/contest_test.py` (10 tests) reusing the same synthetic
    EW-QBAF fixture as `tests/grae_test.py`: unit tests for `select_top_k`
    and `backtracking_line_search` (finds a crossing alpha, doesn't mutate
    `model.A` itself, returns `None` when `max_backtracks=0`), and
    end-to-end tests for `contest` (commits the accepted step to `model.A`,
    touches only the traced edge indices, reports `success=False` with
    `iterations == max_iters` when the threshold is unreachable in time,
    and raises on an unfitted model / batch size > 1).

## 2026-07-13

- **Changed threshold criteria for contestation** (`960e8ba`)
  - `contest.py`: the original stopping rule (`target_strength >= threshold +
    margin`, `THRESHOLD = 0.5`) assumed a single-argument acceptance boundary,
    but `GradualAACBR.forward`'s per-class strengths are neither normalized
    nor mutually exclusive (no softmax -- see `forward`'s `strengths @ W`
    combination), so crossing 0.5 says nothing about whether `target_class`
    actually wins the model's `argmax` prediction (the criterion `evals.py`/
    `cli/run.py` actually use to call a sample "misclassified"). Measured
    against the real `outputs/misclassified_qbaf.json` checkpoint (100
    CIFAR-10 samples): 5% already had `target_class` strength `>= 0.5` while
    still losing the `argmax`, and of the 95 that didn't, 87% still had the
    real rival class beating `target_class` by ~1.5 in strength even after
    hypothetically crossing `0.5 + margin` -- the fixed threshold was
    effectively decoupled from the thing the algorithm is supposed to fix.
  - Replaced it with an argmax-relative check: `target_class`'s strength
    must beat the best *other* class's strength (`_target_and_rival`) by
    `margin`. Every forward pass already computes every class's strength at
    once, so tracking the rival costs no extra passes. If `target_class` is
    the model's only default argument (no competing class -- the
    single-topic-argument setting from the original Contestability paper,
    e.g. `tests/contest_test.py`'s synthetic fixture), there is no real
    rival: `_target_and_rival` falls back to a fixed virtual competitor at
    `threshold`, recovering the original absolute criterion for that case.
  - `EdgeTraceStep`/`ContestResult` gained `*_rival_class`/`*_rival_strength`
    fields (renamed `old_strength`/`new_strength`/`final_strength` to
    `*_target_strength` for clarity) so the trace shows whether a step's
    rival changed identity, not just whether the target moved.
  - `run_contest.py`: `--threshold` documented as the single-topic-argument
    fallback only; logging now reports the rival class/strength and the
    achieved vs. required margin per step.

- **Changed to bracket-and-bisect search** (`fe0e607`)
  - `contest.py`: plain backtracking (shrink `alpha` from `alpha_max` until
    a trial crosses the margin, return the first one that does) optimizes
    for "cheap," not "minimal" -- confirmed live on sample 80 of the
    checkpoint above, where `alpha_max=1.0` crossed on the very first trial,
    overshooting the required margin by ~2.2 and moving edge weights by
    `>3` in one step. Renamed `backtracking_line_search` to
    `bisection_line_search`: phase 1 (bracket) is unchanged geometric
    backtracking, used only to find *a* crossing alpha; phase 2 (bisect)
    then binary-searches inside `[last-failing-alpha, first-crossing-alpha]`
    for up to `MAX_BISECTIONS` trials (or until the bracket narrows below
    `BISECT_TOL`) to converge toward the minimal crossing step instead of
    accepting the first one found. Re-run on sample 80: `max_weight_delta`
    dropped from `3.292147` to `0.061085` and the margin overshoot from
    `~2.2` to `0.0012`, for the same `success=True` outcome.
  - Each bisection trial is one more `torch.no_grad()` forward pass (no
    backward pass), so the added cost is linear in `MAX_BISECTIONS` and
    skipped entirely for dead-gradient samples (phase 1 never finds a
    crossing, so phase 2 never runs -- confirmed on sample 91, a live
    example of the kind of sample `sweep_dead_gradients.py` flags: `alpha`
    shrinks through all `MAX_BACKTRACKS` trials with an all-zero effective
    direction, `max_weight_delta=0.0` for all 50 `max_iters`, since `contest`
    has no early-exit for a dead direction and just repeats the same no-op
    step until `max_iters` is exhausted).
  - `tests/contest_test.py`: renamed the `backtracking_line_search` tests to
    `bisection_line_search`, updated for the new 5-tuple return shape and
    `final_target_strength`/`final_rival_*` fields, and added a test
    confirming bisection refines below `alpha_max` when it overshoots on the
    first trial.

- **Changed bisection thresholds** (`fad4ef3`)
  - `contest.py`: re-running sample 47 (initial rival gap only `0.0167`)
    showed phase 1 crossing at `alpha_max=1.0` and then *every* one of the
    10 phase-2 bisection trials also crossing -- `alpha_hi` halved all the
    way down to `0.5^10` without a single failing trial to pull `alpha_lo`
    up, so the loop exhausted its budget (not `BISECT_TOL`) with the margin
    still overshot by 13.5x (`0.1351` against a required `0.01`). Bumped
    `MAX_BISECTIONS` 10 -> 30 and tightened `BISECT_TOL` 1e-4 -> 1e-6; each
    extra step is one cheap forward pass and halves the bracket, so the
    added resolution is exponential (`~1e-9` relative to `alpha_max=1.0`)
    for ~20 extra passes, paid only by samples that reach phase 2 (dead-
    gradient samples like 91 above never do). Re-run on sample 47: margin
    overshoot dropped from `0.1351` to `0.0001` and `max_weight_delta` from
    `0.006209` to `0.001085`.

- **Added dead-gradient bottleneck escape** (uncommitted)
  - New `src/deeparguing/counterfactuals/bottleneck.py`: handles the case
    where `_casebase_grae` comes back uniformly ~0 for a sample, instead of
    letting `contest()` spin uselessly on a zero direction until `max_iters`
    (as sample 91 did before). Reused
    `GradualAACBR.forward(..., return_all_strengths=True)`, which already
    exposes every casebase node's own converged strength per sample -- no
    changes to `gradual_aacbr.py` needed. `find_bottleneck` greedily walks
    the influence graph from the sample towards `target_class`'s default
    argument (following the single strongest edge per hop) and returns the
    first node pinned at exactly 0. Confirmed by working through the
    aggregation math (and a synthetic test that initially assumed
    otherwise, then failed): because `aggregation_func` sums densely over
    *every* source node each iteration (not just structurally-existing
    edges), a *uniformly* zero `_casebase_grae` requires `target_class`'s
    *own* final strength to be pinned at 0 too, not merely some upstream
    node -- otherwise at least the direct one-hop edges into the target
    would show nonzero gradient from whichever source nodes have nonzero
    strength. `select_bottleneck_edges` ranks the bottleneck's incoming
    edges by `|source node's own strength|` -- exactly the local partial
    derivative of its aggregation, read directly off the forward pass
    already computed (the aggregation step is linear, so no backward pass
    needed). `expanding_step_search` is `bisection_line_search`'s mirror
    image: grows `alpha` geometrically from a small `alpha_init` instead of
    shrinking from `alpha_max`, stopping once the bottleneck node's own
    strength is no longer pinned at 0 (a different, weaker condition than
    crossing the classification margin, hence a separate function).
    `find_and_escape_bottleneck` orchestrates the three and falls back to
    `grae_vector` for direction only if the local leverage is itself all
    zero (every incoming source also saturated); if that's also zero,
    returns `None` -- structurally hopeless.
  - `contest.py`: added `LIVE_GRAD_THRESHOLD` (`1e-9` -- **not** re-validated
    here against the real checkpoint's max|grad| distribution per the day's
    compute constraints; confirm the live/dead split is still cleanly
    bimodal before trusting this value on a new checkpoint/dataset). Each
    outer iteration now checks
    `grae_vector`'s magnitude and routes to `find_and_escape_bottleneck`
    instead of `select_top_k`/`bisection_line_search` when dead; both paths
    feed one shared trace-recording/commit block (unified into a common
    6-tuple: `edge_indices, alpha, new_A, new_target_strength,
    new_rival_class, new_rival_strength`), so a dead-then-escaped sample
    falls through to ordinary gradient steps on the next iteration once
    `_casebase_grae` is recomputed and live again. `contest()` imports
    `find_and_escape_bottleneck` lazily (inside the function body) since
    `bottleneck.py` imports several private helpers back from `contest.py`
    -- an eager top-level import would be circular.
  - Deleted `sweep_dead_gradients.py`: its original purpose was to pre-filter
    "dead" samples out before running `contest()`, so the expensive search
    wasn't wasted on ones it couldn't move -- but `contest()` now handles
    that itself (escaping when possible, failing after 1 iteration instead
    of `max_iters` when not), so a sample being "dead" no longer implies it
    should be skipped, and the script's own docstring advice to that effect
    was now actively wrong. No remaining callers depend on it (`run_contest.py`
    never imported it; the dependency ran the other way, the script reusing
    `run_contest.py`'s `load_model`/`load_sample`).
  - Added `tests/bottleneck_test.py` (6 tests) on a hand-built 4-node ReLU
    EW-QBAF (bypassing `fit()`, same approach as `grae_test.py`'s
    property-based fixtures): `find_bottleneck` locates the saturated node,
    `select_bottleneck_edges` prefers the higher-leverage source,
    `find_and_escape_bottleneck` picks that edge and returns `None` when
    every attacker is itself pinned at 0 (structurally hopeless), and two
    `contest()` end-to-end cases -- starts dead, escapes, then finishes via
    ordinary bisection steps; and the structurally-hopeless case reports
    `success=False` after exactly 1 iteration rather than looping to
    `max_iters`.
  - `contest.py`: re-running sample 47 (initial rival gap only `0.0167`)
    showed phase 1 crossing at `alpha_max=1.0` and then *every* one of the
    10 phase-2 bisection trials also crossing -- `alpha_hi` halved all the
    way down to `0.5^10` without a single failing trial to pull `alpha_lo`
    up, so the loop exhausted its budget (not `BISECT_TOL`) with the margin
    still overshot by 13.5x (`0.1351` against a required `0.01`). Bumped
    `MAX_BISECTIONS` 10 -> 30 and tightened `BISECT_TOL` 1e-4 -> 1e-6; each
    extra step is one cheap forward pass and halves the bracket, so the
    added resolution is exponential (`~1e-9` relative to `alpha_max=1.0`)
    for ~20 extra passes, paid only by samples that reach phase 2 (dead-
    gradient samples like 91 above never do). Re-run on sample 47: margin
    overshoot dropped from `0.1351` to `0.0001` and `max_weight_delta` from
    `0.006209` to `0.001085`.
