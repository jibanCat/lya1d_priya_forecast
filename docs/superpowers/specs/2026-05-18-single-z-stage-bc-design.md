# Single-z forecast pipeline — Stage B + Stage C design

**Date:** 2026-05-18
**Branch:** `single_z_forecast_clean`
**Status:** design approved, pending spec review

## 1. Background and motivation

The `single_z_forecast_clean` branch introduced a student-facing single-z
forecast pipeline (commit `799a63a`, "Stage A"): one YAML controls one
forecast, one CLI (`scripts/run_pipeline.py`) dispatches, three modes share
the config schema. Only **Stage A** landed:

- `gp_only` — Fisher on the GP emulator only (σ_GP). Implemented.
- `forecast_only` — student PySR Pareto CSVs → equations → Fisher. **Stub**
  (`pipeline.py` raises `NotImplementedError`).
- `refit_and_forecast` — refit single-z PySR per parameter, emit CSVs, then
  forecast. **Stub**.

This spec covers implementing the two stubbed modes and the student-facing
documentation/onboarding, so a student can follow the full single-z
procedure (regenerate training data → refit → forecast → aggregate)
cleanly, end to end.

Scope confirmed with the user:

- Both `forecast_only` (Stage B) and `refit_and_forecast` (Stage C).
- Stage C refit uses **default smart PySR kwargs** + Phase 1.5 Pareto-front
  selection. **No HPO.**
- Refit covers **all 13 single-z bins**, batched on SLURM.
- Rewrite `docs/ONBOARDING.md` and add a `notebooks/` folder with one
  tutorial per mode.

## 2. Architecture (Approach A — single-z unit + thin batch driver)

The **single-z unit** stays exactly as Stage A established it: one YAML =
one z-bin = one forecast. `PipelineConfig` stays strictly single-z (one
`redshift:`). Stages B and C are new `pipeline.py` functions on that unit.
Batching over the 13 z-bins and across-z aggregation live in **separate
driver scripts**, not inside the pipeline.

Rejected alternatives:

- **Multi-z-aware config** (`redshifts: [list]`): breaks the "single-z"
  abstraction Stage A and the student story are built on; forces a
  mode × single/multi matrix into `pipeline.py`.
- **Stage C as pure SLURM-script generator**: pushes SLURM mechanics onto
  the student, splits the refit story across a script and a manual step.

### 2.1 The factored refit unit

One pure function does the smallest piece of work — refit *one parameter
at one z-bin*:

```
refit_one_param_single_z(param, z, cfg) -> writes one Pareto CSV
```

Both consumers call it, so there is no duplicated refit code path:

- **Mode C, one z-bin** (`run_refit_and_forecast`): loops all 11 params
  in-process (smart kwargs, sequential), then runs the Stage-B forecast on
  the fresh CSVs. One command, ~30 min, notebook-friendly.
- **SLURM array task**: each task calls it for
  `(param = PARAMS[$SLURM_ARRAY_TASK_ID], z = $Z)`. The batch driver
  submits 13 such array jobs (one per z-bin).

### 2.2 File map

New / changed files:

| File | What |
|------|------|
| `src/priya_forecast/single_z/pipeline.py` | Implement `run_forecast_only`, `run_refit_and_forecast` (replace stubs) |
| `src/priya_forecast/single_z/refit.py` | New — `refit_one_param_single_z` + the 11-param loop |
| `src/priya_forecast/single_z/combine.py` | New — additive Taylor combine (default) + multiplicative/joint |
| `scripts/regen_1pvar.py` | New — regenerate per-param LF/HF 1pvar training data from the emulator |
| `scripts/run_batch.py` | New — fan one base YAML over the 13 z-bins |
| `scripts/aggregate_z.py` | New — collect 13 per-z results → across-z σ(z) view |
| `slurm/single_z_refit.slurm` | New — array job template (`--array=0-10`, parametrized by `Z`) |
| `src/priya_forecast/_vendored/data/pareto_baseline/` | New — vendored baseline Pareto CSVs (one Stage-C run, all 13 z) |
| `docs/ONBOARDING.md` | Rewrite — single-z pipeline guide |
| `notebooks/0{1,2,3}_*.ipynb` | New — per-mode tutorials |

Reused as-is:

- `plot_fisher_corner` — `src/priya_forecast/diagnostics/forecast_plots.py`
- `plot_equation_card` (loss–complexity) — `src/priya_forecast/diagnostics/equation_card.py`
- `pick_equation` — `src/priya_forecast/models/pysr_model.py`
- `SMART_REFIT_PYSR_KWARGS` — `src/priya_forecast/refit_1d_pysr.py`
- Stage-A Fisher / likelihood machinery — `single_z/pipeline.py`, `ksdata_likelihood.py`
- Equation porting — `scripts/port_pysr_equations.py`

## 3. Data regeneration — `scripts/regen_1pvar.py`

The original PySR training data (`InferenceLyaData/1pvar/{lf,hf}_{param}_
npoints50_datacorrFalse.hdf5`) was generated from Martin Fernandez's
simulations on a **different k-range**. The single-z forecast scores
against the kodiaq-squad emulator, so the refit must train on the
**kodiaq-squad k-range**.

`regen_1pvar.py` replaces those HDF5s:

- For each of the 11 PRIYA parameters: build the 1pvar design — the other
  10 parameters pinned at fiducial, this one swept over 50 points across
  its prior range.
- Evaluate the **emulator at `cfg.gp.basedir`** at LF (`r=0.4`) and HF
  (`r=0.8`), on the kodiaq-squad k-grid, for all 13 z-bins.
- Write `{lf,hf}_{param}_npoints50.hdf5` with the same
  `flux_vectors` / `kfkms` / `params` / `zout` schema as the originals,
  into a gitignored `data/single_z_1pvar/`.

**Verified:** `data/kodiaq_gp/` (the Stage-A `gp.basedir`) is a stripped
copy of `~/lya_emulator_full/kodiaq_2_2_4_6-48-48/` — identical
`emulator_params.json`, identical `trained_mf/` z-bin pickles, identical
`mf_emulator_flux_vectors` MD5. The full dir only adds diagnostics
(`loo_fps`, `seed_converge`, `kmax2.0_*`). So generating training data
from `cfg.gp.basedir` makes the PySR training data and the forecast
ground-truth GP literally the same object — self-consistent by
construction.

## 4. Stage C — `refit_and_forecast`

Per-z-bin flow (`run_refit_and_forecast(cfg)`, one z):

1. **Load training payload.** Read `data/single_z_1pvar/`, slice to the
   config's z-bin. Inputs `(θ_norm, k_norm, resolution)` min-max
   normalized from LF; target `flux_norm = (P_F − mean_k) / std_k`, with
   `(mean_k, std_k)` taken from the **multi-D fiducial** (see
   `student_pysr_contract` item 3 — the LF-emulator Sobol over the
   multi-D prior cube, not a 1D mean).
2. **Refit 11 params.** Loop `refit_one_param_single_z` with
   `SMART_REFIT_PYSR_KWARGS` (operators `+ − * / ^`, unary
   `exp/log/square`, `^` constrained `(-1, 0)`, `niter=50`, `maxsize=20`,
   `maxdepth=10`). No HPO. `use_anova_loss` stays a config knob, default
   `false` (`feedback_anova_loss_impact`: marginal in practice).
3. **Emit Pareto CSVs** at `<output_dir>/refit/z{z}/pareto_{param}.csv` —
   full Pareto fronts, never reduced.
4. **Forecast.** Hand off to the Stage-B path with
   `pareto_csvs.source: from_refit` pointed at step 3's directory.

SLURM path (used only by the batch driver): `slurm/single_z_refit.slurm`
is `slurm/refit_array.slurm` adapted — `--array=0-10` over the 11 params,
`Z` passed via `--export`; each task calls `refit_one_param_single_z`.
The batch driver submits 13 such jobs (one per z-bin); `regen_1pvar.py`
runs once up front (it is z-vectorized).

Equation selection from each Pareto front uses the config `pick` rule —
`best_loss` (default, the Phase 1.5 rule), `complexity_le:N`,
`accuracy_at:tol`, or `row:I` — already implemented in `pick_equation`.

## 5. Stage B — `forecast_only`

Per-z-bin flow (`run_forecast_only(cfg)`, one z):

1. **Load Pareto CSVs** for the z-bin, per `pareto_csvs.source`:
   - `bundled_baseline` → vendored `_vendored/data/pareto_baseline/z{z}/`
   - `per_parameter` → student-supplied paths in the YAML
   - `from_refit` → `<output_dir>/refit/z{z}/` (Stage-C handoff)
2. **Pick one equation per param** — `pick_equation(df, rule)`.
3. **Compile equations to callables** via `port_pysr_equations.py` →
   `eq_i(θ_i_norm, k_norm, r)`.
4. **Build the combined model** `P_F(θ, k)` in `combine.py`. Default is
   the additive 1st-order Taylor combine (`student_pysr_contract` item 5):

   ```
   P_norm(θ, k) = Σ_i [eq_i(θ_i_norm, k_norm, 0.8) − eq_i(0.5, k_norm, 0.8)]
                + (1/n) Σ_i eq_i(0.5, k_norm, 0.8)
   P_F(θ, k)    = P_norm(θ, k) · std_k_global + mean_k_global
   ```

   `multiplicative` and `joint` remain selectable via the YAML
   `combine:` field. `0.5` is the per-param fid_norm approximation the
   student hard-codes; `0.8` is HF resolution.
5. **Three Fisher forecasts** on the kodiaq-squad `(k)` grid for that z:
   - `σ_GP` — full emulator (reuses Stage-A `run_gp_only` machinery).
   - `σ_perfect_1D` — the emulator's exact 1D responses fed through the
     same additive combine; isolates combine-structure loss from PySR-fit
     loss.
   - `σ_PySR` — the fitted-equation combined model.
   - plus ratios `σ_PySR/σ_GP`, `σ_perfect_1D/σ_GP`.
6. **Deliverables** written to `<output_dir>/z{z}/`:
   - forecast table, scorecard
   - Fisher corner plot (`plot_fisher_corner`, overlaying GP /
     perfect_1D / PySR)
   - the picked per-D equations
   - per-D resolution correction `eq_i(θ,k,0.8) − eq_i(θ,k,0.4)`
     (`student_pysr_contract` item 6 / `forecast_deliverables`)
   - loss–complexity diagnostic plot per param (`plot_equation_card`),
     picked row marked.

**To verify during implementation, not assumed here:** the exact
definition of `σ_perfect_1D` and the de-normalization details will be
cross-checked against `scripts/train_and_forecast.py` and
`scripts/forecast_original_design.py` before the combine math is locked,
so Stage B reproduces the established multi-z numbers.

### 5.1 Pareto CSV tracking

The full Pareto fronts are first-class artifacts (the user plots
loss–complexity diagnostics from them afterward):

- Full fronts preserved verbatim, never reduced to the picked row.
- For `bundled_baseline` / `per_parameter` inputs, Stage B **copies** the
  source CSVs into `<output_dir>/z{z}/pareto/` so every forecast run is a
  self-contained record of exactly which fronts produced it.
- The picked row per param is recorded in `<output_dir>/z{z}/manifest.json`
  (rule used, chosen `Complexity` and `Loss`).

## 6. Batch driver and across-z aggregator

### 6.1 `scripts/run_batch.py`

Fans one base YAML over the 13 z-bins by overriding `redshift` and
`output_dir` per bin (in-memory derived configs — no 13 YAML files).

- **`gp_only` / `forecast_only`** — no SLURM. Loops the pipeline over the
  13 bins in-process, then calls the aggregator.
- **`refit_and_forecast`** — SLURM is asynchronous, so two explicit
  phases (no live-queue coupling):
  - `--phase submit` → runs `regen_1pvar.py` once if
    `data/single_z_1pvar/` is missing, submits the 13 array jobs
    (`single_z_refit.slurm`, `--array=0-10`, one per z-bin), prints job
    IDs, writes `batch_manifest.json`.
  - `--phase collect` → checks all `refit/z{z}/pareto_*.csv` are present,
    runs the Stage-B forecast per z-bin, then the aggregator. Fails loud,
    listing every missing `(z, param)` cell.
  - Optional `--chain` submits `collect` as a `sbatch
    --dependency=afterok` job for the one-command case — off by default.

### 6.2 `scripts/aggregate_z.py`

Pure post-processing — reads files, runs no model. Reads the 13 per-z
`<output_dir>/z{z}/` results, writes `<output_dir>/aggregate/`:

- σ(z) trend plot per parameter — `σ_GP`, `σ_perfect_1D`, `σ_PySR`
  overlaid vs z.
- Combined σ-table (params × z-bins) and ratio table
  (`σ_PySR/σ_GP` vs z).
- A roll-up scorecard linking each per-z scorecard.

## 7. Documentation and notebooks

### 7.1 `docs/ONBOARDING.md` rewrite

Keep the existing Fisher-math derivation (the doc's strength) but reframe
around the single-z pipeline: the three modes, the
`θ → P_F → Fisher → σ` chain, the additive-Taylor combine, and the
`σ_GP / σ_perfect_1D / σ_PySR` ladder. New reading map points at
`single_z/{config,pipeline,refit,combine}.py`. The student procedure
(regen → refit → forecast → aggregate) is the spine of the document.

### 7.2 `notebooks/` — one tutorial per mode

Each runs end-to-end on a single z-bin so it completes in minutes:

- `01_gp_only.ipynb` — run `gp_only`; read the forecast table, scorecard,
  corner plot; Fisher sanity check.
- `02_forecast_only.ipynb` — load `bundled_baseline` CSVs, pick equations,
  combine, forecast; show the loss–complexity plot and the GP-vs-PySR
  corner.
- `03_refit_and_forecast.ipynb` — `regen_1pvar` + one-z in-process refit +
  forecast; then point at `run_batch.py` for all 13 z-bins on SLURM.

## 8. Error handling

All failures are loud; there is no silent fallback.

- Missing `data/single_z_1pvar/` → Stage C raises "run regen_1pvar
  first".
- Missing LF/HF data for a z-bin → error; never fall back to
  1D normalization.
- Missing Pareto CSV, or an equation that fails to parse → error naming
  the `(z, param)`.
- `run_batch.py --phase collect` → lists every missing `(z, param)` cell
  before aborting.

## 9. Testing

Mirrors Stage A's `tests/test_single_z_pipeline.py` conventions (unit +
hypothesis tests, slow end-to-end smokes gated behind env vars). No new
dependencies (per `build_conventions`).

- `combine.py` — unit tests with known-input/known-output; assert the
  combine recovers the anchor at fiducial θ.
- `refit_one_param_single_z` — fast test with PySR mocked; a real run
  gated behind a `RUN_SLOW_*` env var.
- `aggregate_z.py` — tested on fixture per-z directories (pure
  post-processing).
- `run_batch.py` — test the 13-config fan-out and the SLURM-submission
  command string *without* submitting (subprocess mocked).
- `regen_1pvar.py` — test the 1pvar design construction (shapes, param
  sweep ranges) with the emulator call mocked; a real run gated.
- Per-mode end-to-end smokes gated behind env vars, like Stage A's
  `RUN_SLOW_GP_ONLY`.

## 10. Implementation staging

The spec designs all of B + C + docs coherently; implementation proceeds
in this order so the student-facing default lands first and the docs can
be written against working code:

1. `regen_1pvar.py` + `combine.py` (+ tests) — shared foundations.
2. Stage B `run_forecast_only` (+ tests). Generate and vendor the
   `bundled_baseline` Pareto CSV set once Stage C exists; until then
   Stage B is exercised via `per_parameter` fixtures.
3. Stage C `refit.py` + `run_refit_and_forecast` + `slurm/single_z_refit.slurm`
   (+ tests). Run it once across all 13 z-bins to produce the vendored
   `bundled_baseline` set.
4. `run_batch.py` + `aggregate_z.py` (+ tests).
5. `docs/ONBOARDING.md` rewrite + the three notebooks.

## 11. Out of scope

- Multi-z / cross-coupling PySR (the `fine-tune-pysr` branch work).
- HPO for the refit (explicitly excluded — smart kwargs only).
- Phase 3 Ap σ-ratio remediation (`docs/AP_REMEDIATION_PLAN.md`) —
  paused, unrelated.
