# priya-forecast

PRIYA Lyman-α P1D Fisher / MCMC forecast, with the forward model
swappable between the GP emulator and PySR-derived analytic equations.
Use this repo to **take your trained PySR equations and score them**:
how close is `σ_PySR` to `σ_GP`, parameter by parameter?

**Read first**: [`docs/ONBOARDING.md`](docs/ONBOARDING.md) — math-first
walkthrough up to Phase 1.5 (Fisher, additive main-effect combine, the
ANOVA loss, the at-fid normalization). Then this README for the
student-facing reward loop.

---

## Quick start: score your own PySR CSVs at a chosen z bin

You trained PySR per-parameter at some redshift `z` and ended up with
one `hall_of_fame_<param>_z<z>.csv` Pareto front per parameter. Wire
them into the forecast in three steps.

### 1. Install + PYTHONPATH

```bash
pip install -e ".[forecast,pysr,gp,dev]"

# Upstream lyaemu (sbird/lya_emulator) supplies the GP. On Greatlakes:
export PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:$PWD/src
# On a fresh machine: clone https://github.com/sbird/lya_emulator first
# and point PYTHONPATH at your clone.
```

### 2. Write a YAML pointing at your CSVs

Copy [`configs/eqns/pysr_v1.yaml`](configs/eqns/pysr_v1.yaml) as a
starting template and edit the per-parameter entries. The schema:

```yaml
name: my_pysr_run                   # any label you like
model: pysr
redshift: 3.6                       # MUST match your PySR training z
combine: multiplicative             # multiplicative | additive | joint
fiducial_p1d: data/priya_fiducial/p1d_z3.6.npz   # cached GP P_F at fid

parameters:
  ns:
    pareto_csv: /path/to/hall_of_fame_ns_z3.6.csv
    pick: best_loss                 # see "pick rules" below
    fiducial: 0.97                  # physical fid value of this param
    variables: [ns, k]              # input columns, in the order PySR saw them
  Ap:
    pareto_csv: /path/to/hall_of_fame_Ap_z3.6.csv
    pick: complexity_le:15
    fiducial: 1.46
  # ... one entry per parameter you want in the forecast
```

**Fiducial values** must match `fiducial_vector()` in
`src/priya_forecast/parameters.py` (e.g. `ns: 0.983`, `Ap: 1.46`,
`hub: 0.6726`, `omegamh2: 0.1430`, etc.). Use `0.983` for `ns`, not
`0.97` — the production fid is what the GP was trained at, and any
mismatch poisons the additive combine. `configs/eqns/pysr_v1.yaml`
already lists the correct values; copy from there.

**Fiducial p1d cache**: `fiducial_p1d:` points at a `.npz` with arrays
`(k, p1d)` cached at fid. The script **auto-creates** this file if
missing — just give it a path under `data/priya_fiducial/` or
`results/_cache/` and it'll fill in `gp.predict(fid, k, z)` on first
run.

**`pick:` rules** — which Pareto-front row to use:

| Rule              | Picks |
|---|---|
| `best_loss`       | row with the lowest `Loss` (PySR's built-in best). |
| `complexity_le:N` | among rows with `Complexity ≤ N`, the lowest `Loss`. Use this when you've decided on an interpretability budget. |
| `accuracy_at:tol` | among rows with `Loss ≤ tol`, the smallest `Complexity`. Right answer for "smallest equation that's still accurate." |
| `row:I`           | the I-th row by 0-indexed position (escape hatch). |

**`combine:` rules** — how the per-parameter equations stitch into a
multi-parameter prediction (the math is in `docs/ONBOARDING.md § 2`):

- `multiplicative`: `P̂(θ, k) = P_fid(k) · ∏ᵢ [eq_i(θ_i, k) / eq_i(fid_i, k)]`
- `additive`:       `P̂(θ, k) = P_fid(k) + Σᵢ [eq_i(θ_i, k) − eq_i(fid_i, k)]`
- `joint`: a single equation in `(θ_1, ..., θ_n, k)`; set
  `joint_expression: ...` and omit per-parameter `pareto_csv`.

**Normalization**: not exposed in the YAML — `scripts/train_and_forecast.py`
hard-codes `{"mode": "auto", "fix": {"r": 0.8}}`, which derives the
per-(z,k) `(mean_k, std_k)` by sampling the GP at fixed-fiducial-rest
and pins the resolution feature to `r = 0.8` (HF). This is the right
default for the student loop. If you need a different mode (`identity`
or load from `mean_flux_low.txt` files), edit `build_from_yaml()` in
the script directly.

### What this workflow can and cannot score

This script is built for the **single-z recipe** PySR contract
(`pysr_mf_given.py`). Three cases:

| Your PySR was trained on...           | Works in this script? |
|---|---|
| `(θ_i, k)` — 2 features, single z     | ✅ Yes. Set `variables: [<param>, k]`. |
| `(θ_i, k, r)` — 3 features (HF/LF resolution), single z | ✅ Yes. Set `variables: [<param>, k, r]`. The script pins `r = 0.8` at scoring time. |
| `(θ_i, k, resolution, z_norm)` — 4 features, multi-z (production) | ❌ **No** — see below. |

**Why multi-z 4-feature CSVs don't load**: `compile_equation`
(`src/priya_forecast/models/pysr_model.py:265-272`) requires every
non-`{param, k}` variable to be assigned a constant in `fix:`. The
script's `fix:` is hard-coded to `{"r": 0.8}` and not exposed in the
YAML schema, so a `z_norm` feature has no place to land. (Pinning
`z_norm` to a constant would also defeat the multi-z property of the
equation, so this isn't a small fix.) **To score the multi-z 4-feature
fits described in `docs/ONBOARDING.md`**, use the production path:
`scripts/refit_all_11_params.py` (training) →
`scripts/multi_z_aggregate.py` (multi-z Fisher) →
`scripts/holdout_multid.py` (validation), all of which understand the
4-feature MultiZ schema natively.

### 3. Run the forecast

```bash
python scripts/train_and_forecast.py \
    --params ns Ap hub omegamh2 \
    --equations configs/eqns/my_pysr_run.yaml \
    --z 3.6 \
    --output results/my_pysr_run_z3.6/
```

`--params` is the subset you want in the forecast (subset of the 11).
`--z` must match the YAML's `redshift:` and your PySR training z.
The script loads eBOSS DR14 P1D data + covariance for that z (vendored
in `src/priya_forecast/_vendored/data/`) and computes Fisher for three
models:

1. **`GP_reference`** — the GP emulator itself (upper bound: best
   anyone could do).
2. **`perfect_1D_slices`** — the GP under multiplicative combine,
   evaluated parameter-by-parameter (upper bound: best a 1D-product
   PySR set could match exactly).
3. **`<your YAML name>`** — your equations.

### 4. Read the output

In `results/my_pysr_run_z3.6/`:

- **`scorecard.md`** — the headline. Shows per-parameter
  `σ_student / σ_perfect_1D` (how close you are to the 1D-product
  ceiling) and `σ_perfect_1D / σ_GP` (how much the 1D-factorization
  assumption costs). Plus geomean rollups and an "off-fid MSE ratio"
  that catches equations that match at fid but extrapolate badly.
- **`summary.md`** — per-equation-set diagnostics (rel-err, complexity,
  Pareto pick, etc.).
- **PNGs in the same dir** — `forecast_corner.png`, `forecast_sigma.png`,
  `eq_card_*.png`, `residual_*.png`. These land directly in the output
  directory (no `figures/` subdir).

Targets:
- `σ_student / σ_perfect_1D < 1.5` (geomean) → 1D PySR is converged.
- `off-fid MSE ratio < 2` → equations track the GP off-fid too.
- If both met: 1D-factorization is the bottleneck — graduate to
  multi-D PySR (`scripts/run_multid_pysr.py`).

---

## What else lives here

- `scripts/train_and_forecast.py` — the student-facing reward loop above.
  Single-z, eBOSS DR14 covariance.
- `scripts/refit_all_11_params.py` + `scripts/refit_one_param.py` — the
  **production** per-1D PySR refit pipeline (multi-z, KSData covariance).
  This is what generates Phase 1.5 / Phase 2 results in `results/`.
  Different beast from the student loop above; see `docs/PAPER_NOTES.md`
  for the production design.
- `scripts/holdout_multid.py` — multi-D Sobol hold-out validation.
- `scripts/closure_at_simdat_target.py` — off-fid Fisher closure at
  `θ_target_simdat` (σ_PySR vs σ_MCMC).
- `src/priya_forecast/` — the library (Fisher, likelihood, models,
  per-1D + pair refits, normalization, Pareto filters). See
  `docs/ONBOARDING.md § 7` for a math-ordered reading map.
- `tests/` — unit + property-based tests
  (`PYTHONPATH=src pytest tests/ -q`).
- `configs/eqns/pysr_v1.yaml` — template YAML to copy.
- `docs/`:
  - `ONBOARDING.md` — math-first walkthrough.
  - `PAPER_NOTES.md` — production design log + canonical scorecard.
  - `PAIR_FIT_PLAN.md` — Phase 2 pair cross-coupling design.
  - `AP_REMEDIATION_PLAN.md` — Phase 3 plan for the Ap σ-ratio gap.

---

## Project status (2026-05-07)

Phase 1.5 (per-1D PySR + additive combine, smart kwargs for IGM
thermal block) and Phase 2 (4-pair cross-coupling) are landed; PR #2
(Phase 2) is open and mergeable. Headline: 4-pair PySR emulator hits
**2.35% mean / 7.05% p99 / 12.11% max** rel-err on the 11-θ Sobol
hold-out at z=3.6, KSData k-grid. See `docs/PAPER_NOTES.md` for the
full scorecard.
