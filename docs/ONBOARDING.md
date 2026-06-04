# priya-forecast — single-z pipeline guide

This guide covers the **single-z forecast pipeline** that lives in
`src/priya_forecast/single_z/`. One YAML file controls one forecast at
one redshift bin; three modes let you go from a fast GP sanity-check all
the way through a full PySR refit and symbolic-model forecast.

The document is in five parts:

1. What the forecast computes (θ → P_F → Fisher → σ).
2. The three modes and when to use each.
3. The student procedure: regen → refit → forecast → aggregate.
4. The additive-Taylor combine and the σ ladder.
5. A reading map for the source files.

---

## 1. What the forecast computes

### 1.1 The θ → P_F → Fisher → σ chain

We have a PRIYA Lyman-α flux power spectrum `P_F(θ; k, z)` parameterized
by **11 cosmology + IGM-thermal parameters θ** (table below). We have a
measured covariance `C` on a fixed k-grid at one redshift bin. The
question is: **how tightly do those data constrain each θ_i?**

For a Gaussian likelihood with parameter-independent covariance, the
**Fisher information matrix** is

```
  F_ij = (∂m/∂θ_i)^T  C^{-1}  (∂m/∂θ_j)
```

where `m(θ)` is the model prediction `P_F(θ; k)` stacked over the k-grid
and `C` is the measurement covariance. The **marginalized 1σ error** on
`θ_i` is

```
  σ_i = sqrt( (F^{-1})_{ii} )
```

The derivatives are evaluated by a 5-point centered stencil with adaptive
step halving (see `src/priya_forecast/fisher.py`). The step starts at
`step_frac × (prior_hi − prior_lo)` and halves until the relative change
in `F_{ii}` drops below `rel_tol`.

### 1.2 The 11 PRIYA parameters

| # | name | fiducial | prior |
|---|------|---------|-------|
| 0 | `dtau0` | −0.009 | (−0.4, 0.25) |
| 1 | `tau0` | 1.090 | (0.75, 1.25) |
| 2 | `ns` | 0.983 | (0.8, 1.05) |
| 3 | `Ap` | 1.46 | (1.2, 2.6) — units of 10⁻⁹ internally |
| 4 | `herei` | 4.0 | (3.5, 4.5) |
| 5 | `heref` | 2.765 | (2.2, 3.2) |
| 6 | `alphaq` | 1.74 | (1.3, 3.0) |
| 7 | `hub` | 0.688 | (0.65, 0.75) |
| 8 | `omegamh2` | 0.1439 | (0.140, 0.146) |
| 9 | `hireionz` | 7.24 | (6.5, 8.0) |
| 10 | `bhfeedback` | 0.050 | (0.03, 0.07) |

Source: `src/priya_forecast/parameters.py` — the single source of truth
for fiducials and prior bounds.

### 1.3 The two data sources

The covariance `C` is always a **real measured covariance** — never
synthetic:

- `data.source: kodiaq` → KODIAQ-SQUAD (Karaçaylı et al. 2021),
  loaded via `KSDataLikelihood`. `conservative: true` drops the first
  4 k-bins. `mock_data: gp` uses the GP at fiducial as the data vector
  (so the Fisher is a "mock" forecast, not a fit to real power spectra,
  but the noise model is real).
- `data.source: eboss_dr14` → eBOSS DR14, loaded via `GaussianLikelihood`
  + the internal `load_eboss` function. Also uses real measurement noise.

---

## 2. The three modes

All three modes are controlled by the `mode:` key at the top of the YAML.
They share the same config schema (`src/priya_forecast/single_z/config.py`)
and the same entry point (`scripts/run_pipeline.py`).

### 2.1 `gp_only` — Fisher on the GP emulator (σ_GP)

Uses the GP emulator directly as the forward model. No PySR at all.
Output: `forecast_table.txt`, `scorecard.md`.

**When to use:** sanity check that the pipeline wires up correctly; get
the GP reference σ_GP before running any symbolic model.

```bash
python scripts/run_pipeline.py --config configs/single_z/example.yaml \
    --mode gp_only
```

### 2.2 `forecast_only` — load Pareto CSVs → equations → σ ladder

Loads per-parameter Pareto CSV files, picks one equation per parameter
(Fisher-safety filter then `pick:` rule), builds the additive-Taylor
combined model, and runs the three Fisher forecasts: **σ_GP**,
**σ_perfect_1D**, **σ_PySR**.

Pareto CSVs can come from three places (set by `pareto_csvs.source:`):

- `bundled_baseline` — vendored baseline CSVs shipped with the repo
  (`src/priya_forecast/_vendored/data/pareto_baseline/z{z}/`).
- `per_parameter` — paths given explicitly in the YAML under
  `pareto_csvs.per_parameter:`.
- `from_refit` — output directory populated by a previous
  `refit_and_forecast` run (`<output_dir>/refit/z{z}/pareto_{param}.csv`).

If no Pareto CSVs are available at runtime, the mode falls back to
emitting only σ_GP and σ_perfect_1D and marks σ_PySR as unavailable in
the scorecard.

**When to use:** you already have Pareto CSVs (from a prior refit or the
bundled baseline) and want to compare equations against the GP.

```bash
python scripts/run_pipeline.py --config configs/single_z/example.yaml \
    --mode forecast_only
```

### 2.3 `refit_and_forecast` — run PySR refits, then forecast

Runs the full loop: build LF + HF GP models, call PySR per parameter
(11 in-process sequential refits), write `pareto_{param}.csv` into
`<output_dir>/refit/z{z}/`, then run the three Fisher forecasts from the
fresh CSVs.

Requires: Julia + PySR installed and the environment variables
`PYTHON_JULIAPKG_PROJECT` and `JULIA_DEPOT_PATH` pointing at a pre-built
Julia depot (see the Greatlakes working environment notes).

**When to use:** you want to refit PySR equations from scratch at a new
z-bin or after changing the training data.

```bash
python scripts/run_pipeline.py --config configs/single_z/example.yaml \
    --mode refit_and_forecast
```

---

## 3. The student procedure

The full end-to-end student workflow is a four-step pipeline:

```
regen_1pvar.py  →  refit (PySR)  →  forecast (Fisher)  →  aggregate_z.py
```

### Step 1 — Regenerate per-parameter training data

```bash
python scripts/regen_1pvar.py \
    --basedir data/kodiaq_gp \
    --output  data/single_z_1pvar
```

This sweeps each of the 11 PRIYA parameters over 50 points across its
prior (with the other 10 held at fiducial) and evaluates the LF and HF GP
emulators on the kodiaq-squad k-grid at all 13 z-bins. It writes
`{lf,hf}_{param}_npoints50.hdf5` into `data/single_z_1pvar/`.

Key facts:
- Uses `data/kodiaq_gp/` as the GP basedir — the same emulator the
  forecast scores against, so training data and ground truth are
  self-consistent.
- Stores **raw P_F** (not `k·P_F/π`), matching the PySR target convention.
- Covers all 13 z-bins (`z = 2.2, 2.4, …, 4.6`). Run once; each z-bin
  uses a slice of the same HDF5s.

Optional flags: `--kmin`, `--kmax`, `--nk`, `--params` (subset).

### Step 2 — Refit (PySR)

For a single z-bin, use `run_pipeline.py` with `mode: refit_and_forecast`.
For all 13 z-bins at once on SLURM (recommended for the full run):

```bash
# Submit 13 SLURM array jobs (one per z-bin, 11 params each):
python scripts/run_batch.py \
    --config configs/single_z/example.yaml \
    --mode   refit_and_forecast \
    --phase  submit

# After all jobs finish, collect results and forecast:
python scripts/run_batch.py \
    --config configs/single_z/example.yaml \
    --mode   refit_and_forecast \
    --phase  collect
```

Each SLURM task runs `scripts/refit_one_param_single_z.py` for one
`(param, z)` pair and writes `pareto_{param}.csv`. The array job template
is `slurm/single_z_refit.slurm`.

You can also refit a single `(param, z)` pair directly:

```bash
python scripts/refit_one_param_single_z.py \
    --param ns --z 3.6 \
    --basedir data/kodiaq_gp \
    --output-dir results/single_z_run
```

### Step 3 — Forecast

For a single z-bin:

```bash
python scripts/run_pipeline.py \
    --config configs/single_z/example.yaml \
    --mode   forecast_only
```

This reads the Pareto CSVs (from `pareto_csvs.source:`), picks one
equation per parameter, builds the combined model, and runs the three
Fisher forecasts. Outputs: `forecast_table.txt`, `scorecard.md`,
`corner.png`, per-label `fisher_{GP,perfect_1D,PySR}.npz`.

### Step 4 — Aggregate across z

```bash
python scripts/aggregate_z.py --base results/single_z_run
```

Reads all 13 per-z `fisher_*.npz` files and writes
`results/single_z_run/aggregate/`:
- `sigma_vs_z.png` — σ(z) trend plot per parameter, GP / perfect_1D /
  PySR overlaid.
- `sigma_table.md` — params × z table with σ values.

`run_batch.py` calls `aggregate_z.aggregate()` automatically after the
forecast loop completes.

---

## 4. The additive-Taylor combine and the σ ladder

### 4.1 The combine formula

The single-z forward model in `forecast_only` and `refit_and_forecast`
modes is built by `src/priya_forecast/single_z/combine.py` using
`refit_taylor.AdditiveTaylorModel` in **`local_anchored` mode**:

```
P_F(θ, k) = P_GP(θ_fid, k)
           + Σ_i [ eq_i(θ_i, k, r=0.8) − eq_i(θ_i_fid, k, r=0.8) ]
```

where each `eq_i` is a single-z PySR equation with inputs
`(θ_i_norm, k_norm, resolution)` mapped to `[0, 1]`, and `0.8` is the
HF resolution feature.

Properties worth remembering:

1. **Anchored at fiducial.** At `θ = θ_fid`, every bracket is zero, so
   `P_F(θ_fid, k) ≡ P_GP(θ_fid, k)`. The forecast is exactly the GP at
   the fiducial point by construction.
2. **Axis-wise exact w.r.t. the 1D surrogate.** If only `θ_i` is varied,
   all other brackets vanish and the response is exactly the 1D PySR
   equation's prediction (offset from its own fiducial evaluation). The
   accuracy relative to the GP is whatever the 1D PySR fit achieves.
3. **Cross-parameter interactions are dropped.** The combine is a first-
   order functional ANOVA (additive main-effect model); pair-wise and
   higher interaction terms between distinct parameters are absent.
4. **GP-slice fallback.** If `refits[param_name]` is `None`, that
   parameter's contribution falls back to a GP 1D slice:
   `P_GP(θ_fid except θ_i, k) − P_GP(θ_fid, k)`. The parameter's
   gradient in the Fisher computation then matches the GP gradient exactly.

### 4.2 The σ ladder

The three Fisher forecasts differ only in their forward model:

| Label | Forward model | Meaning |
|-------|--------------|---------|
| **σ_GP** | Full GP emulator | The baseline: best possible single-z Fisher constraint |
| **σ_perfect_1D** | Additive combine with GP 1D slices for every parameter (all refits=None) | The combine's ceiling with the exact GP responses |
| **σ_PySR** | Additive combine with fitted PySR equations | The constraint achievable from the symbolic model |

The ratios `σ_PySR / σ_GP` and `σ_perfect_1D / σ_GP` measure information
loss at two different levels.

**Important scientific subtlety: σ_perfect_1D ≡ σ_GP for the additive
combine.**

Because the Fisher matrix is built from first derivatives, and the
additive combine reproduces the GP's exact first derivatives along each
parameter axis (when all refits are GP-sliced, the deviation term is the
GP 1D slice itself), the Fisher matrices for σ_GP and σ_perfect_1D are
identical up to numerical precision. In practice `σ_perfect_1D / σ_GP ≈
1.000` for every parameter.

This means **the meaningful headline metric is `σ_PySR / σ_GP`** — the
constraining power lost to PySR fit error. A ratio of 1.0 means the
symbolic model is as constraining as the GP; a ratio > 1 means the PySR
equation is less informative along that parameter axis.

The ratio `σ_perfect_1D / σ_GP` is provided as a numerical sanity check
that the pipeline is self-consistent, not as an independent measure of
combine quality.

---

## 5. Reading map

The single-z pipeline is factored across six source files. Read them in
this order:

| File | What it does |
|------|-------------|
| `src/priya_forecast/single_z/config.py` | YAML schema — `PipelineConfig`, all sub-configs, `load_config()`. Start here to understand the knobs. |
| `src/priya_forecast/single_z/pipeline.py` | Entry points: `run()`, `run_gp_only()`, `run_forecast_only()`, `run_refit_and_forecast()`. The dispatcher and output writers. |
| `src/priya_forecast/single_z/forecast.py` | `forecast_only` internals: `resolve_pareto_csvs()`, `build_refit_from_pareto()`, `run_three_fisher()`. |
| `src/priya_forecast/single_z/refit.py` | `refit_and_forecast` internals: `refit_one_param_single_z()`, `pysr_kwargs_for_cfg()`, `kodiaq_k_grid()`. |
| `src/priya_forecast/single_z/combine.py` | `build_combined_model()` — thin wrapper over `refit_taylor.AdditiveTaylorModel`. |
| `src/priya_forecast/single_z/training_data.py` | HDF5 I/O for regenerated 1pvar data: `write_1pvar_hdf5()`, `load_1pvar()`, `regenerate_param()`. |

Supporting library files (not single-z-specific):

| File | What it does |
|------|-------------|
| `src/priya_forecast/parameters.py` | The 11 PRIYA params: names, fiducials, priors. |
| `src/priya_forecast/fisher.py` | Fisher matrix: 5-point stencil, adaptive step halving, `FisherResult`. |
| `src/priya_forecast/refit_taylor.py` | `AdditiveTaylorModel` (single-z) and `MultiZAdditiveTaylorModel` (multi-z). |
| `src/priya_forecast/ksdata_likelihood.py` | KODIAQ-SQUAD likelihood with real covariance. |
| `src/priya_forecast/likelihood.py` | `GaussianLikelihood` (eBOSS DR14). |
| `src/priya_forecast/models/pysr_model.py` | `load_pareto_csv()`, `pick_equation()`, equation compilation. |
| `src/priya_forecast/pareto_filters.py` | `is_fisher_stencil_safe()`, `has_pathological_constant()` — the Fisher-safety filters applied before `pick_equation`. |

---

## 6. One-time setup

```bash
git clone <this repo> && cd priya-forecast
pip install -e ".[forecast,pysr,gp,dev]"
export PYTHONPATH=/path/to/lya_emulator_full:$PWD/src
```

On Greatlakes the upstream emulator is at
`/home/mfho/student_projects/lya_emulator_full`.

The `data/kodiaq_gp/` directory (the GP basedir) is already committed as a
stripped copy of the full emulator directory; no extra prep is needed for
`gp_only` or `forecast_only` modes. For `refit_and_forecast`, the Julia
depot environment variables must be set (see the working environment notes
in `.claude/memory/working_environment.md`).

Tutorial notebooks for each mode live in `notebooks/`:
- `notebooks/01_gp_only.ipynb`
- `notebooks/02_forecast_only.ipynb`
- `notebooks/03_refit_and_forecast.ipynb`
