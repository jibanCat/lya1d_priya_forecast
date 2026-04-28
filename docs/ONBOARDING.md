# priya-forecast onboarding

This guide walks a new student through running a Lyman-α P1D forecast from
PySR equations they trained themselves. The framework's job is to take the
equations + the eBOSS DR14 covariance + the GP emulator (as ground truth /
fallback model), and report what 1σ constraints those equations would
yield on the 11 PRIYA parameters.

---

## 1. What the forecast does (in one paragraph)

For a model `m(θ, k)` and eBOSS DR14 measurement `d ± C` at one redshift,
the framework computes the Gaussian likelihood

    log L(θ) = -0.5 (d - m(θ))ᵀ C⁻¹ (d - m(θ))

Then either the **Fisher matrix** at fiducial (a fast Gaussian
approximation) or **MCMC** (full posterior) gives 1σ marginalized errors
on each of the 11 parameters. You can swap `m` between the **GP emulator**
(the published PRIYA model) and a **PySR equation set** (your analytic
fits). The point of the comparison is: if PySR equations give nearly the
same constraints as the GP, then the equations are publishable as a
drop-in interpretable replacement.

---

## 2. Setup (one-time)

```
git clone <this repo>
cd priya-forecast
pip install -e ".[forecast,pysr,gp,dev]"
```

You also need the upstream `lyaemu` package (sbird/lya_emulator) on your
PYTHONPATH for the real GP. On Greatlakes it's already cloned at
`/home/mfho/student_projects/lya_emulator_full`. Set:

```
export PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:$PWD/src
```

The eBOSS DR14 data + covariance is vendored inside the repo
(`src/priya_forecast/_vendored/data/boss_dr14_data/`); you don't need to
download anything.

---

## 3. The end-to-end flow

```
[your PySR training run]                     [this framework]

hall_of_fame_*.csv  ────────────┐
mean_flux_low_*.txt  ────────┐  │
std_flux_low_*.txt   ──┐     │  │
                       │     │  │
                       ▼     ▼  ▼
                ┌────────────────────────────┐
                │  configs/eqns/my_pysr.yaml │  ← you write this
                └────────────┬───────────────┘
                             │
                ┌────────────▼───────────────────┐
                │  priya-forecast run            │  ← framework runs
                │      Fisher / MCMC             │
                │  vs GP emulator + eBOSS cov    │
                └────────────┬───────────────────┘
                             │
                ┌────────────▼───────────────────┐
                │ results/<run>/                 │
                │   fisher.npz                   │
                │   chain.h5                     │
                │   figures/*.png                │
                └────────────────────────────────┘
```

---

## 4. Writing your YAML

For each parameter you trained PySR on, point the YAML at your
`hall_of_fame_<param>_z<z>.csv` and pick which row of the Pareto front to
use:

```yaml
# configs/eqns/my_pysr.yaml
name: my_pysr
model: pysr
redshift: 3.6
combine: multiplicative                         # or additive / joint
fiducial_p1d: data/priya_fiducial/p1d_z3.6.npz  # cached GP at fid

parameters:
  bhfeedback:
    pareto_csv: pysr_outputs/hall_of_fame_bhfeedback_z3.6.csv
    pick: best_loss             # or complexity_le:15 / accuracy_at:1e-3 / row:I
    fiducial: 0.05              # PHYSICAL units, not normalized
    variables: [bhfeedback, k, resolution]   # column order PySR was trained on
  ns:
    pareto_csv: pysr_outputs/hall_of_fame_ns_z3.6.csv
    pick: complexity_le:15
    fiducial: 0.983
  # ... 11 entries total
```

### `pick:` rules

| Rule                    | What it picks |
|-------------------------|---------------|
| `best_loss`             | the row with the lowest `Loss` value |
| `complexity_le:N`       | among rows with `Complexity ≤ N`, the one with lowest `Loss` |
| `accuracy_at:tol`       | among rows with `Loss ≤ tol`, the one with smallest `Complexity` |
| `row:I`                 | the I-th row, by 0-indexed position |

`accuracy_at:` is the right choice for the paper — "the smallest equation
that still hits target loss." Use `complexity_le:15` if you've decided in
advance how interpretable the equation should be.

### `combine:` rules

The forecast is built from per-parameter equations in one of three ways:

- `multiplicative`: `P(θ, k) = P_fid(k) · ∏ᵢ [fᵢ(θᵢ, k) / fᵢ(θᵢ_fid, k)]`
- `additive`     : `P(θ, k) = P_fid(k) + Σᵢ [fᵢ(θᵢ, k) - fᵢ(θᵢ_fid, k)]`
- `joint`        : a single equation in (θ_1, ..., θ_11, k); set `joint_expression:`.

Multiplicative is exact when each parameter scales the amplitude
independently (often the case for cosmology); additive is exact when the
true response is a sum of independent terms. The repo ships a justification
test (`tests/test_combine_justification.py`) that lets you compare both on
synthetic ground truth before deciding.

### `normalization:` block (passed at construction time, not in YAML)

The framework supports three modes:

- `{"mode": "auto"}`: derive (mean_k, std_k) by sweeping the chosen
  parameter via the GP at fixed-fiducial-rest. Reproduces the student's
  `mf_*.py` recipe but uses the GP instead of simulations.
- `{"mode": "files", "mean_flux": ".../mean_flux_low_<subset>.txt", "std_flux": ".../std_flux_low_<subset>.txt"}`:
  load the `.txt` files your PySR training script saved.
- `{"mode": "identity"}`: no normalization (equation is in physical units).

Default is `identity` — set explicitly if your equations were trained on
normalized inputs (the usual case).

### `fix:` map for non-forecast inputs

If your PySR equations were trained on `(θ, k, resolution)`, the
`resolution` column isn't a forecast parameter. Declare it inside the
normalization block:

```python
{"mode": "auto", "fix": {"resolution": 0.8}}
```

The framework refuses to compile any equation with a free variable other
than its parameter and `k`, so silent misuse is impossible.

---

## 5. Running the forecast

```
priya-forecast run \
    --config configs/default.yaml \
    --eqn configs/eqns/my_pysr.yaml \
    --mode fisher \
    --output results/my_run_2026_04_28/
```

Outputs:

- `fisher.npz` — Fisher matrix, covariance, σ vector, correlation matrix.
- `fisher_table.md` — human-readable 1σ summary, one row per parameter.
- `figures/*.png` — diagnostic figures (see the gallery below).

Use `--mode mcmc` for the full posterior (slower; minutes-to-hours).

To compare every YAML in `configs/eqns/` against the GP baseline:

```
priya-forecast compare --eqn-dir configs/eqns/ --output results/compare/
```

---

## 6. Diagnostic figures (gallery)

The repo regenerates a sample set under `docs/figures/` via
`scripts/regen_sample_figures.py`. Each forecast run produces the same set
under its own output directory.

### `fig01_gp_at_fiducial.png` — sanity check
Overlay of the GP prediction at fiducial vs the eBOSS DR14 data points
with ±σ error bars at `z=3.6`. **If the line doesn't track the points
within their errors, something is wrong upstream** (wrong z-bin, wrong
covariance, wrong fiducial values).

### `fig02_gp_param_sensitivity.png` — what the data sees
For each of the 11 parameters, `d ln P_F / dθ̂` per prior-width.
Parameters with large absolute values are well-constrained; parameters
sitting near zero across the k-grid are essentially unconstrained.

### `fig03_fisher_1d.png` — single-parameter constraint
1σ marginalized error from a 1D forecast (only that parameter floats;
others held at fid), normalized to the prior width. This is the
"if I knew everything else, how well would I measure this?" lower bound.

### `fig04_fisher_2d.png` / `fig05_fisher_3d.png` / `fig06_fisher_4d.png`
Same bar chart but jointly marginalizing over 2 / 3 / 4 parameters. Watch
how individual σ's grow as more degeneracies are opened — that growth
*is* the science of degeneracy-breaking.

### `fig07_fisher_corner.png` — full-cov visualization
Fisher-Gaussian corner: 1D marginals on the diagonal, 1σ confidence
ellipses off-diagonal. Identifies which parameter pairs are most
correlated (visually, the most-elongated ellipses).

When you have multiple equation sets to compare (PySR-v1, PySR-v2, GP),
overlay them in this corner — same axes, different colors. If the PySR
contour overlaps the GP contour, your equations are forecast-equivalent
to the emulator.

---

## 7. Common gotchas

**1. "PySR equation references unknown symbols [...]"**
Your YAML declared `variables: [ns, k]` but the equation also uses `r` or
`x2`. Either add the missing variable to `variables:` and put it in
`fix:`, or hand-edit the equation.

**2. "Covariance not positive-definite"**
You set `cov_scale` very small. The DR14 covariance is fine; the scale
multiplier dropped you below the numerical floor.

**3. "Fisher matrix not invertible"**
You're varying parameters with zero gradient (e.g., the equation doesn't
depend on `bhfeedback`). Drop them from the `params=` list in your
forecast call. The 4-parameter `(ns, Ap, hub, omegamh2)` subset is a safe
starting point; the others typically need many redshifts to constrain.

**4. "z=X.X not in emulator's z-grid"**
The GP emulator is trained at z = 2.2, 2.4, ..., 4.6 in steps of 0.2.
Snap your `redshift:` config entry to one of those.

**5. The MCMC chain is short — sigma's look weird**
Look for `MCMC may not be converged: chain length ... < 50 * tau_max`
in the warnings. Bump `n_steps:` in `configs/default.yaml`.

---

## 8. Where the tests live

Every module has unit tests + property-based tests under `tests/`:

- `test_parameters.py` — the 11 PRIYA params, fiducial + prior bounds
- `test_config.py` — YAML loaders + validation
- `test_data.py` — eBOSS DR14 load + binning
- `test_normalization.py` — round-trip of mean_flux/std_flux
- `test_pysr_model.py` — sympy whitelist + Pareto pick + combine math
- `test_gp_model.py` — Mock + real GP adapter
- `test_likelihood.py` — Gaussian chi² + cov_scale + NaN rejection
- `test_fisher.py` — closed-form check on linear models
- `test_mcmc.py` — recovers fiducial within 4σ on a 2D toy
- `test_combine_justification.py` — additive vs multiplicative on synthetic + MockGP

Run all of them with:

```
PYTHONPATH=src python -m pytest tests/ -q
```

Adding a new feature? Add a unit test + at least one `hypothesis`
property-based test in the same PR. The repo has a CI rule that requires
both.

---

## 9. Where to ask for help

- Code questions / bugs → file in this repo's GitHub issues.
- PySR training pipeline questions → upstream `priya_pysr` repo.
- GP emulator questions → upstream `lya_emulator_full` repo (sbird).
- Forecast methodology → check `tests/test_combine_justification.py`
  for the additive-vs-multiplicative numbers, then the PRIYA paper.
