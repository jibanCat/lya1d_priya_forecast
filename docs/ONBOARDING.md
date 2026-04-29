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

## 9. The reward loop: `scripts/train_and_forecast.py`

This is the script you run after every PySR retraining. It takes your
equation set and produces a production scorecard with three gauges:

```
PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \
    python scripts/train_and_forecast.py \
        --params dtau0 Ap ns alphaq \
        --equations path/to/your_pysr.yaml \
        --output results/run_$(date +%Y%m%d)/
```

The scorecard reports three numbers per parameter, plus geometric means:

| Gauge | What it tells you |
|---|---|
| **σ_student / σ_perfect_1D** | how far you are from the 1D-product upper bound. Target < 1.5 (geomean). |
| σ_perfect_1D / σ_gp | always ≈ 1 at Fisher level by chain rule — sanity check, not actionable. |
| **off-fid MSE ratio** | residuals vs GP at random off-fid Sobol points, in eBOSS-σ² units. Target < 2. |

Built-in references:

```
--equations none        # only score the references
--equations published   # the four equations from the user-quoted paper draft
--equations <path.yaml> # your YAML
```

### What the scorecard tells you to do

- If `σ_student / σ_perfect_1D` is large (currently ~6× geomean for the
  published equations): your PySR runs are under-trained. Try larger
  `maxsize` (30+), more `niterations` (200+), and `model_selection="accuracy"`
  with a tight target loss.
- If `off-fid MSE ratio` is large (currently ~66× for published): even at
  fid the gradients look OK but extrapolation breaks down. Train on a
  denser Sobol set covering more of the prior, not just near fid.
- If both are small: you've saturated 1D-product. Now train *joint*
  multi-D equations (`combine: joint` with a single equation in all
  varying params) and rerun. Phase 5's coupling-matrix heatmap will show
  where the joint approach actually buys you something.

### Why σ_perfect_1D = σ_gp at Fisher level

By the chain rule, at the fiducial point:
`∂P_pysr/∂θ_i = P_fid · (1/f_i_fid) · df_i/dθ_i = ∂GP/∂θ_i` whenever the
1D slice f_i matches the GP at fid. So Fisher cannot see the cost of the
1D-factorization assumption — that cost shows up only at off-fid points
(in the off-fid MSE gauge above) or in MCMC posteriors with non-Gaussian
curvature.

## 10. Multi-D PySR — `scripts/run_multid_pysr.py`

When `train_and_forecast.py` says "graduate to multi-D PySR" (i.e. your
1D equations are converged and you want the cross-term gain), run the
multi-D trainer. It samples Sobol points in your chosen subspace,
evaluates the GP, runs **real PySR** to discover a single joint
equation in (θ_1, ..., θ_k, k), then scores it via the same scorecard:

```
PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env \
JULIA_DEPOT_PATH=$HOME/.julia \
PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \
    python scripts/run_multid_pysr.py \
        --params dtau0 Ap \
        --n-train 128 --niter 200 --maxsize 30 \
        --output results/multid_2d_dtau0_Ap/
```

- `--params`: forecast subspace (start with 2 or 3 — 4D is hard for
  PySR at modest budgets).
- `--n-train`: Sobol training-set size (more = better fit, slower).
- `--niter`: PySR iterations. 30 is a smoke test; 200+ is real.
- `--maxsize`: Pareto-front complexity cap. 25 is a reasonable starting
  point; 40 if you want richer equations.

The script saves the discovered Pareto CSV to `<output>/hall_of_fame.csv`
and the chosen equation's scorecard to `<output>/scorecard.md`. Iterate
on (`niter`, `maxsize`, `n_train`) until σ_student / σ_perfect_1D drops
below ~1.5 across the chosen subspace.

A small budget (n_train=32, niter=30, maxsize=25, params=4) is enough
to confirm the plumbing but produces equations that miss most of the
parameter dependence (e.g., alphaq dropped by parsimony). For
publishable runs, expect to spend tens of minutes per equation.

## 11. The coupling-matrix diagnostic — `scripts/run_coupling_matrix.py`

This is the script that produces the **headline science figure** for
the paper: a coupling matrix that quantifies how much information the
1D-factorization assumption is losing for each parameter pair.

```
PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \
    python scripts/run_coupling_matrix.py \
        --params dtau0 tau0 ns Ap herei heref alphaq hub omegamh2 hireionz bhfeedback \
        --order 4 --n-train 96 --n-test 256 \
        --output results/coupling_matrix/
```

The script:
1. Sweeps Sobol points per (θ_i, θ_j) pair on the GP at fid-others.
2. Fits a 1D polynomial per param (multiplicatively combined into a
   1D-product baseline).
3. Fits a 2D polynomial per pair (the joint reference).
4. Computes `coupling[i,j] = (MSE_1D-product − MSE_2D-joint) / MSE_1D-product`
   on a held-out 2D Sobol test set.
5. Renders the 11×11 heatmap.

**Polynomial backend**: total runtime for all 55 pairs at n_train=96,
order=4 is ~5-10 minutes. Real PySR backend is much slower; see Phase 6
HPO + `scripts/run_multid_pysr.py` for that path.

**Reading the matrix**: cells near 0 mean "1D-factorization is fine for
this pair"; dark cells mean "you need joint multi-D PySR for this pair
to capture the GP's response correctly." The mode of the matrix's
distribution tells you whether the paper's 1D approach is structurally
sound or needs a redo.

If a few specific cells are dark (e.g., omegamh2 × hub), the paper
should report those pairs as known limitations. If most cells are
dark, the 1D approach itself is broken at the chosen z and the paper
needs a multi-D rerun.

## 12. Where to ask for help

- Code questions / bugs → file in this repo's GitHub issues.
- PySR training pipeline questions → upstream `priya_pysr` repo.
- GP emulator questions → upstream `lya_emulator_full` repo (sbird).
- Forecast methodology → check `tests/test_combine_justification.py`
  for the additive-vs-multiplicative numbers, then the PRIYA paper.
