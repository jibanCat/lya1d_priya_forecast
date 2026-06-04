# Multi-z Stage 7 — joint multi-z Fisher forecast (`F = Σ_z F(z)`)

**Date:** 2026-06-01
**Branch:** `single_z_forecast_clean`
**Status:** design approved, pending spec review

## 1. Background and motivation

The single-z pipeline (Stages 1–6) is complete and validated end to end,
including the Stage 6 `log(P)` target that attenuated the Fisher's-Mirage
severity at z=3.6. Two findings from those runs motivate this stage:

1. **Single-z all-11-param Fisher is rank-deficient.** σ_GP explodes for
   the IGM-thermal parameters (herei/heref/alphaq/hireionz) because a
   single redshift bin does not break their degeneracies. This is
   physical, not a bug — PRIYA itself is fit multi-z, and the thermal
   parameters need redshift leverage.
2. **σ_PySR is derivative-unfaithful** ("Fisher's Mirage", arXiv:2406.06067).
   Stage 6's `log(P)` target reduced the *severity* (mean |log10(σ_PySR/σ_GP)|
   0.615 → 0.366, deep-Mirage params 3 → 0) but did not eliminate it.

The 2026-05-18 research verdict prioritized: **#1 log(P)** (Stage 6, done),
**#2 multi-z forecast `F = Σ_z F(z)`** (this stage), **#3 Sobolev
derivative loss** (Stage 8). Stage 7 unblocks the IGM-thermal parameters
by combining the Fisher information across redshift bins.

## 2. Key architectural finding

`KSDataLikelihood` is **already multi-z native**. Its constructor takes a
`z_min`/`z_max` range, slices the full KODIAQ-SQUAD covariance over that
range (`all_cov[np.ix_(kept_idx, kept_idx)]`, including any cross-z terms),
and builds `z_blocks`. `_predict_stacked` loops the blocks calling
`model.predict(theta, k_block, z_value)` and stacks one joint data vector
(`src/priya_forecast/ksdata_likelihood.py:150,205–213`).

Consequence: a multi-z Fisher forecast does **not** require new Fisher
math. Feed one z-spanning `KSDataLikelihood` a model whose `predict(θ,k,z)`
responds to z, and the **existing validated `fisher_matrix`** returns the
joint Fisher — which *is* `Σ_z F(z)` plus whatever cross-z covariance the
real data carries.

## 3. Approach: joint likelihood (A), with per-z-sum (B) as test oracle

**Approach A (primary).** One `KSDataLikelihood(z_min, z_max)` over the
range + multi-z models + existing `fisher_matrix`. Minimal new Fisher
code; reuses Stages 1–6.

**Approach B (oracle only).** Per-z `compute_fisher_F_phys` summed with
`combine_fisher_phys_arrays` (the legacy `scripts/multi_z_aggregate.py`
path). Assumes block-diagonal-in-z covariance.

A and B are mathematically identical when the covariance is block-diagonal
in z, and A is strictly more correct otherwise. **A is the deliverable; B
is retained as a cross-check test** (Section 7, test 2) — if the real
KODIAQ covariance has cross-z terms, the A-vs-B test reveals it rather
than hiding it.

## 4. Package layout

New sibling package `src/priya_forecast/multi_z/` mirroring `single_z/`,
reusing the shared numerical blocks. Stages 1–6 `single_z/` code is
**untouched**.

```
src/priya_forecast/multi_z/
  config.py     # MultiZPipelineConfig (parallels PipelineConfig)
  pipeline.py   # DISPATCH = {gp_only, forecast_only, refit_and_forecast}
  forecast.py   # run_three_fisher_multiz + output writers
  combine.py    # build_combined_model_multiz (wraps MultiZAdditiveTaylorModel)
  refit.py      # build_refit_from_pareto_multiz (4-input CSV → Refit1DResult)
scripts/
  refit_one_param_multi_z.py   # one param → 4-input Pareto CSV
slurm/
  multi_z_refit.slurm          # 11-param array
```

Reused as-is: `MultiZAdditiveTaylorModel` (`refit_taylor.py`),
`refit_1d_multiz_for_param` / `compute_local_normalization_multiz` /
`_build_training_matrix_multiz` (`refit_1d_pysr.py`),
`MultiZNormalizationSpec` (`models/normalization.py`), `fisher_matrix` /
`compute_fisher_F_phys` / `combine_fisher_phys_arrays` (`fisher.py`),
`KSDataLikelihood`.

## 5. The three models

Each of the three forecast σ's comes from a model plugged into the joint
likelihood:

| Forecast | Model | Invariant |
|----------|-------|-----------|
| σ_GP | HF GP `P1DModel` directly | baseline |
| σ_perfect_1D | `MultiZAdditiveTaylorModel(refits=None)` | GP-slice fallback everywhere → **must ≡ σ_GP** (rtol 1e-3), the Stage-6 anchor identity, now per-z |
| σ_PySR | `MultiZAdditiveTaylorModel(refits=<4-input eqs>)` | the actual test |

### 5.1 Log-space branch in `MultiZAdditiveTaylorModel` (the new numerics)

Today `MultiZAdditiveTaylorModel.predict` is linear-only. Mirror Stage 6's
`AdditiveTaylorModel` transcription to the per-z dict structure:

- add `log_space: bool` field;
- build per-z caches `_log_p_gp_fid_per_z[z]` and
  `_eq_at_fid_logpf[(pname, z)]` (logs of the existing fiducial caches);
- log-space `predict(theta, k, z)`:
  `out_log = _log_p_gp_fid_per_z[z].copy()`; for each refit param accumulate
  `r.predict_log(θ_i, k, z) − _eq_at_fid_logpf[(pname, z)]`; for each
  GP-slice fallback param accumulate `log P_slice − log P_gp_fid`; return
  `exp(out_log)`;
- positivity guards raising a clear `ValueError` on every log path
  (Stage 6 convention).

This is a direct transcription of validated single-z code; no new
mathematics.

## 6. Config, modes, refit data flow

### 6.1 `MultiZPipelineConfig`

Parallels `PipelineConfig`, with the z-bin field generalized:

- `mode`: `gp_only | forecast_only | refit_and_forecast`
- `z_min`, `z_max` (default the KODIAQ range, `2.6 … 4.2`) — replaces
  single-z `redshift`
- `parameters`, `k_range`, `data`, `combine="additive"`, `pick`,
  `target_space="linear"|"log"`, `pareto_csvs`, `fisher` — identical to
  single-z
- `validate()` rejects `combine="multi_d"` on the log path (Stage 6 rule)

### 6.2 Modes (dispatch mirrors single-z)

- **`gp_only`** — joint σ_GP only. The decisive baseline: tests whether
  `F = Σ_z F(z)` resolves the IGM-thermal rank-deficiency.
- **`forecast_only`** — load per-param 4-input Pareto CSVs → σ_GP /
  σ_perfect_1D / σ_PySR + corner.
- **`refit_and_forecast`** — train 4-input equations in-process via
  `refit_1d_multiz_for_param`, then forecast.

### 6.3 Refit interchange = per-param 4-input Pareto CSVs

The clean-pipeline format (per-param CSV), **not** the legacy bundled
`multid_refit.pkl`:

- `scripts/refit_one_param_multi_z.py` + `slurm/multi_z_refit.slurm`
  (11-param array) emit `pareto_<param>.csv` for the 4-input equation
  `eq(θ_norm, k_norm, res, z_norm)`, via `refit_1d_multiz_for_param`.
  SLURM reuses the single-z env block (`PYTHON_JULIAPKG_PROJECT`,
  `JULIA_DEPOT_PATH`, `TARGET_SPACE`).
- `forecast_only` reconstructs refits via a multi-z
  `build_refit_from_pareto_multiz` that rebuilds a 4-input `Refit1DResult`
  carrying `z_min`/`z_max` + `MultiZNormalizationSpec`.
- The legacy `results/refit_multid_z2.6-4.2/multid_refit.pkl` is the
  migration/oracle input for the A-vs-B cross-check, **not** the
  production format.

## 7. Outputs

`multi_z/forecast.py` writes to `cfg.output_dir`, mirroring single-z
naming so existing tooling/notebooks transfer:

- `fisher_{GP,perfect_1D,PySR}.npz` — the **joint** Fisher/covariance over
  the z-range.
- `forecast_table.txt` — σ_GP / σ_perfect_1D / σ_PySR per param + the
  σ_PySR/σ_GP ratio (the Fisher's-Mirage metric).
- `corner.png` — joint 11-param corner. Headline: IGM-thermal params
  should now be constrained, not rank-deficient.
- `scorecard.md` — mean |log10(σ_PySR/σ_GP)|, sub-1 Mirage count,
  deep-Mirage count, **GP-slice fallback count** — same metrics as
  Stage 6's COMPARISON.md.
- `COMPARISON.md` — multi-z vs. the Stage 6 single-z z=3.6 numbers:
  (1) σ_GP no longer explodes for IGM-thermal params, (2) the effect of
  multi-z leverage on the Mirage.

Optional, low cost: per-z σ_GP-vs-z diagnostic reusing the
`aggregate_z.py` plotting style.

## 8. Testing (TDD, mirroring the Stage 6 suite)

1. `perfect_1D ≈ GP` on the joint multi-z Fisher (rtol 1e-3) — linear and
   log space.
2. **A-vs-B equivalence**: joint Fisher (A) == `Σ_z compute_fisher_F_phys`
   (B) when the covariance is block-diagonal in z — pins A against the
   legacy machinery as an oracle.
3. Log-space units: per-z positivity guards raise `ValueError`;
   `exp(log predict) == linear predict` at θ=fid; anchor identity holds
   per z.
4. `combine="multi_d"` rejected on the log path.
5. Gated end-to-end (`RUN_SLOW_*`) `refit_and_forecast` on real KODIAQ
   over a 2-bin range.
6. `build_refit_from_pareto_multiz` round-trips a 4-input CSV back to a
   working `Refit1DResult`.

## 9. Risks and watch-items

- **4-input PySR is harder to fit** than single-z; equations may fail the
  x0-dependence / rel-err gates → GP-slice fallback. The forecast stays
  valid (perfect_1D ≡ GP), but the σ_PySR story weakens if many params
  fall back. Surface the fallback count in the scorecard.
- **Joint Fisher conditioning** should improve over single-z, but a
  too-narrow z-range may remain soft on IGM-thermal params — the z-range
  is config-driven so it can be widened.
- **Cross-z covariance unknown.** If KODIAQ's real covariance has cross-z
  terms, A and B diverge; the A-vs-B test surfaces this rather than hiding
  it.
- **Environment quirks** (HANDOFF 2026-05-20): numpy<2 + five legacy-numpy
  GPy/paramz patches; PySR needs `PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env`
  and `JULIA_DEPOT_PATH=$HOME/.julia`. Reuse the single-z SLURM env block.

## 10. Out of scope

- Stage 8 (Sobolev derivative-matching loss) — the next stage; addresses
  the residual Mirage that multi-z leverage does not.
- z-interpolation of the covariance (nearest-z lookup only, per
  `MultiZNormalizationSpec.denormalize_flux`).
- Retiring the legacy `scripts/multi_z_aggregate.py` — kept as the B oracle.
