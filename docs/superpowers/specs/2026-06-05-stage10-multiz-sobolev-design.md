# Stage 10 — Sobolev loss + derivative gate, mirrored to multi-z

**Date:** 2026-06-05
**Branch:** `stage10-multiz-sobolev`
**Status:** design approved (user reviews on diagnostic plots, not the spec).

## 1. Motivation

Stage 9 proved the Sobolev derivative loss recovers ns at single-z. But the
single-z σ_PySR/σ_GP metric is **rank-deficiency-confounded** (Stage 8), so the
*scientific* payoff can only be seen on the **well-conditioned multi-z** forecast
(Stage 7's regime, where σ is interpretable). Stage 10 mirrors the Sobolev refit
+ derivative gate into the multi-z path and evaluates there, with diagnostic
plots as the deliverable.

## 2. Scope

Mirror two single-z capabilities into multi-z:
- **Sobolev loss** in `refit_1d_multiz_for_param`.
- **Derivative-validation gate** in multi-z equation selection.

Architecture unchanged otherwise (per-param multi-z 4-input refits, additive
combine, joint Fisher). λ=5 (re-validated on multi-z).

## 3. Design

### 3.1 Multi-z Sobolev weights (`sobolev_target_weights_multiz`)

Mirror `sobolev_target_weights` (Stage 9), with two multi-z differences:
- **Per-row z:** each Sobol training point `r` has its own redshift
  `z_per_row[r]`. The GP target gradient is `∂logP/∂θ` evaluated at that point's
  `(θ_r, k, z_r)`, per fidelity (LF gradient for LF rows, HF for HF rows).
- **Per-(z,k) std:** the multi-z norm is `MultiZNormalizationSpec` with
  `std_flux` shape `(n_z, n_k)`. The normalization factor for row `r` uses
  `std_flux[z_index(z_r), :]`.

Row order matches `_build_training_matrix_multiz`: `X_act = vstack([X_lf, X_hf])`,
within a fidelity point-major / k-minor (`x_param.ravel()` over `(n_total,
n_k)`). Boundary-clamp the θ-perturbation to `[x_param_min, x_param_max]` (the
Stage 9 emulator-range fix). Normalized to `(x0, std)` space:
`weight = (∂logP/∂θ_phys) · (x_param_max−x_param_min) / std_k(z_r)`.

### 3.2 Wire Sobolev into `refit_1d_multiz_for_param`

Add `use_sobolev`, `sobolev_lambda`, `sobolev_h` kwargs (mirror Stage 9 Task 4):
when on, build the multiz weights, set `args["loss_function"] =
make_sobolev_loss(λ, h)` (drop `elementwise_loss`), `model.fit(X_act, Y_act,
weights=...)`. Guard that `gp_lf`/`gp_hf` are present. Thread the flag through
`multi_z.refit.refit_one_param_multi_z` + `scripts/refit_one_param_multi_z.py`
+ `slurm/multi_z_refit.slurm` (`USE_SOBOLEV`/`SOBOLEV_LAMBDA`).

### 3.3 Multi-z derivative gate

Mirror single-z's `build_refit_from_pareto_gated` →
`build_refit_from_pareto_multiz_gated` in `multi_z/refit.py`: iterate Fisher-safe
Pareto rows in best-loss order, reconstruct each 4-input `Refit1DResult`, and
accept the first that passes the gate; raise `ValueError` if none (caller
GP-slice fallback). The gate metric aggregates over the z-range: for each z in
the model's z-grid, compute `equation_param_gradient(refit, z=z)` and
`gp_param_gradient(gp, z=z)` (both already z-aware, Stage 8) and require the
**median over (k, z)** of `|∂eq/∂θ ÷ ∂logP_GP/∂θ − 1| ≤ derivative_tol`. Wire a
per-(param) target-gradient stack into `multi_z.forecast.load_refits` +
`run_three_fisher_multiz`'s reconstruction path.

### 3.4 Config

Reuse `PySRConfig.use_sobolev` / `sobolev_lambda`. `MultiZPipelineConfig` already
has `derivative_tol`. Production config `configs/multi_z/stage10_z2.6-4.2.yaml`
(forecast_only, from_refit, target_space=log, derivative_tol=0.25).

## 4. Evaluation — the diagnostic plots (deliverable)

Run the multi-z Sobolev refit array (z∈[2.6,4.2], λ=5) + forecast, then produce:

1. **Mirage bar chart:** per-param `σ_PySR/σ_GP` (or |log10|) — **Stage 7 multi-z
   (value-loss) vs Stage 10 (Sobolev)** side by side. The headline: does Sobolev
   pull the ratios toward 1?
2. **Gradient-faithfulness bar chart:** per-param median gradient error
   (value-loss vs Sobolev), gate line at 0.25.
3. **Corner plot:** the Stage 10 joint multi-z Fisher (σ_GP / perfect_1D / PySR).
4. **GP-slice summary:** which params still fall back (target: only the
   genuinely-hard ones).

Saved to `results/multi_z_stage10/` and sent to the user.

## 5. Testing

Unit: `sobolev_target_weights_multiz` per-(fidelity,z) + row order (stub GP +
z_per_row payload); multiz refit wiring (stubbed PySR); the multi-z gate metric
over z (synthetic faithful/unfaithful). Gated e2e: multi-z ns Sobolev refit
passes the gate. No-regression: Sobolev off = unchanged.

## 6. Risks

- **Row-order/per-(z,k)-std in the weights** — the make-or-break piece; tested
  hard (mirror Stage 9 Task 2's ordering test, with varying z).
- **λ=5 transfer to multi-z** — re-validate (λ-sweep gate before production).
- **Gate over z** — aggregating across the z-range may be stricter; watch the
  GP-slice count.
- **Cost** — Sobolev ~2× per fit; multi-z fits already ~30–40 min; watch wall.

## 7. Out of scope

- hub-specific investigation (separate; hub may stay GP-slice).
- Retiring the stashed lever-#1.
