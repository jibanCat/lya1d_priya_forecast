# Stage 10 — multi-z Sobolev Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Mirror the Sobolev loss into the multi-z refit so the well-conditioned multi-z forecast shows the Mirage reduction (the interpretable σ single-z couldn't give). Diagnostic plots are the deliverable.

**Architecture:** Add `sobolev_target_weights_multiz` (per-(fidelity,z) normalized target gradient) + thread `use_sobolev` through `refit_1d_multiz_for_param` + the multi-z refit driver/CLI/SLURM. The existing `multi_z` forecast (Stage 7) reads the resulting equations. Gate (multi-z) is a Phase-2 add-on if needed.

**Tech Stack:** `.venv` (numpy<2), PySR/Julia. `PYTHONPATH=src .venv/bin/python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-06-05-stage10-multiz-sobolev-design.md`.

---

## Task 1: `sobolev_target_weights_multiz` (the make-or-break piece)

**Files:** Modify `src/priya_forecast/sobolev_loss.py`; Test `tests/test_stage10_weights.py`.

Mirror `sobolev_target_weights` but: each Sobol row has its own `z` (`payload["z_per_row"]`), the GP gradient is evaluated at `(θ_r, k, z_r)` per fidelity, and the std is per-(z,k) (`MultiZNormalizationSpec.std_flux[z_index(z_r), :]`). Row order matches `_build_training_matrix_multiz` (LF then HF, point-major/k-minor). Boundary-clamp the perturbation.

- [ ] Write the failing test (stub GP with z-dependent gradient, 2 points at 2 z's, verify per-(fidelity,z) values + LF-then-HF order + that a boundary point doesn't exceed range).
- [ ] Implement `sobolev_target_weights_multiz(*, payload, param_idx, gp_lf, gp_hf, norm, z_min, z_max, x_param_min, x_param_max, h=1e-3)` (takes the `MultiZNormalizationSpec` `norm` for per-(z,k) std via `norm._z_index`). Reuse a clamped per-point gradient helper.
- [ ] Run tests; commit.

## Task 2: wire Sobolev into `refit_1d_multiz_for_param`

**Files:** Modify `src/priya_forecast/refit_1d_pysr.py` (`refit_1d_multiz_for_param`); Test `tests/test_stage10_refit_wiring.py`.

Add `use_sobolev`, `sobolev_lambda`, `sobolev_h` kwargs. When on: guard `gp_lf`/`gp_hf` present; build weights via `sobolev_target_weights_multiz` (using `payload`, `param_idx`, `norm`, `ranges["x_param_min/max"]`, `z_min`, `z_max`); `args["loss_function"] = make_sobolev_loss(λ,h)`, drop `elementwise_loss`; `model.fit(X_act, Y_act.reshape(-1,1), weights=weights)`. Off-path byte-identical.

- [ ] TDD (stub PySR + `_generate_1pvar_multiz_inline`); run; commit.

## Task 3: thread the flag (multi-z driver + CLI + SLURM)

**Files:** `src/priya_forecast/multi_z/refit.py` (`refit_one_param_multi_z`), `scripts/refit_one_param_multi_z.py`, `slurm/multi_z_refit.slurm`; Test `tests/test_stage10_thread.py`.

Mirror Stage 9 Task 5: pass `cfg.pysr.use_sobolev`/`sobolev_lambda` into `refit_1d_multiz_for_param`; add `--use-sobolev`/`--sobolev-lambda` to the script; add `USE_SOBOLEV`/`SOBOLEV_LAMBDA` passthrough to the SLURM. TDD; commit.

## Task 4: λ validation + production run + plots (HPC)

- [ ] Validate λ=5 transfers to multi-z: refit ns multi-z with `use_sobolev` locally; confirm gate-faithful (mirror Stage 9 validation, over the z-range).
- [ ] Submit `--export=...,USE_SOBOLEV=1,SOBOLEV_LAMBDA=5.0 --array=0-10%3 slurm/multi_z_refit.slurm` (OUTPUT_DIR=results/multi_z_stage10, Z_MIN=2.6, Z_MAX=4.2, account=yueyingn0).
- [ ] Run the multi-z forecast (`scripts/run_pipeline_multi_z.py --config configs/multi_z/stage10_z2.6-4.2.yaml`).
- [ ] **Produce diagnostic plots** (`scripts/plot_stage10_diagnostics.py` → `results/multi_z_stage10/`):
  1. per-param σ_PySR/σ_GP: Stage 7 (value-loss) vs Stage 10 (Sobolev), bar chart;
  2. per-param gradient error (value vs Sobolev), gate line 0.25;
  3. the Stage 10 joint corner plot;
  4. GP-slice summary.
- [ ] Write `results/multi_z_stage10/COMPARISON.md`; update HANDOFF. Commit.

## Phase 2 (only if Sobolev equations need filtering): multi-z gate

`build_refit_from_pareto_multiz_gated` + per-(param,z) target gradients wired into `multi_z.forecast.load_refits`, gate metric median over (k,z). Mirror single-z `build_refit_from_pareto_gated`.

## Self-review
- Task 1 row-order + per-(z,k) std is the critical correctness property → full review.
- λ transfer to multi-z is the live unknown → Task 4 validation gate before production.
- Gate deferred to Phase 2 (the existing forecast reads Sobolev equations directly first).
