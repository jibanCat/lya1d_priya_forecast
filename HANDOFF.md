# HANDOFF — single-z forecast pipeline

**Last updated:** 2026-05-20
**Branch:** `single_z_forecast_clean` (NOT merged to `main`; 57 commits ahead)
**Tip when this was written:** `fe3112b` (Stage 6 results + COMPARISON.md)
**Open PR:** https://github.com/jibanCat/lya1d_priya_forecast/pull/3

---

## TL;DR — where to resume

The single-z student-facing forecast pipeline is **built and working
end to end** through Stage 6, including a real-data production run.
Stage 6 (`target_space: log`) shipped and the z=3.6 KODIAQ-SQUAD
`refit_and_forecast` confirmed the structural fix:

- **mean |log10(σ_PySR/σ_GP)| dropped 0.615 → 0.366 (40% reduction)**
- **deep-Mirage params (< 0.2× ratio) went from 3 → 0**
- 8 of 11 params closer to Fisher-faithful; 3 regressed mildly (tau0, ns, herei)
- sub-1 Mirage count unchanged at 7/11 — log(P) attenuates *severity*, not headcount

Full comparison in `results/single_z_stage6_log/COMPARISON.md`.
Corner plot at `results/single_z_stage6_log/corner.png` (untracked but reproducible).

**Next:** Stage 7 (multi-z Fisher `F = Σ_z F(z)`) and Stage 8 (Sobolev
derivative-matching loss). Stage 8 addresses what log(P) alone cannot.

---

## What the pipeline is

A single-redshift-bin Lyman-α P1D Fisher forecast over the 11 PRIYA
parameters. One YAML config, one CLI (`scripts/run_pipeline.py`), three
modes:

- `gp_only` — Fisher on the GP emulator (σ_GP).
- `forecast_only` — load PySR Pareto CSVs → equations → σ_GP /
  σ_perfect_1D / σ_PySR + corner plot.
- `refit_and_forecast` — run single-z PySR refits per parameter → emit
  Pareto CSVs → forecast.

Student procedure: `regen_1pvar.py` (regenerate LF/HF training data from
the emulator) → refit → forecast → `aggregate_z.py` (across-z view).
Two real measured covariances: KODIAQ-SQUAD (`KSDataLikelihood`,
Karaçaylı+2021) and eBOSS DR14 (`load_eboss`) — selected by
`data.source`. No synthetic covariances.

**Code:** `src/priya_forecast/single_z/{config,pipeline,forecast,refit,
combine,training_data}.py`; scripts `regen_1pvar.py`, `run_pipeline.py`,
`run_batch.py`, `aggregate_z.py`, `refit_one_param_single_z.py`;
`slurm/single_z_refit.slurm`; `docs/ONBOARDING.md` + `notebooks/01-03`.
**~340 tests pass, 10 gated skip, 1 pre-existing unrelated failure**
(`test_h6_explicit_cross_terms_*` in `test_pysr_hypothesis.py` — predates
Stage 6, numpy-scalar TypeError).

## Status by stage

| Stage | What | State |
|-------|------|-------|
| 1 | foundations: `regen_1pvar`, `combine.py` | done |
| 2 | `forecast_only` | done |
| 3 | `refit_and_forecast` (+ seed-retry) | done |
| 4 | `run_batch` + `aggregate_z` | done |
| 5 | `ONBOARDING.md` rewrite + 3 notebooks | done |
| 6 | **log(P) SR target + log-space combine** | **done** (15 commits, production run validated) |
| 7 | multi-z Fisher `F = Σ_z F(z)` | high-level only |
| 8 | Sobolev derivative-matching loss | not started |

## The key scientific findings

From the real z=3.6 `refit_and_forecast` runs on KODIAQ-SQUAD:

1. **σ_perfect_1D ≡ σ_GP, exactly.** The additive combine reproduces the
   GP's first derivatives, and Fisher is first-order. The "3-σ ladder"
   collapses to σ_GP vs σ_PySR. Confirmed in both linear and log space.
2. **Single-z all-11-param Fisher is rank-deficient** — σ_GP explodes for
   the IGM-thermal parameters. Expected; needs multi-z (Stage 7).
3. **σ_PySR was derivative-unfaithful in linear mode** — σ_PySR/σ_GP
   spanned 0.09×–27×, including physically impossible sub-1 ratios
   ("Fisher's Mirage", arXiv:2406.06067). **Stage 6 (log target)
   attenuates the severity (max sub-1 deviation now 0.35× vs 0.09×
   before), but does not eliminate it — Stage 8 is required.**

**Research verdict (literature review captured in `memory/active_work.md`):**
- **#1 — fit `log(P)` not `P`** → ✅ Stage 6 done.
- **#2 — multi-z forecast `F = Σ_z F(z)`** → Stage 7.
- **#3 — Sobolev loss** (penalize `[∂_θ logP_SR − ∂_θ logP_GP]²`) → Stage 8.

## Stage 6 — done (recap)

Spec: `docs/superpowers/specs/2026-05-19-single-z-stage6-log-target-design.md`.
Plan: `docs/superpowers/plans/2026-05-19-single-z-stage6-log-target.md`.
Implementation: 15 commits across 8 tasks (`07aac6b` … `fe3112b`),
TDD-driven via `superpowers:subagent-driven-development`; each task got
spec-compliance + code-quality review with a fix loop, plus a
whole-Stage opus review (APPROVED FOR MERGE).

What landed:
- `target_space: linear | log` config flag wired end-to-end through
  config → `compute_local_normalization` → `_build_training_matrix` →
  `Refit1DResult.predict_log` → `AdditiveTaylorModel` log-space combine →
  `run_three_fisher`.
- Positivity guards with clear `ValueError` on every log-space code path.
- `scripts/refit_one_param_single_z.py` got `--target-space {linear,log}`;
  `slurm/single_z_refit.slurm` got matching `TARGET_SPACE` env var.
- New gated end-to-end test `test_refit_and_forecast_log_space_end_to_end`
  and the central log-space sanity test
  `test_run_three_fisher_log_space_perfect_equals_gp` (asserts σ_perfect_1D
  ≈ σ_GP at rtol=1e-3 in log space too).
- Production YAML: `configs/single_z/stage6_log_z3.6.yaml`.
- Production results: `results/single_z_stage6_log/{corner.png,
  scorecard.md, COMPARISON.md, fisher_*.npz, forecast_table.txt,
  refit/z3.6/pareto_*.csv}`. corner.png + .npz are untracked
  (reproducible); COMPARISON.md is committed.

## Stage 7 — multi-z Fisher (next)

High-level only; no design spec yet. The pattern is clear from Stage 6:
- Mirror the Stage 6 `log_space` threading through the multi-z functions
  (`MultiZAdditiveTaylorModel`, `_build_training_matrix_multiz`,
  `compute_local_normalization_multiz`, `refit_1d_multiz_for_param`).
- Build a `MultiZAdditiveTaylorModel` log-space branch with per-z
  `_log_p_gp_fid[z]` and `_eq_at_fid_logpf[(pname, z)]` caches.
- The Fisher sum `F = Σ_z F(z)` is mathematically straightforward (chain
  rule keeps gradients at fid equal in linear and log space), so the
  cross-z sum should "just work" — no shape mismatches anticipated.
- Stage 7 unblocks the IGM-thermal params whose single-z Fisher is
  rank-deficient (the "max |log10|" outlier in Stage 6 comparison is
  dtau0 at 20.9×, characteristic of weakly-constrained directions).

## Stage 8 — Sobolev derivative loss (after Stage 7)

Add a derivative-matching term to the PySR loss:
`L = MSE(P_SR, P_GP) + λ · ‖∂_θ logP_SR − ∂_θ logP_GP‖²`.
GP-derived target gradients are computed once and fed to PySR as a
custom Julia loss. This closes the "PySR has the right values but the
wrong derivatives" gap that Stage 6 attenuated but did not eliminate.
Latent risk: requires a `LossFunction` Julia callable; not all PySR
versions support it cleanly.

## How to run things

- **Fast tests:** `PYTHONPATH=src pytest tests/ -q` — ~340 pass, 10 skip.
- **Slow/emulator tests:** need `lyaemu` importable —
  `PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full` — and
  `data/kodiaq_gp/`. Gated by env vars: `RUN_SLOW_GP_ONLY`,
  `RUN_SLOW_FORECAST_ONLY`, `RUN_SLOW_REFIT`, `RUN_SLOW_REGEN_1PVAR`.
- **PySR / Julia gotcha:** any PySR run MUST set
  `PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env` and
  `JULIA_DEPOT_PATH=$HOME/.julia` — otherwise juliapkg tries a read-only
  system path and fails. The SLURM scripts set these; ad-hoc runs must too.
- **Regenerated training data:** `data/single_z_1pvar/` (gitignored,
  22 HDF5s) — produced by `scripts/regen_1pvar.py`; already generated.
- **Forecast results:** `results/single_z_stage{1,2,3,3_subset,6_log}/`
  (untracked; not gitignored, just not committed).

### Stage 6 reproduction recipe

```bash
# 11-task SLURM array, ~5 min/task wall:
sbatch --export=ALL,REPO=$(pwd),BASEDIR=data/kodiaq_gp,\
       OUTPUT_DIR=results/single_z_stage6_log,Z=3.6,TARGET_SPACE=log \
       --array=0-10 slurm/single_z_refit.slurm

# Forecast (~4 min, emulator load dominates):
PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full \
  python scripts/run_pipeline.py --config configs/single_z/stage6_log_z3.6.yaml
```

## Environment (gotchas discovered 2026-05-20)

- **numpy must be <2 on this machine.** Greatlakes' centrally-installed
  numpy is now 2.x; the user previously installed numpy 2.4.6 to
  `~/.local/`. GPy 1.13.2's compiled cython extensions are built against
  numpy 1.x's `numpy.dtype` (96 bytes); numpy 2.x shrunk it to 88 bytes,
  producing `ValueError: numpy.dtype size changed`. Fix:
  `pip install --user "numpy<2"` (1.26.4 verified working).
- **Five one-line legacy-numpy-API patches** were applied to
  `~/.local/lib/python3.11/site-packages/`:
  - `paramz/model.py:36` and `paramz/core/index_operations.py:32`
  - `GPy/util/pca.py:13`, `GPy/core/sparse_gp_mpi.py:6`,
    `GPy/models/ss_mrd.py:16`
  Each replaces a removed numpy-internal import
  (`numpy.linalg.linalg.LinAlgError`, `numpy.lib.function_base.vectorize`)
  with its public-API equivalent. These survive on numpy<2 too (they're
  upstream-correct), so the downgrade + patches are belt-and-braces.
- **paramz 0.9.6 + GPy 1.13.2** is the working pinning.

## Test status

- Linear path of every Stage 6 test: byte-equivalent to pre-Stage-6.
- Stage 6 unit tests: positivity guards, log-mean normalization,
  exp/log inverse consistency, anchor-identity at θ=fid, multi_d mode
  rejection.
- Stage 6 integration test: σ_perfect_1D ≈ σ_GP in log-space (rtol=1e-3).
- Pre-existing unrelated failure:
  `tests/test_pysr_hypothesis.py::test_h6_explicit_cross_terms_help_modestly_on_non_separable_truth`
  — numpy scalar TypeError, predates Stage 6 (file last touched at commit
  `7b30a3b`).

## Pointers

- Design specs + plans: `docs/superpowers/specs/`,
  `docs/superpowers/plans/`.
- Memory index:
  `~/.claude/projects/-home-mfho-lya1d-priya-forecast/memory/MEMORY.md`
  — `active_work.md` has the full status + research verdict;
  `student_pysr_contract.md`, `igm_thermal_z_dependence.md`,
  `headline_findings.md` are the relevant science memories.
- Stage 6 PR: https://github.com/jibanCat/lya1d_priya_forecast/pull/3.
- This file replaced an older HANDOFF.md that described the pre-single-z
  multi-z Phase 0-7 work; that history is in git and in
  `docs/PAPER_NOTES.md`.
