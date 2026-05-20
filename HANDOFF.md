# HANDOFF — single-z forecast pipeline

**Last updated:** 2026-05-20
**Branch:** `single_z_forecast_clean` (NOT merged to `main`; ~57 commits ahead)
**Tip when this was written:** `d4445c5` (Stage 6 + SLURM wiring; results pending commit)

---

## TL;DR — where to resume

The single-z student-facing forecast pipeline is **built and working
end to end** (Stages 1–6, plus the Stage 6 production run on real data).
Stage 6 (`target_space: log`) shipped and a z=3.6 KODIAQ-SQUAD
refit_and_forecast confirmed the structural fix works: **mean
\|log10(σ_PySR/σ_GP)\| dropped 0.615 → 0.366 (40% reduction); deep-Mirage
params (< 0.2× ratio) went from 3 → 0**. Full comparison in
`results/single_z_stage6_log/COMPARISON.md`.

**Next:** Stage 7 (multi-z Fisher `F = Σ_z F(z)`) and Stage 8 (Sobolev
derivative-matching loss). 7 of 11 params still have sub-1 PySR/GP
ratios in log mode (the *severity* dropped, not the headcount) — Stage 8
addresses what log(P) alone cannot.

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
**332 tests pass** (gated emulator/PySR tests skip without lyaemu).

## Status by stage

| Stage | What | State |
|-------|------|-------|
| 1 | foundations: `regen_1pvar`, `combine.py` | done |
| 2 | `forecast_only` | done |
| 3 | `refit_and_forecast` (+ seed-retry) | done |
| 4 | `run_batch` + `aggregate_z` | done |
| 5 | `ONBOARDING.md` rewrite + 3 notebooks | done |
| 6 | **log(P) SR target + log-space combine** | **planned, not implemented** |
| 7 | multi-z Fisher `F = Sum_z F(z)` | high-level only |
| 8 | Sobolev derivative-matching loss | not started |

## The key scientific findings (why Stages 6-8 exist)

From the real z=3.6 `refit_and_forecast` run on KODIAQ-SQUAD:

1. **sigma_perfect_1D == sigma_GP, exactly.** The additive combine
   reproduces the GP's first derivatives, and Fisher is first-order. The
   "3-sigma ladder" collapses to sigma_GP vs sigma_PySR.
2. **Single-z all-11-param Fisher is rank-deficient** — sigma_GP explodes
   for the IGM-thermal parameters. Expected; needs multi-z.
3. **sigma_PySR is derivative-unfaithful** — sigma_PySR/sigma_GP spanned
   0.09x-27x, including physically impossible sub-1 ratios ("Fisher's
   Mirage", arXiv:2406.06067). Seed-retry + a well-conditioned 3-param
   subset removed the NaNs and rank-deficiency but the ratios stayed
   0.59-2.5 — the derivative-faithfulness problem is **structural**.

**Research verdict (web search, full report in `memory/active_work.md`):**
- **#1 — fit `log(P)` not `P`** → Stage 6.
- **#2 — multi-z forecast `F = Sum_z F(z)`** → Stage 7.
- **#3 — Sobolev loss** (penalize `[d_theta logP_SR - d_theta logP_GP]^2`)
  → Stage 8.

## Stage 6 — ready to execute

Spec: `docs/superpowers/specs/2026-05-19-single-z-stage6-log-target-design.md`
(reviewed by a technical-review agent, then revised — 6 fixes applied).
Plan: `docs/superpowers/plans/2026-05-19-single-z-stage6-log-target.md`
— **8 tasks, all code written out**, designed for subagent-driven
execution. Adds a `target_space: linear | log` config flag; in `log`
mode the SR equation trains on normalized `log(P)` and the combine is
additive-in-log. The linear path is untouched (default) so the two are
comparable. `TaskList` IDs 19-26 are the 8 Stage-6 tasks (none started).

**Resume:** invoke `superpowers:subagent-driven-development` on that
plan, dispatch Task 1. After Stage 6 lands, re-run the z=3.6
`refit_and_forecast` with `target_space: log` and compare
sigma_PySR/sigma_GP to the linear baseline.

## How to run things

- **Tests:** `PYTHONPATH=src pytest tests/ -q`
- **Slow/emulator tests:** need `lyaemu` importable —
  `PYTHONPATH=src:/home/mfho/lya_emulator_full` — and `data/kodiaq_gp/`.
  Gated by env vars: `RUN_SLOW_GP_ONLY`, `RUN_SLOW_FORECAST_ONLY`,
  `RUN_SLOW_REFIT`, `RUN_SLOW_REGEN_1PVAR`.
- **PySR / Julia gotcha:** any PySR run MUST set
  `PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env` and
  `JULIA_DEPOT_PATH=$HOME/.julia` — otherwise juliapkg tries a read-only
  system path and fails. The SLURM scripts set these; ad-hoc runs must too.
- **Regenerated training data:** `data/single_z_1pvar/` (gitignored, 22
  HDF5s) — produced by `scripts/regen_1pvar.py`; already generated.
- **Forecast results:** `results/single_z_stage{1,2,3,3_subset}/`
  (gitignored).

## Pointers

- Design specs + plans: `docs/superpowers/specs/`, `docs/superpowers/plans/`.
- Memory index: `~/.claude/projects/-home-mfho-lya1d-priya-forecast/memory/MEMORY.md`
  — `active_work.md` has the full status + research verdict;
  `student_pysr_contract.md`, `igm_thermal_z_dependence.md`,
  `headline_findings.md` are the relevant science memories.
- Branch is **not merged** — the merge/PR decision is the user's.
- This file replaced an older HANDOFF.md that described the pre-single-z
  multi-z Phase 0-7 work; that history is in git and in `docs/PAPER_NOTES.md`.
