# HANDOFF — single-z forecast pipeline

**Last updated:** 2026-06-03
**Branch:** `single_z_forecast_clean` (NOT merged to `main`)
**Tip when this was written:** `185aed5` (Stage 7 code complete: multi_z package)
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
| 7 | multi-z Fisher `F = Σ_z F(z)` (joint, Approach A) | **done** — production run validated; IGM-thermal rank-deficiency lifted |
| 8 | Sobolev derivative-matching loss | not started (informed by in-flight SR-emulator lit review) |

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

## Stage 7 — multi-z Fisher (DONE — production run validated 2026-06-03)

**Production result** (`results/multi_z_stage7/`, full writeup in its
`COMPARISON.md`): the multi-z joint Fisher over z∈[2.6,4.2] on KODIAQ
**lifts the IGM-thermal rank-deficiency** — σ_GP herei 26.7→0.36,
heref 94→1.07, alphaq 235→1.43, hireionz 86→4.34 vs single-z z=3.6.
σ_perfect_1D ≡ σ_GP confirmed (linear+log, gated test). A-vs-B cross-z
diagnostic: joint (A) ~3–5% tighter than the legacy per-z-sum (B) →
Approach A is correct, legacy was biased. 8/11 params got Fisher-safe PySR
equations; ns/bhfeedback/dtau0 → GP-slice (their multi-z 4-input equations
failed the Fisher-safe gate after long retry loops). Mirage persists
(mean |log10(σ_PySR/σ_GP)| ≈ 0.35) — Stage 8's job.

**Env note (2026-06-03):** the central mamba python drifted to numpy 2.x and
the `~/.local` numpy<2 pin vanished, breaking GPy on the nodes. Fixed with a
reproducible **project venv** (`.venv`, pinned `requirements.lock.txt`,
pyproject caps numpy<2/pandas<3, SLURM uses `$REPO/.venv/bin/python`) — see
README. Also: concurrent array tasks contend on the shared `~/.julia_env`
flock (NFS ENOLCK/ESTALE); the SLURM script now staggers Julia init — submit
multi-param arrays with `--array=...%3`.

### Recap of the build (code, Tasks 1–9)

Spec: `docs/superpowers/specs/2026-06-01-multi-z-stage7-fisher-design.md`.
Plan: `docs/superpowers/plans/2026-06-01-multi-z-stage7-fisher.md`.
Built subagent-driven (TDD + per-task spec/quality review), 12 commits
`9aa985b`…`185aed5`.

**Architecture (Approach A — the key finding):** `KSDataLikelihood` is
already multi-z native (`z_min`/`z_max` range, loops `z_blocks` calling
`model.predict(θ,k,z)`, stacks one joint data vector). So the multi-z
Fisher = **one z-spanning `KSDataLikelihood` + the existing
`fisher_matrix`** — almost no new Fisher math. The legacy per-z-sum
(`combine_fisher_phys_arrays`, `scripts/multi_z_aggregate.py`) is kept
only as a diagnostic oracle.

**What landed (`src/priya_forecast/multi_z/`):** `config.py`
(`MultiZPipelineConfig`, z_min/z_max), `combine.py`
(`build_combined_model_multiz`), `refit.py` (`refit_one_param_multi_z`
+ `build_refit_from_pareto_multiz`, CSV + `norm_*.npz` sidecar),
`forecast.py` (`run_three_fisher_multiz` joint + `shared_k_and_z_grid`
guard + `load_refits`), `pipeline.py` (3 modes + `DISPATCH` + `run`).
Plus `MultiZAdditiveTaylorModel.log_space` branch (`refit_taylor.py`),
`MultiZNormalizationSpec.save_npz/load_npz`, scripts
`run_pipeline_multi_z.py` + `refit_one_param_multi_z.py`,
`slurm/multi_z_refit.slurm`, `configs/multi_z/stage7_z2.6-4.2.yaml`.
~20 new multi_z tests pass (2 gated-skip locally).

**Two findings from the build:**
1. **Critical bug caught + fixed** (Task 5): refit reconstruction had been
   normalizing θ with the prior bounds instead of the empirical Sobol
   training range used by `predict_normalized` — would have corrupted
   σ_PySR. The `norm_*.npz` sidecar now persists the empirical
   `x_param_min/max` + `k_min/max`; regression test pins it.
2. **KSData covariance is NOT block-diagonal in z** (its own docstring
   says so). This *confirms Approach A is correct* and means the legacy
   per-z-sum (Approach B / `multi_z_aggregate.py`) was producing biased
   multi-z Fisher forecasts for KSData. The A-vs-B test was reframed from
   an equality assertion into a cross-z-bias **diagnostic**.

Stage 7 unblocks the IGM-thermal params whose single-z Fisher is
rank-deficient (Stage 6's dtau0 outlier at 20.9×).

### Reproduction recipe (cluster — needs Greatlakes + emulator + the `.venv`)

> Build the project venv first (see README); submit refits with `%3`; cavestru0
> was out of billing minutes 2026-06 — this run used `--account=yueyingn0`.

```bash
cd /home/mfho/lya1d_priya_forecast

# Task 10 — one-param calibration (fast, measure real wall time first):
PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia \
PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full \
  python scripts/refit_one_param_multi_z.py --param ns --z-min 3.4 --z-max 3.6 \
    --basedir data/kodiaq_gp --output-dir results/multi_z_stage7_smoke \
    --n-total 64 --niterations 20

# Task 11 — full 11-param refit array (~20-44 CPU-hr, ~30-45 min wall):
sbatch --export=ALL,REPO=$(pwd),BASEDIR=data/kodiaq_gp,\
OUTPUT_DIR=results/multi_z_stage7,Z_MIN=2.6,Z_MAX=4.2 \
--array=0-10 slurm/multi_z_refit.slurm

# Task 11 — production forecast (writes results/multi_z_stage7/{corner.png,
#   scorecard.md, forecast_table.txt, fisher_*.npz}):
PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full \
  python scripts/run_pipeline_multi_z.py --config configs/multi_z/stage7_z2.6-4.2.yaml

# Gated tests (validate perfect_1D==GP + A-vs-B cross-z diagnostic):
PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full \
  RUN_SLOW_FORECAST_ONLY=1 pytest tests/test_multi_z_forecast_joint.py \
  tests/test_multi_z_A_equals_B.py -q
```

Then write `results/multi_z_stage7/COMPARISON.md` (multi-z vs Stage 6
single-z z=3.6: IGM-thermal σ_GP no longer rank-deficient; Mirage delta;
A-vs-B cross-z bias).

## Stage 8 — Sobolev derivative loss (after Stage 7)

Add a derivative-matching term to the PySR loss:
`L = MSE(P_SR, P_GP) + λ · ‖∂_θ logP_SR − ∂_θ logP_GP‖²`.
GP-derived target gradients are computed once and fed to PySR as a
custom Julia loss. This closes the "PySR has the right values but the
wrong derivatives" gap that Stage 6 attenuated but did not eliminate.
Latent risk: requires a `LossFunction` Julia callable; not all PySR
versions support it cleanly.

**Literature-informed levers (2026-06-03 SR-emulator review, full notes in
`docs/SR_EMULATOR_LITERATURE_NOTES.md`).** The syren family (arXiv:2311.15865,
2506.08783, 2510.18749) never validates derivative accuracy — so our Sobolev
loss + a derivative-validation gate are genuine extensions, not reinventions.
Three highest-ROI changes to fold into Stage 8, each attacking Fisher's-Mirage
at a different layer:
1. **Ratio-response target** (target layer): fit `log[P(θ)/P(θ_fid)]` per
   parameter, not raw `log P`. The derivative IS `∂logP/∂θ` (the Fisher
   quantity), so this attacks the Mirage at the SR target and composes with
   the anchor + the Sobolev loss. Biggest single lever.
2. **`aq(x,y)=x/√(1+y²)` operator, drop raw `/`** (operator layer): raw
   division makes poles/spurious curvature near zeros — a mechanical cause of
   derivative-unfaithful equations. `aq` is bounded/smooth. PySR custom binary
   operator; low effort.
3. **Derivative-validation selection gate** (selection layer): reject equations
   on `median|∂logP_SR/∂logP_GP − 1|`, not value RMSE; plus a train/val
   loss-gap reject (syren's overfitting guard).

Deeper architectural flag (tradeoff, not a directive): syren fits ONE joint
multivariate expression; our per-parameter-1D + additive combine drops
cross-terms — relevant to the herei×alphaq coupling
(`memory/headline_findings.md`). Scoped experiment: a joint 2-param refit on
herei–alphaq to measure what the additive combine leaves on the table.

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
