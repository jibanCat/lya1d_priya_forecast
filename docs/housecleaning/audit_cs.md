# CS / RSE audit — reproducibility, packaging, dead code

Auditor: research-software-engineering reviewer
Date: 2026-06-09
Repo: `/home/mfho/lya1d_priya_forecast` (branch `stage10-multiz-sobolev`)
Scope: structure map, clean-install README outline, legacy/dead-code removal,
reproducibility gaps. Verified against a live test run (412 passed / 14 skipped,
55 s) and a live emulator-free figure regen (3 figures from committed sidecars).

---

## 0. Structure map (verified)

- `src/priya_forecast/` (61 tracked files): the library. Single package, editable
  install via `pyproject.toml` (`where=["src"]`). **No fully-dead modules** — every
  module is imported by at least one of src/scripts/tests (min self-excluded
  reference count = 2). Subpackages: `models/` (gp/pysr/base/normalization),
  `single_z/`, `multi_z/`, `diagnostics/`, `_vendored/` (eBOSS DR14 + KODIAQ-SQUAD
  data files, intentionally vendored). Current-headline modules:
  `grad_faith_io.py`, `pareto_diag.py`, `sobolev_loss.py`, `derivative_gate.py`.
- `scripts/` (35 drivers + `scripts/smoke/`): mix of current diagnostic drivers and
  superseded phase drivers. See LEGACY table.
- `tests/` (60 files): healthy. Fast suite green; slow/GP/PySR tests correctly gated
  behind `RUN_SLOW_*` env vars + `lyaemu` availability.
- `configs/`: `single_z/`, `multi_z/`, `eqns/`, `hpo/`, plus `default.yaml`,
  `diagnostic.yaml`.
- `slurm/` (5): cluster array jobs; `$REPO`-parameterized but `lya_emulator_full`
  PYTHONPATH is hard-coded.
- `notebooks/` (3): `01_gp_only`, `02_forecast_only`, `03_refit_and_forecast` —
  pedagogy for the *forecast* era, not the diagnostic.
- `docs/`: onboarding + per-phase plans/specs under `docs/superpowers/`; the current
  source of truth is `docs/PARETO_FAITHFULNESS_WALKTHROUGH.md`.
- Packaging: `pyproject.toml` (numpy<2 / pandas<3 caps, optional-deps groups,
  `priya-forecast` console entry point) + `requirements.lock.txt` (full pinned
  stack, numpy 1.26.4 / GPy 1.13.2 / pysr 1.5.10) + project `.venv`.
- Root clutter: 132 untracked `slurm-*.out` (gitignored), `outputs/` 126 MB
  (gitignored PySR/Julia scratch), `src/priya_forecast.egg-info/` (untracked).

---

## 1. README outline a clean install + usage MUST contain

The current `README.md` (dated 2026-05-07) documents the **superseded** σ_PySR/σ_GP
forecast loop, not the current **derivative-faithfulness diagnostic**. A rewrite
should be structured as:

1. **What this is (1 paragraph).** Distills the PRIYA multi-fidelity GP Lyα-P1D
   emulator into per-parameter PySR equations and runs a *derivative-faithfulness
   diagnostic* (does ∂P/∂θ match the GP — the only thing a Fisher forecast uses).
   Point at `docs/PARETO_FAITHFULNESS_WALKTHROUGH.md` and `HANDOFF.md` as truth.
2. **Two-tier usage up front.** (a) *Emulator-free*: regenerate the headline
   diagnostic figures from committed sidecars — no GP, no Julia, no cluster data.
   (b) *Full*: retrain/re-gate, which needs the GP + Julia + cluster data.
3. **Clean install (tier-a, the common case).**
   - `python3.11 -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.lock.txt` then `pip install -e . --no-deps`
     (exact, reproducible) **or** `pip install -e ".[forecast,pysr,gp,dev]"` (flexible).
   - **Why numpy<2 / pandas<3**: GPy 1.13.2 cython ABI; already in caps + lock.
   - `export PYTHONPATH=$PWD/src` and run
     `python scripts/make_diagnostic_figs.py --out-dir results/single_z_stage_pareto_diag`
     → 3 figures. **This is the whole story for a reviewer reproducing the paper.**
4. **Full install (tier-b) prerequisites — each is a hard external dependency:**
   - **lya_emulator (sbird)** — supplies the GP; NOT on PyPI. Clone
     `https://github.com/sbird/lya_emulator`, add to `PYTHONPATH`. Document that the
     repo currently hard-codes `/home/mfho/student_projects/lya_emulator_full`.
   - **GP basedir** — `InferenceLyaData/Emulator_Files` (`emulator_params.json` +
     GP pickles). Hard-coded default in `models/gp_model.py` / `config.py`; document
     the override (`basedir=` / `gp_emulator_basedir:` in `configs/default.yaml`).
   - **Julia/PySR** — `export PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env` and
     `export JULIA_DEPOT_PATH=$HOME/.julia` (CLI auto-`setdefault`s these; scripts/
     SLURM do not).
   - **Regenerated data**: `data/kodiaq_gp/` (via `scripts/prep_kodiaq_gp.py`) and
     `data/single_z_1pvar/` (via `scripts/regen_1pvar.py`) — both gitignored.
5. **Test gate.** `PYTHONPATH=src pytest tests/ -q -k "not slow"`
   (expected: ~412 pass, ~14 skip without `lyaemu`/cluster data). Document the
   `RUN_SLOW_*` opt-in env vars for the GP/PySR-backed tests.
6. **Entry points.** Console script `priya-forecast` (forecast/multid/coupling/hpo)
   — note this is the *forecast-era* CLI and does **not** cover the diagnostic; the
   diagnostic is reached only via `scripts/make_diagnostic_figs.py` /
   `scripts/eval_grad_faithfulness.py` / `scripts/make_grad_faith_sidecars.sh`.
7. **Repo map + doc index** (one line each: ONBOARDING, WALKTHROUGH, HANDOFF).
8. **Retire `README_v2.md`** into a `docs/REPRODUCE_phase15_phase2.md` archive (it is
   a Phase 1.5/2 reproducer) so there is exactly one top-level README.

---

## 2. LEGACY / dead / superseded — removal table

`confidence` = how safe to delete (high = no live reference, superseded by name in
HANDOFF; med = era-stale but possibly cited by paper/history; low = needs owner call).

| Path | Why | Confidence |
|---|---|---|
| `slurm-multid_pysr-49350027.out`, `slurm-multid_pysr-49355289.out` | Two SLURM stdout logs committed before `slurm-*.out` was gitignored; pure run scratch, no consumer. | **high** |
| `.claude/.nfs000000152d21863000002c61` | NFS silly-rename turd (already shows `D` in git status); delete + commit removal. | **high** |
| `.claude/scheduled_tasks.lock` | Machine-local lock file, should never have been tracked. | **high** |
| `src/priya_forecast.egg-info/` | Untracked build artifact (not in git, matched by `*.egg-info/` ignore); `rm -rf`. | **high** |
| `scripts/compare_eqn_sets.py` | Unimplemented stub: `main()` raises `SystemExit("...not yet implemented (phase 7)")`; phase 7 never happened. Zero references. | **high** |
| `docs/.ipynb_checkpoints/`, `docs/superpowers/**/.ipynb_checkpoints/`, `configs/eqns/.ipynb_checkpoints/`, root `.ipynb_checkpoints/` | Jupyter checkpoint copies (untracked); add `**/.ipynb_checkpoints/` to `.gitignore` and `rm -rf`. | **high** |
| `outputs/` (126 MB) | PySR/Julia per-run scratch, gitignored; safe to wipe to reclaim space (regenerated on next run). | **high** |
| 132 untracked root `slurm-*.out` | Cluster stdout clutter, gitignored; `rm slurm-*.out`. | **high** |
| `scripts/compare_pysr_winners.py` | Plots the `val_mse / fisher_aware / sigma_targeted` PySR-HPO comparison (PYSR_HYPOTHESIS era); not referenced by HANDOFF/walkthrough/slurm. | **med** |
| `scripts/run_pysr_hypothesis.py`, `src/priya_forecast/pysr_hypothesis.py`, `docs/PYSR_HYPOTHESIS.md`, `results/pysr_hypothesis/`, `docs/figures/pysr_hypothesis/` | The "why does PySR underperform the GP" hypothesis sweep — diagnosed and folded into the Sobolev-loss result; superseded by the faithfulness diagnostic. Still has a passing test (`test_pysr_hypothesis.py`) so removal must drop the test too. | **med** |
| `scripts/run_residual_pysr.py`, `src/priya_forecast/refit_residual.py` | Residual-PySR path; MEMORY records IGM thermal params need multi-z not residual-PySR — abandoned approach. | **med** |
| `scripts/forecast_original_design.py`, `scripts/regen_sample_figures.py`, `scripts/replot.py`, `scripts/port_pysr_equations.py` | One-off forecast-era figure/port utilities; not wired into any current pipeline or doc. | **med** |
| `results/refit_phase2_production*`, `results/refit_optionC_*phase1_5*`, `results/holdout_multid_phase*`, `results/closure_at_simdat_*`, `results/published_scorecard/`, `results/single_z_stage8/`, `results/multi_z_stage7/`, `results/refit_multid_z2.6-4.2*`, `results/smoke_ap_log_target/` | Phase 1.5 / Phase 2 / Stage 7-8 committed result trees. Zero references from HANDOFF / walkthrough / README; superseded by the diagnostic. Largest tracked-bloat source (≈220 of 310 `results/` files). Archive to a `results-archive` tag/branch then delete from working tree. | **med** |
| `LOCAL_PAPER_HANDOFF.md`, `README_v2.md` | Era-specific handoffs (laptop paper-writing; Phase 1.5/2 reproducer). Consolidate into `docs/` archive; keeps root to one README + HANDOFF. | **med** |
| `docs/AP_REMEDIATION_PLAN.md`, `docs/PAIR_FIT_PLAN.md`, `docs/PYSR_PERFORMANCE.md` | Phase-2/3 design docs for the σ-ratio/pair-coupling work the diagnostic pivot retired. Move under `docs/superpowers/` archive, don't delete (paper provenance). | **low** |
| `scripts/refit_residual.py` consumers + Taylor/`refit_taylor.py` | `refit_taylor` is still imported by tests; keep unless the Taylor-combine path is formally dropped — owner call. | **low** |
| `notebooks/0{1,2,3}_*.ipynb` | Forecast-era tutorials; still correct for the GP/forecast API but not the diagnostic. Keep only if README points students at them; otherwise stale. | **low** |
| `scripts/run_pipeline.py`, `scripts/run_pipeline_multi_z.py`, `scripts/run_batch.py`, `scripts/aggregate_z.py` | Single-z/multi-z stage orchestration with live tests (`test_run_batch`, `test_aggregate_z`). Tested → keep; flag only if stage pipeline is retired. | **low** |

---

## 3. Reproducibility gaps (concrete)

1. **No CLI entry point for the headline science.** `pyproject.toml`'s
   `priya-forecast` console script and `src/priya_forecast/cli.py` expose only
   `forecast / multid / coupling / hpo` (the superseded forecast era). The current
   diagnostic is reachable only by running `scripts/make_diagnostic_figs.py`,
   `scripts/eval_grad_faithfulness.py`, and the `scripts/make_grad_faith_sidecars.sh`
   shell wrapper directly. Add a `priya-forecast diagnose-figs` / `grad-faith`
   subcommand (and note the existing `diagnose` subcommand is itself a stub that just
   raises `SystemExit`).
2. **`scripts/make_diagnostic_figs.py` is cwd-locked.** Input dirs are hard-coded as
   *relative* strings (`VALUE = "results/single_z_stage6_log/refit/z3.6"`, etc.), so
   the script `FileNotFoundError`s unless invoked from the repo root (verified from
   `/tmp`). Resolve these against `Path(__file__).resolve().parents[1]` or expose
   `--value-dir/--sobolev-dir/--budget-dir`.
3. **Hard-coded absolute `/home/mfho/...` paths in 30+ files.** Two flavors:
   (a) `lya_emulator_full` PYTHONPATH baked into every GP-backed script docstring
   *and* runtime `_LYAEMU = Path("/home/mfho/student_projects/lya_emulator_full")`
   (`scripts/*.py`, `slurm/*.slurm`); (b) GP basedir default
   `/home/mfho/student_projects/InferenceLyaData/Emulator_Files` in
   `models/gp_model.py:37`, `config.py:82`, `configs/default.yaml:14`, and the 1pvar
   dir in `refit_1d_pysr.py:51`. The GP basedir is at least *overridable*
   (constructor `basedir=` / config), but has no env-var fallback. Recommend an
   `LYA_EMULATOR_HOME` / `PRIYA_GP_BASEDIR` env-var convention with these as
   last-resort defaults; at minimum document the clone+export in the README.
4. **`slurm/*.slurm` hard-code the emulator PYTHONPATH** (`$REPO` is parameterized,
   `lya_emulator_full` is not) → jobs are non-portable off this user's account.
5. **GP/PySR-backed surface is structurally untestable in CI.** 14 tests skip without
   `lyaemu` + `data/kodiaq_gp/` + Julia; this is correctly gated but means the GP
   adapter, KSData likelihood, and full single-z/multi-z pipelines have **no
   executable coverage in a clean checkout**. The emulator-free sidecar path is the
   only end-to-end-reproducible surface — lean on it and document the gap.
6. **Two committed-then-gitignored SLURM logs** (`slurm-multid_pysr-*.out`) violate
   the repo's own `slurm-*.out` ignore rule — symptom of artifact discipline drift;
   `git rm --cached` them.
7. **README/pyproject description lag the science.** README headline is the
   σ_PySR/σ_GP forecast (explicitly disavowed in HANDOFF as a forced-Jacobian
   confound); a reviewer following README reproduces the retired claim, not the
   diagnostic. Highest-leverage doc fix.
8. **No top-level "reproduce the paper in one command" target.** The emulator-free
   path works (verified) but is buried in HANDOFF; promote it to a Makefile target or
   README tier-a so it is the first thing a referee runs.

---

## Appendix — verification evidence

- Fast tests: `PYTHONPATH=src pytest tests/ -q -k "not slow"` → **412 passed, 14
  skipped, 16 warnings, 55.9 s**.
- Emulator-free figures: `scripts/make_diagnostic_figs.py --out-dir /tmp/diag_test`
  → **3 figures (png+pdf)** from 67 committed `grad_faith_*.csv` sidecars, no GP.
- cwd-lock: same script from `/tmp` → `FileNotFoundError` on relative `results/...`.
- Module liveness: every `src/priya_forecast/*` module has ≥1 non-self importer.
- `scripts/compare_eqn_sets.py` → `main()` raises `SystemExit(... phase 7)`.
