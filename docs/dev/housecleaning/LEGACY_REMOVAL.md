# Legacy removal log — conservative cleanup

Date: 2026-06-09
Branch: `stage10-multiz-sobolev`
Source: LEGACY-REMOVAL table in `docs/housecleaning/audit_cs.md` §2 (cross-checked
against `docs/housecleaning/audit_lya.md`).
Policy: remove ONLY high-confidence dead/superseded **tracked code/scratch** with a
verified zero-importer check; everything medium/low confidence, or anything under the
protected trees (`results/`, `data/`, `configs/`, `slurm/`, and the current diagnostic
code), is written as a PROPOSED-FOR-HUMAN entry instead of being deleted.

Test gate: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q -k "not slow"`
(project `.venv`, numpy 1.26.4; Julia env vars `PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env`,
`JULIA_DEPOT_PATH=$HOME/.julia`).

- Baseline (before removals): **412 passed, 14 skipped**.
- After removals: **412 passed, 14 skipped** — unchanged. No file needed reverting.

No `git commit` was made (per instructions); removals are staged in the index only.

---

## REMOVED (via `git rm`)

| Path | Why | Importer / reference check |
|---|---|---|
| `scripts/compare_eqn_sets.py` | Unimplemented stub from a phase that never happened: `main()` only `raise SystemExit("compare_eqn_sets.py is not yet implemented (phase 7).")`. Audit HIGH-confidence. Dead code. | `grep -rn compare_eqn_sets` over `*.py/*.md/*.slurm/*.sh/*.yaml` → only the self-match inside the file. Zero importers, zero doc/slurm references. |
| `slurm-multid_pysr-49350027.out` | SLURM stdout log committed to the repo **root** before `slurm-*.out` was added to `.gitignore` (`.gitignore:20`). Pure run scratch, not code; violates the repo's own ignore rule. Audit HIGH-confidence. NOTE: repo-root file, NOT in the protected `slurm/` directory. | Log file — no Python/doc importer possible; `git ls-files` confirms it was tracked. |
| `slurm-multid_pysr-49355289.out` | Same as above — second committed-then-gitignored SLURM stdout log in the repo root. Audit HIGH-confidence. | Same as above. |

**Count removed: 3 tracked files.**

---

## PROPOSED-FOR-HUMAN (NOT deleted this session)

Confidence column mirrors the audit; the "why deferred" column records the specific
live reference or scope rule that blocked an automatic removal.

| Path | Why (audit) | Confidence | Why deferred (this session) |
|---|---|---|---|
| `.claude/scheduled_tasks.lock` | Machine-local harness lock file, should never have been tracked. Audit HIGH. | high | Harness internal, not project legacy code, and the file is an **active session lock** (contains live `sessionId`/`pid` for the running agent). Out of "legacy code" scope; deleting a live lock risks the running harness. Human should `git rm --cached` it and add `.claude/` to `.gitignore` when no session is active. |
| `.claude/.nfs000000152d21863000002c61` | NFS silly-rename turd (already `D` in git status). Audit HIGH. | high | Harness/filesystem internal under `.claude/`, not project code; already deleted on disk (shows ` D`). Bundle with the lock-file cleanup as a separate `.claude` hygiene commit. |
| `src/priya_forecast.egg-info/` | Untracked build artifact (matched by `*.egg-info/` ignore). Audit HIGH. | high | Untracked → outside `git rm` scope; `rm -rf` is a working-tree wipe, not a tracked-code removal. Safe for a human to `rm -rf`; regenerated on next editable install. |
| `outputs/` (126 MB), 132 root `slurm-*.out`, `**/.ipynb_checkpoints/` | Gitignored PySR/Julia scratch + cluster stdout + Jupyter checkpoints. Audit HIGH. | high | All untracked/gitignored — not tracked code. Working-tree disk cleanup (`rm -rf`), human call; not a code removal. Recommend adding `**/.ipynb_checkpoints/` to `.gitignore`. |
| `scripts/compare_pysr_winners.py` | Plots the PYSR_HYPOTHESIS-era HPO comparison; not referenced by HANDOFF/walkthrough/slurm. Audit MED. | med | Part of the `pysr_hypothesis` cluster below — see coordinated-removal note. No live importer found, but it belongs with the cluster and the cluster has a passing test, so defer the whole group together. |
| `scripts/run_pysr_hypothesis.py`, `src/priya_forecast/pysr_hypothesis.py`, `docs/PYSR_HYPOTHESIS.md`, `results/pysr_hypothesis/`, `docs/figures/pysr_hypothesis/` | "Why does PySR underperform the GP" sweep; superseded by the Sobolev/faithfulness result. Audit MED. | med | **`tests/test_pysr_hypothesis.py` imports `priya_forecast.pysr_hypothesis`** (verified). Removing the module breaks a currently-passing test, so removal must drop the test too — a coordinated multi-file change requiring an owner decision (it changes the test count). `results/pysr_hypothesis/` is also under the protected `results/` tree. |
| `scripts/run_residual_pysr.py`, `src/priya_forecast/refit_residual.py` | Residual-PySR path; MEMORY records IGM thermal params need multi-z, not residual-PySR. Audit MED. | med | `src/priya_forecast/refit_residual.py` (`fit_residual`) is **imported by `scripts/run_residual_pysr.py`** (line 42) and both are cited in `docs/PAPER_NOTES.md`. Not dead — superseded approach with live cross-refs; owner call to retire script + module + doc note together. |
| `scripts/forecast_original_design.py` | One-off forecast-era figure utility. Audit MED. | med | Referenced in `docs/superpowers/specs/2026-05-18-single-z-stage-bc-design.md`. No code importer, but documented as part of a spec; era-stale, owner call. |
| `scripts/regen_sample_figures.py` | One-off forecast-era figure utility. Audit MED. | med | Named in `tests/test_student_equations.py:83` (an assertion **error message** steering a human to re-run it) and in `docs/FIGURES.md`. Not imported, but the test message documents a live workflow; owner call. |
| `scripts/port_pysr_equations.py` | One-off equation-port utility. Audit MED. | med | Referenced in `docs/superpowers/specs/2026-05-18-single-z-stage-bc-design.md`. No code importer; era-stale spec utility, owner call. |
| `scripts/replot.py` | One-off forecast-era figure utility. Audit MED. | med | Referenced in `LOCAL_PAPER_HANDOFF.md` and adjacent to `scripts/closure_at_simdat_target.py`. No code importer; owner call (paper provenance). |
| `LOCAL_PAPER_HANDOFF.md`, `README_v2.md` | Era-specific handoffs (laptop paper-writing; Phase 1.5/2 reproducer). Audit MED. | med | Docs, not code; consolidation/relocation under `docs/` is an editorial decision, not a dead-code removal. Keeps root to one README + HANDOFF — owner call. |
| `results/refit_phase2_production*`, `results/refit_optionC_*phase1_5*`, `results/holdout_multid_phase*`, `results/closure_at_simdat_*`, `results/published_scorecard/`, `results/single_z_stage8/`, `results/multi_z_stage7/`, `results/refit_multid_z2.6-4.2*`, `results/smoke_ap_log_target/` | Phase 1.5/2/Stage 7-8 committed result trees; ≈220 of 310 `results/` files; zero current-doc references. Audit MED ("archive then delete"). | med | **Protected tree (`results/`) — explicitly off-limits this session.** Recommend archiving to a `results-archive` tag/branch before any deletion; owner call. |
| `docs/AP_REMEDIATION_PLAN.md`, `docs/PAIR_FIT_PLAN.md`, `docs/PYSR_PERFORMANCE.md` | Phase-2/3 design docs for the retired σ-ratio/pair-coupling work. Audit LOW. | low | Paper-provenance docs; audit itself says "move, don't delete." Editorial relocation, owner call. |
| `src/priya_forecast/refit_taylor.py` + Taylor-combine path | `refit_taylor` is still imported by tests; keep unless Taylor-combine is formally dropped. Audit LOW. | low | **Live test importer** — keep. Only revisit if the Taylor-combine path is formally retired. |
| `notebooks/01_*.ipynb`, `notebooks/02_*.ipynb`, `notebooks/03_*.ipynb` | Forecast-era tutorials, correct for the GP/forecast API but not the diagnostic. Audit LOW. | low | Pedagogy; keep if README points students at them, otherwise stale. Owner call. |
| `scripts/run_pipeline.py`, `scripts/run_pipeline_multi_z.py`, `scripts/run_batch.py`, `scripts/aggregate_z.py` | Single-z/multi-z stage orchestration. Audit LOW. | low | **Live tests** (`test_run_batch`, `test_aggregate_z`) — keep. Flag only if the stage pipeline is retired. |

**Count proposed: 18 table rows** (groups several multi-file clusters), spanning the
full HIGH-but-out-of-scope, MED, and LOW tiers of the audit.

---

## Verification evidence

- Importer checks run with `grep -rn <name>` over `*.py *.md *.slurm *.sh *.yaml *.toml`,
  excluding `docs/housecleaning/` and `docs/pr_review/`.
- `compare_eqn_sets.py`: zero references outside itself → removed.
- Root `slurm-multid_pysr-*.out`: tracked (`git ls-files`), gitignored by rule
  (`.gitignore:20`), no possible importer → removed.
- Post-removal fast suite: 412 passed / 14 skipped (identical to baseline), so no
  removal needed reverting.
