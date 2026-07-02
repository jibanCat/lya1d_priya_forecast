# Per-z Sobolev "production mode" run + full paper update — design

**Date:** 2026-06-30
**Branch (code):** `stage10-multiz-sobolev`
**Paper repo:** `/home/mfho/Latex/Knowledge-Distillation-using-PySR-with-PRIYA-suite` (stays uncommitted)
**Run-id:** `prod-20260630-perz-sobolev`
**Production output dir:** `results/paper_production_20260630_perz_sobolev_z2.6-4.2/`
**SLURM account:** `yueyingn0` (⚠️ **expires 2026-07-01** — compute must be submitted/finished today)

## Goal

Lock the paper onto a clean, self-documenting "production" recipe and propagate it
end-to-end into the draft. Production recipe = **one PySR model per parameter per
redshift** (single-z, 11 params × {2.6, 3.6, 4.2}), trained with the **Sobolev /
derivative-matching loss** (= the "gradient loss"; same mechanism), **no ANOVA
loss**, log target. Deliverables: 1D prediction error, a multi-D (2D+3D) best/worst
prediction-error diagnostic (results-only — NOT in Fig 1), and a loss-vs-complexity
(Pareto) figure comparing the fit **with vs without** the gradient loss.

## Decisions (locked with user 2026-06-30)

1. Redshift grid = **z ∈ {2.6, 3.6, 4.2}** (benchmark + cross-z).
2. **Fresh higher-budget run via sbatch.** Output to a self-explanatory paper-production
   folder; every paper fig/table gets a `%`-comment git stamp (run-id + output dir +
   git hash + timestamp).
3. 2D/3D diagnostic = **broad multi-D sweep** (best/worst rel-err over several combos).
4. Paper = **full sweep now** (remove Figs 10–17; remake Figs 1–4 + all 6 tables;
   update all equations; text changes in `\additions`/`\mfho`; uncommitted).
5. **Remove ALL ANOVA content from the paper** (eq + appendix subsection + every mention).
6. **Rerun the seed band** at the production budget (confirmed).
7. SLURM account = **`yueyingn0`** (expires 2026-07-01).

## Production budget (tunable — flag in review)

- `target_space = log` (MANDATORY with Sobolev), `use_sobolev = 1`, `sobolev_lambda = 5`.
- `maxsize = 20` (UNCHANGED ceiling — preserves the established taxonomy / ns_budget
  control / seed-band narrative; the ns_budget panel separately probes maxsize=35).
- "Higher budget" = `populations = 48` (was 24) + `niter = 200` + multithreading.
- `seed = 0` for headline fits; `seeds 0–4` for the seed band.
- Value (no-gradient) baseline = **plain MSE, ANOVA OFF** (`smart_kwargs = False`),
  same `maxsize/niter/populations`, `target_space = log` (so `value_mse` is comparable).
- *Alternative offered:* bump `maxsize → 30` for deeper equations; this would
  re-open the taxonomy and require re-checking seed-band/ns_budget. Default is NO.

## Critical correctness items (from recon)

- **Sobolev⇄log guard:** `--use-sobolev` with default `--target-space linear` silently
  matches a linear-P slope against a log-P target. Add a guard (error or auto-log).
- **ANOVA off:** Sobolev runs already overwrite the loss (ANOVA auto-off). For the
  value baseline, the single-z CLI does NOT expose `smart_kwargs`, so it would
  silently include ANOVA. Add `--no-smart-kwargs` (plain MSE) to the CLI.
- **`use_anova_loss` config flag is dead** (never read) — do not rely on it.
- The reproducible artifact is the **committed Pareto + grad_faith CSVs**; the notebook
  replays them emulator-free. PySR itself runs multithreaded (not bit-reproducible).

## Output folder layout

```
results/paper_production_20260630_perz_sobolev_z2.6-4.2/
  RUN_MANIFEST.md                 # run-id, git hash, date, exact commands, budget, env
  refit_sobolev/z{2.6,3.6,4.2}/   pareto_<p>.csv + grad_faith_<p>.csv   (11 params)
  refit_value/  z{2.6,3.6,4.2}/   pareto_<p>.csv + grad_faith_<p>.csv   (plain MSE)
  seed_band/    z3.6_seed{0..4}_{value,sobolev,budget}/refit/...
  multid/       bestworst sweep CSVs (2D pairs + 3D triples, mean/90th/max rel-err)
  forecast/     scorecard.md, fisher_*.npz, corner.pdf (if produced)
  figures/      every regenerated paper figure (pdf)
  tables/       every regenerated paper table (tex fragment)
```

## Figure plan (final set, after removing 10–17)

| new # | label | source | action |
|---|---|---|---|
| 1 | `fig:tau0_ap_pred` | `scripts/regen_fig1.py` | regen from production Sobolev fits; **keep 1D-only**; larger fonts |
| 2 | `fig:holdout_validation` | `deliverables.write_holdout_validation` | regen from production; larger fonts |
| 3 | `fig:dtau0_p1d_pred` | new `regen_fig3.py` | **REMAKE** (1D dtau0 pred + ratio); git stamp |
| 4 | `fig:denorm_dtau0-ap` | new `regen_fig4.py` | **REMAKE** (2D denorm dtau0–Ap); git stamp |
| 5 | `fig:seed_band` | `aggregate_seed_band.py` | regen at production budget |
| 6 | `fig:pareto_faith` | `make_diagnostic_figs.py` / `plot_pareto_faithfulness.py` | regen — **this is the loss-vs-complexity WITH vs WITHOUT gradient loss** (value vs Sobolev, y=`value_mse`) |
| 7 | `fig:faith_scorecard` | `make_diagnostic_figs.py` | regen from production |
| 8 | `fig:ns_budget` | `make_diagnostic_figs.py` | regen (Sobolev vs maxsize-35 control) |
| 9 | `fig:crossz` | `make_diagnostic_figs.py` | regen from production cross-z |
| 10 (new) | `fig:multid_bestworst` | new `regen_multid.py` (additive-Taylor combine of Sobolev 1D eqs vs GP) | **NEW** consolidated 2D+3D best/worst rel-err diagnostic (replaces old 11–17) |

Removed: old Figs 10–17 (`fig:ns_p1d_pred`, `fig:dtau0-ap-ns`, `fig:dtau0-ap-ns_rel-error`,
`fig:ns-hub-hireionz_rel-error`, `fig:dtau0-ap_rel-error`, `fig:ns-hub`,
`fig:ns-hub-hireionz`, `fig:dtau0-ap`) + their `\ref`s.

## Table plan (all remade + git stamp)

| # | label | action |
|---|---|---|
| 1 | `tab:param_table` | verify 11 params + priors vs code; git stamp |
| 2 | `tab:stats_table` | regen 1D rel-err @ z=3.6 (LF/HF) via `regen_table2.py` |
| 3 | `tab:rmse_pe_table` | regen RMSE/%err for 1D/2D/3D sets from the multi-D sweep |
| 4 | `tab:stats_28_table` | **re-point z=2.8 → production grid** (z=2.6 & 4.2 cross-z) — flag change |
| 5 | `tab:faith_taxonomy` | regen taxonomy (grad_err value→Sobolev) from production |
| 6 | `tab:per1d_eqs` | regen per-param fit stats + GP-slice notes |

## Equation plan (update to current code/model)

- `eqn:tau0_eq`, `eqn:ap_eq` — **replace with the NEW production τ₀/Ap equations**; git stamp.
- `eq:sobolev` — verify L = ⟨(y_eq−y_GP)²⟩ + λ⟨(∂y_eq−∂y_GP)²⟩, λ=5, **log target**, finite-diff.
- `eq:grad_err`, `eq:fisher_singlez`, `eq:norm`, `eq:combine_{1,2}`, `eq:combine` — verify vs code.
- `eq:anova_loss` (appendix) — **REMOVE entirely** (equation, the appendix subsection, and
  every prose mention of ANOVA / dimension-balanced loss). User decision 2026-06-30.
- `eqn:multid_smoke`/`eqn:multid_slurm` (appendix) — likely removed with the appendix figs.

## Provenance / git-stamp convention (new)

Immediately above each figure/table environment in `oja_template.tex`:
```
%ref: <figs/...pdf or table>
%   run-id: prod-20260630-perz-sobolev
%   regen:  <scripts/regen_*.py>   (paper repo)
%   data:   lya1d_priya_forecast/results/paper_production_20260630_perz_sobolev_z2.6-4.2/...
%   git:    <code-repo commit hash> @ stage10-multiz-sobolev
%   date:   2026-06-30 HH:MM
```

## Orchestration (agent teams)

- **Phase 0** (done): 4-agent context recon. **Cross-validation team**: smoke-test the
  CLI/run commands, verify regen-script input paths exist, confirm GP loads — before compute.
- **Phase A**: plumbing (populations knob → slurm; Sobolev⇄log guard; `--no-smart-kwargs`)
  + unit & hypothesis tests; commit per change.
- **Phase B**: sbatch production arrays → folder + manifest; sidecars.
- **Phase C**: figures + tables — fan-out agents (one per figure/table family).
- **Phase D**: reproducibility team — minimal per-figure scripts + a notebook (emulator-free).
- **Phase E**: paper team — apply the figure/table/equation/provenance plan; build PDF; uncommitted.
- **Phase F**: final cross-validation — figures render, numbers match CSVs, notebook runs, PDF builds.

## Out of scope (explicitly)

- Multi-z (one-model-across-z) refit path — user wants per-z; ignore `slurm/multi_z_refit.slurm`.
- The dropped Stage-10 σ "money plot"; `stash@{0}` log_ratio experiment.
- Joint herei×alphaq *2-param PySR* refit (the additive combine still covers the pair in the sweep).
- Paused Phase-3 Ap σ-ratio remediation.

## Risks

- Budget bump could shift borderline verdicts (esp. ns). Mitigation: maxsize unchanged;
  regenerate ALL dependent figures/tables/taxonomy consistently from the one production run.
- `regen_fig1.py`/`regen_table2.py` reference `results/refit_phase2_production/` (may be
  stale/absent) — Phase-0 cross-validation must confirm before reuse; repoint to the new dir.
- 2D/3D pair *training* path has no Sobolev; the multi-D **combine** inherits Sobolev via
  the 1D eqs (correct) — the sweep uses the combine, not pair-training.
