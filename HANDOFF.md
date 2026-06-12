# HANDOFF — Lyα P1D PySR forecast / diagnostic

**Last updated:** 2026-06-11
**Branch:** `stage10-multiz-sobolev` (pushed, latest `ea814dc`; **PR #6** open against `main`)

---

## TL;DR — what this repo is now

A pipeline that distills the PRIYA multi-fidelity Gaussian-process (GP) emulator of
the Lyman-α 1D flux power spectrum (P1D) into compact per-parameter symbolic
equations (PySR), and — the current headline — a **derivative-faithfulness
diagnostic** that asks whether those equations are trustworthy for a **Fisher
forecast**.

**Direction (set 2026-06-08):** the project pivoted from a σ_PySR/σ_GP *forecast*
claim to a **diagnostic / failure-modes** result. The 4-agent review found the
σ-ratio confounded by construction (σ_perfect_1D ≡ σ_GP is a forced Jacobian
identity anchored at P_GP(fid); the GP-slice fallback prints GP-derived σ in the
PySR column). The diagnostic is what the GP-as-oracle setup can honestly support.

## Most recent session (2026-06-11): across-seed band → publication blocker DISCHARGED

The 4-referee publishability panel had **one true blocker**: the taxonomy was a
single PySR seed (PySR is stochastic). Ran the across-seed band (5 seeds) and it
holds up. Aggregator: `scripts/aggregate_seed_band.py` (loads the GP once, scores
the value-optimal Fisher-safe equation of every (seed, mode) front); output
`results/seed_band/seed_band_summary.json` + `seed_band.{pdf,png}` (all committed).
Verdict:
- **Budget control is seed-robust:** ns value@maxsize-35 fails the gate at *every*
  seed (0.39–0.67). Deeper search reliably does not fix the slope.
- **Sobolev verdict is seed-stable for 10/11:** 8 clear the gate at all seeds;
  hub + bhfeedback resist at all seeds.
- **Value-loss *selection* is seed-fragile** (wide whiskers, several straddle) →
  sharpened thesis: picking the value-optimal equation is an unreliable route to a
  faithful slope; the **Sobolev objective is the stable one**.
- **ns is borderline, NOT "cured":** Sobolev 0.33 median [0.21, 0.42] straddles the
  gate — the committed single-seed 0.193 was an optimistic draw. Reclassified ns as
  *generative Mirage (borderline)* everywhere.
- **Fisher-free gate defense** (user dropped the covariance/Fisher gate as "too
  hard"): re-classifying at gate ∈ {0.20, 0.25, 0.30} gives an **identical split at
  0.25 and 0.30** → 0.25 is a defensible chosen operating point; readers apply their
  own tolerance from the reported grad_err. No Fisher machinery needed.

Reframe **propagated this session**: walkthrough threshold-robustness table
(committed `ea814dc`) + paper `oja_template.tex` sec:fisher_results (`\additions{}`
seed paragraph + `fig:seed_band`) + taxonomy table ns row + `PAPER_NARRATIVE.md`
§7f. All paper-repo edits left UNCOMMITTED per that repo's CLAUDE.md.

## The diagnostic (the current science)

For each parameter, score whether a symbolic equation's **slope** ∂P/∂θ matches the
GP's — the only thing a Fisher forecast uses. Metric:
`grad_err = median_k |∂P_F^eq/∂θ ÷ ∂P_F^GP/∂θ − 1|` at fiducial, in **linear P_F**
(Fisher-consistent), gate 0.25. The **Sobolev** derivative-matching loss is the
cure. Key results (single-z z=3.6, real KODIAQ-SQUAD):

- **Fisher's Mirage:** an equation can be value-accurate yet slope-wrong.
- **Taxonomy:** robustly faithful {dtau0, tau0, heref, alphaq, hireionz};
  selection-sensitive {Ap, herei, omegamh2}; **generative Mirage, Sobolev strongly
  improves but borderline across seeds** {ns, 0.60→0.33 median [0.21,0.42]};
  **resistant** {hub, bhfeedback}.
- **Budget control:** ns value-loss at maxsize=35 still fails (0.32 single-seed;
  0.39–0.67 across 5 seeds) → the Mirage is generative, not search-starvation.
- **Cross-z (z=2.6/3.6/4.2, retrained):** taxonomy is NOT redshift-uniform — the
  He II reion block (herei, heref, alphaq) is faithful at z≤3.6 and blows up at
  z=4.2 (its imprint weakens; the gate can't adjudicate a near-noise slope).
- **h basis test:** ∂P/∂h is NOT a k-rescaling (corr ≈ −0.25, ~6% var) → h resists
  because its response is **weak/under-determined**, not an AP coordinate-distortion
  basis wall. (The earlier "h = AP" guess is refuted.)

## Code (this session's diagnostic build — all committed, tests pass)

- `src/priya_forecast/grad_faith_io.py` — sidecar format (pure, emulator-free).
- `src/priya_forecast/pareto_diag.py` — `load_front` + `render_grid` (gate rings).
- `scripts/eval_grad_faithfulness.py` — per-candidate grad_err + value_mse (needs GP).
- `scripts/plot_pareto_faithfulness.py`, `scripts/make_grad_faith_sidecars.sh`,
  `scripts/make_diagnostic_figs.py`, `scripts/h_basis_test.py`.
- Tests: `tests/test_grad_faith_io.py`, `tests/test_pareto_diag.py` (5, pass).
- Results: `results/single_z_stage_pareto_diag/` (figures), grad-faith sidecars
  beside each `pareto_*.csv` under `single_z_stage6_log`, `single_z_stage9`,
  `decider_budget_z3.6`, and the cross-z `single_z_z{2.6,4.2}_{value,sobolev}`.
- Walkthrough (source of truth): `docs/PARETO_FAITHFULNESS_WALKTHROUGH.md`.

## Production result plots (the 5 paper figures)

All git-tracked (`.pdf` + `.png`), regenerated emulator-free from the sidecars:

| figure | code repo (source of truth) | paper repo (`\includegraphics`) |
|---|---|---|
| `pareto_faithfulness` (central) | `results/single_z_stage_pareto_diag/` | `~/Latex/…/figs/` |
| `faithfulness_scorecard` (one-glance) | `results/single_z_stage_pareto_diag/` | `~/Latex/…/figs/` |
| `ns_budget_panel` (money plot) | `results/single_z_stage_pareto_diag/` | `~/Latex/…/figs/` |
| `crossz_faithfulness` (z-robustness) | `results/single_z_stage_pareto_diag/` | `~/Latex/…/figs/` |
| `seed_band` (across-seed band) | `results/seed_band/` | `~/Latex/…/figs/` |

Regenerate the first four: `PYTHONPATH=src python scripts/make_diagnostic_figs.py
--out-dir results/single_z_stage_pareto_diag`. Regenerate `seed_band` (needs GP):
`scripts/aggregate_seed_band.py` then its plotter. Paper-repo `figs/` copies are
**uncommitted** (paper repo isn't committed).

## Paper integration (separate repo)

Paper: `~/Latex/Knowledge-Distillation-using-PySR-with-PRIYA-suite/oja_template.tex`.
The diagnostic is integrated into `sec:fisher` (methods: Fisher/grad_err/Sobolev eqs
+ Mirage paragraph) and `sec:fisher_results` (central-result paragraph, seed-
robustness paragraph, **5 figures** incl. `fig:seed_band`, taxonomy table), plus a
Sobolev appendix — all wrapped in a purple `\additions{}` macro (machine-drafted, for
the user to rewrite in voice). PDF builds, **17 pp, `fig:seed_band` resolves** (only
pre-existing `sec:2d_preds`/`sec:3d_preds` refs undefined — unrelated student stubs).
The prep changeset, referee reports, and a phone-readable `PAPER_NARRATIVE.md`
(§7f = seed band) live in `.../pysr_faithfulness_update/`. The paper repo is
intentionally **not committed** (user reviews the PDF in LaTeX Workshop).

**Immediate next step (offered, not yet done):** two of my own
`\additions{[OUTLINE … to flesh out]}` blocks still render literally in purple —
`oja_template.tex:478–483` (methods: stencil through the combined model + the
non-negligible-k mask; gate-as-selection vs Sobolev-as-generative; diagonal-Fisher
caveat) and `:651–657` (discussion: per-parameter table reading; the two resisters
incl. the h basis-test refutation; cross-z implication; honest caveats). Content is
all decided — just needs prose. The student stubs at `:453`/`:489` are NOT mine.

## How to run

- **Fast tests:** `PYTHONPATH=src pytest tests/ -q -k "not slow"` — on a bare clone:
  **411 passed, 14 skipped, 0 failed** (emulator-touching tests skip cleanly); with
  the emulator data present, 412 passed / 13 skipped (`test_real_gp_predicts_at_fiducial`
  is the one env-dependent case: a pre-existing numpy<2/GPy ABI error if GPy imports
  under numpy 2.x; unrelated to this code).
- **Emulator/PySR runs:** need `PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env
  JULIA_DEPOT_PATH=$HOME/.julia` and `PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full`,
  plus the project venv `.venv` (pinned numpy<2; see README) and `data/kodiaq_gp/`.
- **Regenerate the diagnostic figures (emulator-free, from sidecars):**
  `PYTHONPATH=src python scripts/make_diagnostic_figs.py --out-dir results/single_z_stage_pareto_diag`.
- **Reproduce a gate eval (needs GP):** `scripts/make_grad_faith_sidecars.sh <pareto_dir> <z>`.

## Pointers

- Spec/plan: `docs/superpowers/specs|plans/2026-06-08-pareto-faithfulness-diagnostic*`.
- Memory: `~/.claude/projects/-home-mfho-lya1d-priya-forecast/memory/` —
  `active_work.md` (current state + taxonomy + cross-z + h basis test),
  `review_verdict_sr_emulator.md` (the 4-agent review).
- Earlier multi-z Sobolev "money plot" plan (Stage 10 Task 4) is **dropped** in
  favour of the diagnostic; that history is in git.
