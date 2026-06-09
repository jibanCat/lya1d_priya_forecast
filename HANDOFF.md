# HANDOFF — Lyα P1D PySR forecast / diagnostic

**Last updated:** 2026-06-09
**Branch:** `stage10-multiz-sobolev` (pushed; **PR #6** open against `main`)

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

## The diagnostic (the current science)

For each parameter, score whether a symbolic equation's **slope** ∂P/∂θ matches the
GP's — the only thing a Fisher forecast uses. Metric:
`grad_err = median_k |∂P_F^eq/∂θ ÷ ∂P_F^GP/∂θ − 1|` at fiducial, in **linear P_F**
(Fisher-consistent), gate 0.25. The **Sobolev** derivative-matching loss is the
cure. Key results (single-z z=3.6, real KODIAQ-SQUAD):

- **Fisher's Mirage:** an equation can be value-accurate yet slope-wrong.
- **Taxonomy:** robustly faithful {dtau0, tau0, heref, alphaq, hireionz};
  selection-sensitive {Ap, herei, omegamh2}; **generative Mirage cured by Sobolev**
  {ns, 0.60→0.19}; **resistant** {hub, bhfeedback}.
- **Budget control:** ns value-loss at maxsize=35 still fails (0.32) → the Mirage
  is generative, not search-starvation.
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

## Paper integration (separate repo)

Paper: `~/Latex/Knowledge-Distillation-using-PySR-with-PRIYA-suite/oja_template.tex`
(branch `paper-additions`). The diagnostic is integrated into the two Fisher
`\suggest{}` slots + a Sobolev appendix + taxonomy table + 4 figures, all wrapped in
a purple `\additions{}` macro (machine-drafted, for the user to rewrite in voice);
long sections are left as `[OUTLINE]`. PDF builds (17 pp). The prep changeset,
referee reports, and a phone-readable `PAPER_NARRATIVE.md` live in
`.../pysr_faithfulness_update/`. The paper `.tex` is intentionally **not committed**
(reviewed in LaTeX Workshop).

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
