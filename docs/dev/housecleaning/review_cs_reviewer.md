# CS reviewer — housecleaning validation

Reviewer: cs_reviewer
Date: 2026-06-09
Branch: `stage10-multiz-sobolev`
Scope: README install/usage correctness; notebook nbformat + emulator-free execution;
legacy cleanup safety (tests).

## Verdict: PASS (with minor doc fixes)

Everything load-bearing works as written. The emulator-free figure reproducer, both
new emulator-free notebooks, and the fast test suite all run cleanly. The legacy
removal is safe. The only issues are two small documentation inaccuracies (directory
role + one test-status word) — none of which break a command. They are CHANGES-nice,
not CHANGES-blocking.

### What I actually ran (all green)

- `PYTHONPATH=src python scripts/make_diagnostic_figs.py --out-dir /tmp/cs_review_figs`
  → exit 0, wrote `pareto_faithfulness`, `faithfulness_scorecard`, `ns_budget_panel`
  (png+pdf). Emulator-free confirmed.
- `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q -k "not slow"`
  → **412 passed, 14 skipped, 0 errors** in 57s. Matches README and LEGACY_REMOVAL log
  exactly, with the 3 files removed (staged).
- `jupyter nbconvert --execute reproduce_paper_figures.ipynb` → exit 0 (emulator-free;
  imports no GPy/PySR/Julia; reads only committed sidecars).
- `jupyter nbconvert --execute tutorial_01_explore_diagnostic.ipynb` → exit 0
  (top-level imports = numpy/pandas/priya_forecast/IPython only; zero emulator/Julia).
- `nbformat.validate` on all 5 notebooks → all VALID (nbformat 4.5).
- Verified the 3 git-rm'd files have zero live importers; verified the deferred
  `pysr_hypothesis` and `refit_residual` modules DO have live importers (deferral
  correct).

### Verified-correct README claims

- Pins: numpy 1.26.4, GPy 1.13.2, pysr 1.5.10, pandas 2.3.3 — all match
  `requirements.lock.txt`. `pyproject.toml` caps `numpy<2`/`pandas<3`.
- Optional extras `forecast/pysr/gp/hpo/dev` — all present in `pyproject.toml`.
- Console entry point `priya-forecast = priya_forecast.cli:main` — present; `cli.py`
  has `main()`.
- `--out-dir` flag, all 3 output figure names, `scripts/make_grad_faith_sidecars.sh`,
  `scripts/eval_grad_faithfulness.py`, `scripts/plot_pareto_faithfulness.py`,
  `grad_faith_io.py`, `pareto_diag.py` (`load_front`/`render_grid`/`GATE_TOL=0.25`),
  `configs/` YAMLs — all exist as described.
- python3.11 available (3.11.0).

## Issue list (prioritised)

### P1 — Doc inaccuracy: `single_z_stage_pareto_diag/` is the OUTPUT dir, not the sidecar source

The README ("reads the committed grad-faithfulness sidecars … `--out-dir
results/single_z_stage_pareto_diag`") and the `repro_nb` summary ("regenerate … from
the committed `results/single_z_stage_pareto_diag/` sidecars") both imply the input
sidecars live in `results/single_z_stage_pareto_diag/`. They do **not**: that dir
contains only the committed output PNG/PDF figures (0 `grad_faith_*.csv`). The real
committed input sidecars are read from (see `make_diagnostic_figs.py:32-34`):

- `results/single_z_stage6_log/refit/z3.6/` (VALUE, 11 sidecars, tracked)
- `results/single_z_stage9/refit/z3.6/` (SOBOLEV, 11 sidecars, tracked)
- `results/decider_budget_z3.6/refit/z3.6/` (BUDGET, `grad_faith_ns.csv`, tracked)

The script/notebook *code* is correct (`single_z_stage_pareto_diag` is only ever the
`--out-dir`/`OUT`); only the prose mislabels it. Note `decider_budget_z3.6/` shows as
untracked in `git status` because it's a new dir, but `git ls-files` confirms
`grad_faith_ns.csv` IS tracked — so the reproducer is genuinely self-contained.

Fix: in README Usage and the repro-notebook markdown, say "reads the committed
sidecars under `results/single_z_stage{6_log,9}/refit/z3.6/` and
`results/decider_budget_z3.6/refit/z3.6/`; `--out-dir` is only where figures are
written (default `results/single_z_stage_pareto_diag/`)."

### P2 — README: "one … environment **error**" should be "skip"

README:112 says `test_real_gp_predicts_at_fiducial` produces "one pre-existing
numpy<2/GPy environment **error**." It does not error — it **SKIPs** cleanly
(`SKIPPED tests/test_gp_model.py:113: could not import 'lyaemu'`). The full run is
`412 passed, 14 skipped, 0 errors`. Fix: change "error" → "skip" and drop the
implication that anything is broken; "~412 pass, a handful skip" → "412 pass, 14 skip"
is exact.

### P3 (cosmetic) — README "Where to start" lists only 01/02/03, not the two new notebooks

The new `tutorial_01_explore_diagnostic.ipynb` and `reproduce_paper_figures.ipynb`
(the emulator-free, current-diagnostic notebooks this housecleaning added) are not
mentioned in the README's notebook list, which still points only at the older
forecast-era 01/02/03. Recommend adding both, flagged as emulator-free, since they are
the ones aligned with the current headline. (Also: README says 01/02/03 "require the
emulator prerequisites" — 01 and 02 import no emulator module at all; only 03 uses
`GPModel`. Minor over-statement, not blocking.)

## Legacy cleanup safety: SAFE

- 3 files git-rm'd (staged, not committed): `scripts/compare_eqn_sets.py` (verified
  zero importers; was an unimplemented `raise SystemExit(... phase 7)` stub) and the
  two root `slurm-multid_pysr-{49350027,49355289}.out` logs (tracked, gitignored by
  rule, no possible importer).
- Fast suite after removal: 412 passed / 14 skipped — identical to the logged baseline.
  No test referenced any removed file.
- Deferrals are correctly justified: `tests/test_pysr_hypothesis.py:18` imports
  `priya_forecast.pysr_hypothesis`, and `scripts/run_residual_pysr.py:42` imports
  `priya_forecast.refit_residual.fit_residual` — both confirmed live, so leaving them
  for a human was the right call.
