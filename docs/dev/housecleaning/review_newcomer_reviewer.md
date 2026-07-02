# Newcomer reviewer — can a fresh clone install + reproduce a figure from README + repro notebook?

Reviewer role: pretend I just `git clone`d the repo and follow ONLY
`README.md` + `notebooks/reproduce_paper_figures.ipynb`. List every place I get
stuck. All claims below were verified by running commands / executing the
notebook against the actual repo (project `.venv`, numpy 1.26.4).

## Verdict: **CHANGES REQUIRED**

The install path and the test/figure *commands as written* work in THIS working
tree. But the headline reproducer is **broken in a fresh clone**: the
`pareto_*.csv` files it depends on are untracked, so a newcomer who clones and
runs the README's figure command (or the repro notebook) hits a hard
`FileNotFoundError` on the first parameter. That is a blocker. Two smaller
discoverability gaps (README never points at the new repro notebook; Jupyter not
installable from the documented steps) compound it.

---

## Issue list (prioritised)

### P0 — BLOCKER: repro depends on untracked `pareto_*.csv`; fresh clone crashes

The figure reproducer and the repro notebook both call
`load_front(".../pareto_<param>.csv", ".../grad_faith_<param>.csv")`.
`grad_faith_*.csv` are committed; **`pareto_*.csv` are NOT** (never `git add`ed —
not gitignored). Tracked-vs-on-disk check:

| dir | pareto on disk / tracked | grad_faith on disk / tracked |
|---|---|---|
| `results/single_z_stage6_log/refit/z3.6` | 11 / **0** | 11 / 11 |
| `results/single_z_stage9/refit/z3.6` | 11 / **0** | 11 / 11 |
| `results/decider_budget_z3.6/refit/z3.6` | 1 / **0** | 1 / 1 |
| `results/single_z_z2.6_sobolev/refit/z2.6` | 11 / 11 | 11 / 11 |
| `results/single_z_z4.2_sobolev/refit/z4.2` | 11 / 11 | 11 / 11 |

`src/priya_forecast/pareto_diag.py:31` reads the pareto CSV with **no existence
guard** (only the sidecar is guarded, line 32), so a missing pareto file raises.

Reproduced by hiding the untracked files (fresh-clone simulation) and running the
README's exact command:

```
PYTHONPATH=src python scripts/make_diagnostic_figs.py --out-dir <out>
# FileNotFoundError: 'results/single_z_stage6_log/refit/z3.6/pareto_dtau0.csv'
```

The repro notebook fails the same way: cell 4's sanity check only asserts the
`grad_faith_*` sidecars exist (which pass), then cell 6 crashes in `load_front`
on the missing pareto file.

**Fix (one command, safe — files are not gitignored, just unadded):**
```bash
git add results/single_z_stage6_log/refit/z3.6/pareto_*.csv \
        results/single_z_stage9/refit/z3.6/pareto_*.csv \
        results/decider_budget_z3.6/refit/z3.6/pareto_ns.csv \
        results/decider_budget_z3.6/refit/z3.6/grad_faith_ns.csv
```
(The whole `results/decider_budget_z3.6/` tree is untracked — `git add` it.)
Optionally also harden `load_front` to skip a series whose pareto CSV is missing,
so the figure degrades instead of crashing.

### P1 — README never points at the repro notebook (the deliverable is undiscoverable)

`README.md` Usage/"Where to start" lists only the OLD tutorials
`01_gp_only / 02_forecast_only / 03_refit_and_forecast.ipynb` and explicitly says
they "predate the diagnostic pivot." It never mentions
`notebooks/reproduce_paper_figures.ipynb` or
`notebooks/tutorial_01_explore_diagnostic.ipynb`. A newcomer reading only the
README would never find the emulator-free repro notebook.
**Fix:** add a bullet under "Regenerate the diagnostic figures" / "Where to start"
pointing at `reproduce_paper_figures.ipynb` (emulator-free, the notebook form of
`make_diagnostic_figs.py`) and `tutorial_01_explore_diagnostic.ipynb`.

### P1 — Jupyter is not installable from the documented steps

`requirements.lock.txt` and `pyproject.toml` contain **no** jupyter / notebook /
nbconvert / jupyterlab. `nbconvert` and `jupyter_client` are absent from the
`.venv` (only `ipykernel`/`IPython` happen to be present from the base mamba
env — a real fresh venv would have neither). The README never tells a newcomer
how to launch a notebook. So after the documented install, `jupyter lab` /
`jupyter notebook` does not exist.
**Fix:** add `pip install jupyterlab` (or a `[notebook]` extra) to the README's
notebook section, plus a kernel-registration line (see next).

### P2 — repro notebook pins an unregistered kernel `priya-forecast-venv`

`reproduce_paper_figures.ipynb` kernelspec is
`name: priya-forecast-venv` / display `priya-forecast (.venv)`. A fresh clone has
no such kernel, so Jupyter prompts to pick a kernel / may fail headless
(`jupyter nbconvert --execute` needs `--ExecutePreprocessor.kernel_name=python3`).
`tutorial_01_explore_diagnostic.ipynb` uses the generic `python3` — so the two
new notebooks are inconsistent.
**Fix:** either repin the repro notebook to `python3`, or add to the README:
`python -m ipykernel install --user --name priya-forecast-venv --display-name "priya-forecast (.venv)"`.

### P2 — README test-claim wording is inaccurate (not blocking)

README says: "~412 pass, a handful skip; one pre-existing numpy<2/GPy environment
error on `test_real_gp_predicts_at_fiducial`." Actual run:
**412 passed, 14 skipped, 0 errors** in 33s. `test_real_gp_predicts_at_fiducial`
**skips** (via `real_gp_skip_if_unavailable`), it does not error. Tighten to
"412 passed, 14 skipped" and drop the "environment error" phrasing.

### P3 — README implies only `grad_faith_*.csv` are needed; understates pareto dep

"reads the committed grad-faithfulness sidecars (`results/.../grad_faith_*.csv`)"
hides that the reproducer also needs `pareto_*.csv`. After the P0 fix, add the
pareto CSVs to that sentence so the data contract is accurate.

### P3 — build scratch left in `notebooks/`

`notebooks/_build_reproduce_paper_figures.py` and `_build_tutorial_01.py` (the
generators) sit next to the shipped notebooks. Harmless but noise for a newcomer;
move under `docs/housecleaning/` or `scripts/`, or gitignore.

---

## What works (verified, no action needed)

- `.venv` is python3.11 + numpy 1.26.4; `requirements.lock.txt` is 56 plain PyPI
  pins (matplotlib 3.10.9, GPy 1.13.2, pysr 1.5.10 all present — no
  local/git/url/editable entries), so the lockfile is portable.
- README figure command runs clean **in this working tree** (3 figures PNG+PDF).
- `reproduce_paper_figures.ipynb` executes end-to-end **emulator-free** — all 11
  code cells clean, 4 figures written; `GPy / pysr / lyaemu / julia` all confirmed
  NOT imported. (Note: it makes a 4th `crossz_faithfulness` figure the README's
  "three figures" text doesn't mention.)
- `tutorial_01_explore_diagnostic.ipynb` also executes clean and emulator-free.
- `PYTHONPATH=src pytest tests/ -q -k "not slow"` → 412 passed, 14 skipped.
- `priya-forecast` console entry point (`priya_forecast.cli:main`) is importable;
  the package is editable-installed, and the notebook's `sys.path.insert(0, src)`
  makes it work with or without `pip install -e .`.
- numpy<2 / GPy-ABI rationale in README and `pyproject.toml` is correct.
- Emulator-only prereqs are correctly scoped as NOT needed for the reproducer;
  `/home/mfho/student_projects/lya_emulator_full` and `data/kodiaq_gp/` exist on
  this host (a fresh clone elsewhere won't have them, but the README says as much).
