# lya1d_priya_forecast

A pipeline that distills the **PRIYA** multi-fidelity Gaussian-process (GP) emulator
of the Lyman-α 1D flux power spectrum (P1D) into compact, per-parameter symbolic
equations (via **PySR**), and — the current headline — runs a
**derivative-faithfulness diagnostic** on those equations. The diagnostic asks the
only question a Fisher forecast actually cares about: does a symbolic equation that
reproduces the GP's *values* also reproduce its *slopes* `∂P_F/∂θ`? It scores each
of the 11 PRIYA parameters as derivative-faithful or a "Fisher's Mirage" (right
value, wrong slope), characterizes which parameters fail and why, and shows what a
**Sobolev** derivative-matching loss does and does not fix.

> **Current science lives in two docs**, not in the older σ_PySR/σ_GP forecast
> material (which was superseded — see the note at the bottom):
> - **[`HANDOFF.md`](HANDOFF.md)** — state of the project, results, how to run.
> - **[`docs/PARETO_FAITHFULNESS_WALKTHROUGH.md`](docs/PARETO_FAITHFULNESS_WALKTHROUGH.md)**
>   — the source of truth for the diagnostic: figure, metric definition, taxonomy.

---

## Installation

The repo uses a project virtualenv with **pinned** dependencies for reproducibility.

### 1. Clone

```bash
git clone <repo-url> lya1d_priya_forecast
cd lya1d_priya_forecast
```

### 2. Create the venv and install pinned dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate

# Exact, reproducible install (recommended):
pip install -r requirements.lock.txt

# Editable install of this package:
pip install -e .
```

`requirements.lock.txt` is a full, version-pinned lockfile (numpy 1.26.4, GPy
1.13.2, pysr 1.5.10, etc.). `pip install -e .` installs the `priya_forecast`
package and the `priya-forecast` console entry point.

### Why numpy < 2 (important)

`pyproject.toml` caps `numpy<2` and `pandas<3`. GPy 1.13.2's compiled Cython
extensions are built against numpy 1.x's dtype ABI (96 bytes); numpy 2.x shrank it
to 88 bytes, so importing GPy under numpy 2.x raises
`ValueError: numpy.dtype size changed`. pandas is capped `<3` for the same reason
(pandas 3.x requires numpy >= 2). The lockfile already pins compatible versions; if
you install loosely, keep numpy below 2.

### Optional extras

`pyproject.toml` defines optional dependency groups for partial installs (already
covered by the lockfile):

```bash
pip install -e ".[forecast]"   # emcee, getdist, matplotlib
pip install -e ".[pysr]"       # pysr
pip install -e ".[gp]"         # GPy, emukit
pip install -e ".[hpo]"        # optuna
pip install -e ".[dev]"        # pytest, hypothesis, ruff
```

### Prerequisites for emulator / PySR runs (NOT needed for the figure reproducer)

The diagnostic figures regenerate **emulator-free** (see Usage). The following are
only required to *retrain* equations, run the GP, or re-evaluate the gate:

1. **Julia / PySR environment** — PySR needs a Julia backend. Point it at the
   shared project depot:

   ```bash
   export PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env
   export JULIA_DEPOT_PATH=$HOME/.julia
   ```

2. **Upstream emulator on `PYTHONPATH`** — the GP/emulator code lives in a separate
   clone of the sbird `lya_emulator` (here `lya_emulator_full`). It must be on the
   path alongside this repo's `src/`:

   ```bash
   export PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full
   ```

3. **GP data** — the trained PRIYA GP and KODIAQ-SQUAD inputs must be present under
   `data/kodiaq_gp/` (trained emulator, flux vectors, `emulator_params.json`).

---

## Usage

All commands assume the venv is active (`source .venv/bin/activate`) and that you
run them **from the repo root** (the figure script uses relative `results/...`
paths).

### Run the tests

Emulator-free; this is the quickest sanity check on an install:

```bash
PYTHONPATH=src pytest tests/ -q -k "not slow"
```

(412 pass, ~13 skip. `test_real_gp_predicts_at_fiducial` is environment-dependent:
it **skips** when the upstream emulator is absent, or raises a pre-existing
numpy<2/GPy ABI error if GPy is importable under numpy 2.x — unrelated to this code.)

### Regenerate the diagnostic figures (emulator-free — the paper reproducer)

This reads the committed PySR Pareto fronts (`results/.../pareto_*.csv`) and their
grad-faithfulness sidecars (`results/.../grad_faith_*.csv`) and rebuilds all **four**
diagnostic figures — **no GP, no PySR, no Julia required**:

```bash
PYTHONPATH=src python scripts/make_diagnostic_figs.py \
    --out-dir results/single_z_stage_pareto_diag
```

Outputs (PNG + PDF) into the given directory:
- `pareto_faithfulness` — 11-panel Pareto grid, y = `value_mse`, color = `grad_err`,
  gate rings.
- `faithfulness_scorecard` — value-loss vs Sobolev best-loss `grad_err` per
  parameter.
- `ns_budget_panel` — the paired budget-vs-Sobolev comparison for `ns`.
- `crossz_faithfulness` — redshift robustness of the taxonomy (z = 2.6 / 3.6 / 4.2).

(The committed copies live in `results/single_z_stage_pareto_diag/`; pass a scratch
`--out-dir` if you only want to inspect without touching them.)

### Re-evaluate the gate / regenerate sidecars (needs the emulator)

With the emulator prerequisites above in place:

```bash
scripts/make_grad_faith_sidecars.sh <pareto_dir> <z>
```

### Notebooks (and how to launch them)

Jupyter is not in the lockfile (the pipeline doesn't need it), so install it into
the venv and register the kernel once:

```bash
pip install jupyterlab ipykernel
python -m ipykernel install --user --name priya-forecast --display-name "priya-forecast (.venv)"
PYTHONPATH=src jupyter lab        # then open a notebook from notebooks/
```

**Start here — emulator-free (no GP/PySR/Julia):**
- `notebooks/reproduce_paper_figures.ipynb` — regenerates the four diagnostic
  figures from the committed sidecars (the paper reproducer).
- `notebooks/tutorial_01_explore_diagnostic.ipynb` — a guided tour of the
  diagnostic (the Mirage in one table, the taxonomy, the budget control).

**Older GP/forecast walkthroughs (need the emulator prerequisites):**
`01_gp_only.ipynb`, `02_forecast_only.ipynb`, `03_refit_and_forecast.ipynb` —
these predate the diagnostic pivot and still exercise the σ-forecast path; treat
[`docs/PARETO_FAITHFULNESS_WALKTHROUGH.md`](docs/PARETO_FAITHFULNESS_WALKTHROUGH.md)
as the authoritative current result.

---

## Repository layout

```
src/priya_forecast/      Installed package: GP + PySR models, Fisher forecast,
                         the derivative gate, Sobolev loss, grad-faith sidecar
                         I/O (grad_faith_io.py), Pareto diagnostic (pareto_diag.py),
                         single_z/ and multi_z/ pipelines, the priya-forecast CLI.
scripts/                 Drivers and runners: make_diagnostic_figs.py (figure
                         reproducer), eval_grad_faithfulness.py, plot_pareto_
                         faithfulness.py, make_grad_faith_sidecars.sh, refit/
                         pipeline/HPO entry points.
tests/                   pytest suite (run with -k "not slow"); emulator-touching
                         tests are gated/skipped.
notebooks/               Tutorial notebooks 01–03 (GP / forecast walkthroughs).
docs/                    PARETO_FAITHFULNESS_WALKTHROUGH.md (current science),
                         plus design specs, figures, and historical notes.
configs/                 YAML run configs (default, diagnostic, single_z, multi_z,
                         hpo, eqns).
data/                    kodiaq_gp/ (trained GP + KODIAQ-SQUAD inputs),
                         priya_fiducial/, single_z_1pvar/.
results/                 Committed run outputs, including the grad-faith sidecars
                         the figure reproducer reads.
```

---

## Pointers

- **[`HANDOFF.md`](HANDOFF.md)** — current project state, results, run recipes.
- **[`docs/PARETO_FAITHFULNESS_WALKTHROUGH.md`](docs/PARETO_FAITHFULNESS_WALKTHROUGH.md)**
  — the derivative-faithfulness diagnostic, in full.

> **Note on older material.** The project pivoted (2026-06-08) from a σ_PySR/σ_GP
> *forecast* claim to this *diagnostic / failure-modes* result, after a review found
> the σ-ratio confounded by construction (σ_perfect_1D ≡ σ_GP is a forced Jacobian
> identity). Older docs that headline the σ-ratio forecast (e.g. `README_v2.md`,
> `LOCAL_PAPER_HANDOFF.md`, parts of `docs/ONBOARDING.md` and `docs/REPRODUCE.md`)
> are retained for history but are **superseded** by `HANDOFF.md` and the
> walkthrough.
