# REPRODUCE — regenerate the paper's figures

This repo ships **one committed production run** whose cached Pareto fronts and
grad-faith sidecars *are* the paper's artifact. Most figures replay those CSVs
**emulator-free**; a few prediction figures additionally need the GP emulator.

- **Production run:** `results/paper_production_20260630_perz_sobolev_z2.6-4.2/`
  (git-committed: `RUN_MANIFEST.md`, `*/refit/z*/pareto_*.csv` + `grad_faith_*.csv`,
  `seed_band/seed_band_summary.json`, `figures/`).
- **Notebook:** `notebooks/reproduce_paper_figures.ipynb` — runs every **Tier-1**
  figure top-to-bottom and shows it inline; documents the **Tier-2** commands.

## Tiers

| tier | needs | how it reproduces |
|------|-------|-------------------|
| **Tier 1** | `numpy pandas matplotlib scipy` (+ `sympy pyyaml h5py` for the package import) | replays the committed CSVs / JSON — **no GP, PySR, or Julia** |
| **Tier 2** | the GP emulator (`GPy` + `lyaemu`) in the project `.venv` — **Julia/PySR not needed** | re-loads the pickled per-z Sobolev refits and the GP |

## Prerequisites

**Tier 1** (light):

```bash
pip install numpy pandas matplotlib scipy sympy pyyaml h5py
# only the emulator-free helpers are imported: priya_forecast.{parameters,
# grad_faith_io,pareto_diag}. Put src/ on the path: PYTHONPATH=src.
```

**Tier 2** (emulator; also needed to *re-run* fits): the project virtualenv
`.venv` with the multi-fidelity GP emulator installed, and its env:

```bash
export PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env
export JULIA_DEPOT_PATH=$HOME/.julia
export PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full
PY=.venv/bin/python
```

`lya_emulator_full` is the upstream GP (`https://github.com/sbird/lya_emulator`
fork); the trained MF emulator pickles are its default basedir. Julia/PySR are
only required to *re-train* equations (§ Re-run the production fits), not to
reproduce any figure.

## Run the Tier-1 notebook

Open `notebooks/reproduce_paper_figures.ipynb` and **Run All** — the Tier-1 cells
run top-to-bottom with only the light deps and show every figure inline. The
notebook's kernel is `priya-forecast-venv`; point it at any Python that has the
Tier-1 deps (e.g. register the project `.venv`:
`.venv/bin/python -m ipykernel install --user --name priya-forecast-venv`).

Headless equivalent (needs `nbconvert` in the running env; the Tier-1 cells
themselves only import numpy/pandas/matplotlib/scipy):

```bash
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.kernel_name=priya-forecast-venv \
    notebooks/reproduce_paper_figures.ipynb
```

Tier-1 cells write reproduced figures to `results/_repro_scratch/` (git-ignored).

## Figure → command → tier

`PROD=results/paper_production_20260630_perz_sobolev_z2.6-4.2`.
Tier-1 commands are emulator-free (`PYTHONPATH=src python …`). Tier-2 commands
need the `.venv` + the env block above; `FIGREPO` is the paper's LaTeX repo
(`/home/mfho/Latex/Knowledge-Distillation-using-PySR-with-PRIYA-suite`).

| paper figure | file | regen command | tier |
|--------------|------|---------------|:----:|
| Pareto faithfulness (`fig:pareto_faith`) | `pareto_faithfulness.pdf` | `scripts/make_diagnostic_figs.py` ¹ | 1 |
| Faithfulness scorecard (`fig:faith_scorecard`) | `faithfulness_scorecard.pdf` | `scripts/make_diagnostic_figs.py` ¹ | 1 |
| ns budget panel (`fig:ns_budget`) | `ns_budget_panel.pdf` | `scripts/make_diagnostic_figs.py` ¹ | 1 |
| Cross-z faithfulness (`fig:crossz`) | `crossz_faithfulness.pdf` | `scripts/make_diagnostic_figs.py` ¹ | 1 |
| Maxsize sensitivity (budget control) | `maxsize_sensitivity.pdf` | `scripts/regen_maxsize_sensitivity.py --prod $PROD --z 3.6 --out-dir $PROD/figures` | 1 |
| Across-seed band (`fig:seed_band`) | `seed_band.pdf` | plot `seed_band/seed_band_summary.json` (notebook §1.3) | 1 |
| tau0 & Ap prediction (`fig:tau0_ap_pred`) | `pysr_pred_tau0_Ap.pdf` | `$PY $FIGREPO/scripts/regen_fig1.py` ² | 2 |
| dtau0 P1D prediction (`fig:dtau0_p1d_pred`) | `pysr_graphs_3.6_dtau0.pdf` | `$PY $FIGREPO/scripts/regen_fig3.py --param dtau0 --z 3.6 --refit-dir $PROD/sobolev/refit/z3.6 --out-dir $PROD/figures` | 2 |
| 2D de-norm scatter (`fig:denorm_dtau0-ap`) | `2d-denorm-Sobol_dtau0-Ap.pdf` | `$PY $FIGREPO/scripts/regen_fig4.py --params dtau0 Ap --z 3.6 --refit-dir $PROD/sobolev/refit/z3.6 --basedir data/kodiaq_gp --out-dir $PROD/figures` | 2 |
| Multi-D best/worst (Table 3) | `multid_z3.6/multid_bestworst.csv` | `$PY scripts/regen_multid.py --refit-dir $PROD/sobolev --z 3.6 --basedir data/kodiaq_gp --out-dir $PROD/figures/multid_z3.6 --n-sobol 256` | 2 |

¹ The four diagnostic figures come from **one** invocation:

```bash
PYTHONPATH=src python scripts/make_diagnostic_figs.py \
    --value-dir   $PROD/value/refit/z3.6 \
    --sobolev-dir $PROD/sobolev/refit/z3.6 \
    --budget-dir  $PROD/seed_band/z3.6_seed0_budget/refit/z3.6 \
    --crossz-dirs 2.6=$PROD/sobolev/refit/z2.6 \
                  3.6=$PROD/sobolev/refit/z3.6 \
                  4.2=$PROD/sobolev/refit/z4.2 \
    --out-dir     $PROD/figures
```

² `regen_fig1.py` has **no `--refit-dir` flag** — it hard-codes
`results/refit_phase2_production`. Point it at the production artifacts first:
`ln -sfn "$PWD/$PROD/sobolev/refit/z3.6" results/refit_phase2_production`.
For `regen_fig3/4`, `--refit-dir` must be the **z-dir that holds `payloads/` and
`refits/`** (`$PROD/sobolev/refit/z3.6`), not `$PROD/sobolev`. Those pickles were
saved by the production run's `--save-artifacts` and are committed at
`$PROD/sobolev/refit/z3.6/{refits,payloads}/`.

## Re-run the production fits (needs Julia/PySR + SLURM)

The whole run (per-z Sobolev + value baseline + 5-seed band + ns budget control)
is one submit script that self-documents into `RUN_MANIFEST.md`:

```bash
scripts/submit_paper_production.sh                 # submit to GreatLakes SLURM
scripts/submit_paper_production.sh --dry-run       # print sbatch lines only
# overrides:
SLURM_ACCOUNT=<acct> LYA_EMULATOR=/path/to/lya_emulator_full \
    scripts/submit_paper_production.sh
```

After the jobs finish, aggregate the seed band, then regenerate the figures:

```bash
scripts/aggregate_seed_band.py \
    --band-dir $PROD/seed_band --out $PROD/seed_band/seed_band_summary.json
# then the Tier-1 + Tier-2 commands in the table above.
```
