# Production run — per-z Sobolev distillation (paper artifacts)

**run-id** `prod-20260630-perz-sobolev` · **code git** `7aa26af` @ `stage10-multiz-sobolev` · **date** 2026-06-30
Full job list + git stamp: [`RUN_MANIFEST.md`](RUN_MANIFEST.md). How to reproduce every figure/table: repo-root [`REPRODUCE.md`](../../REPRODUCE.md).

## What this is
One PySR model per (parameter, redshift) distilling the PRIYA Lyα-forest P1D GP
emulator. **Recipe:** Sobolev/derivative-matching loss (λ=5, log target, **no ANOVA**)
vs a value-loss baseline (same no-trig operators, plain MSE); `maxsize=20`,
`populations=48`, `niter=200`; z = 2.6/3.6/4.2. Plus a 5-seed band and a
`maxsize∈{30,35,40}` budget/sensitivity sweep. These artifacts back the paper's
derivative-faithfulness ("Fisher's Mirage") results.

## Layout
| path | contents |
|---|---|
| `sobolev/refit/z<z>/`, `value/refit/z<z>/` | per-param `pareto_<p>.csv` (the PySR front) + `grad_faith_<p>.csv` (the emulator-scored sidecar). `sobolev/refit/z3.6/` also has `refits/`+`payloads/` pkls (the prediction-fig artifacts). See `sobolev/README.md` for the CSV schema. |
| `seed_band/z3.6_seed{0..4}_{value,sobolev,budget}/refit/z3.6/` | the 5-seed fronts; aggregated by `scripts/aggregate_seed_band.py` into `seed_band/seed_band_summary.json`. See `seed_band/README.md` for the JSON schema. |
| `budget35_value/`, `budget35_sobolev/` | the `maxsize=35` budget-control arm. |
| `sens_maxsize{30,40}_{value,sobolev}/` | the maxsize-sensitivity sweep. |
| `figures/` | every regenerated paper figure (PDF/PNG) + the table fragments (`*.tex`/`*.txt`) + `maxsize_sensitivity.csv` + `multid_z3.6/multid_bestworst.csv`. See `figures/README.md`. |

## Committed vs regenerable
The **CSVs / sidecars / JSON / table fragments are committed** — the paper's
emulator-free figures + tables (taxonomy, diagnostic figures, sensitivity, seed
band) replay from them with only the light deps in `requirements-figures.txt`
(see `REPRODUCE.md` Step 1, or the reusable module `priya_forecast.paper_figures`
and `notebooks/reproduce_paper.ipynb`). The **PNG/PDF figures** are regenerable;
`data/kodiaq_gp/` (the GP basedir, 43 MB) is **git-ignored** — the GP-backed
prediction figures need it (build via `scripts/prep_kodiaq_gp.py`, `REPRODUCE.md` Step 2).
