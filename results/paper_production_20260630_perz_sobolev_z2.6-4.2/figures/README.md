# Figures + table fragments (paper artifacts)

**code git** `7aa26af` · **run-id** `prod-20260630-perz-sobolev` · 2026-06-30.
Each row: file → what it shows → paper `\label` → regen command → data source.
Env for the GP-backed (tier-2) commands: see `../../../REPRODUCE.md`. `$PROD` = this run dir.
The reusable API is `priya_forecast.paper_figures` (path/format-decoupled); the tutorial
is `notebooks/reproduce_paper.ipynb`.

## Figures
| file | shows | label | regen | source |
|---|---|---|---|---|
| `pareto_faithfulness.pdf` | 11-panel Pareto front, coloured by slope error | `fig:pareto_faith` | `scripts/make_diagnostic_figs.py` (tier-1) | `{value,sobolev}/refit/z3.6/{pareto,grad_faith}_*.csv` |
| `faithfulness_scorecard.pdf` | value vs Sobolev knee grad_err per param | `fig:faith_scorecard` | same | same |
| `ns_budget_panel.pdf` | the n_S Mirage + budget arm | `fig:ns_budget` | same (`--budget-dir budget35_value/refit/z3.6`) | + `budget35_value/refit/z3.6` |
| `crossz_faithfulness.pdf` | z=2.6/3.6/4.2 robustness (**figure dropped from the paper**; data still valid) | `fig:crossz` (commented) | same (`--crossz-dirs …`) | `sobolev/refit/z{2.6,3.6,4.2}` |
| `maxsize_sensitivity.pdf` | grad_err vs budget, value vs Sobolev | `fig:maxsize_sens` (commented; numbers kept in §4.4) | `scripts/regen_maxsize_sensitivity.py --prod $PROD --z 3.6` | `sens_maxsize*/`, `budget35_*/`, `{value,sobolev}/refit/z3.6` |
| `seed_band.pdf` | across-seed grad_err band | `fig:seed_band` | notebook §1.3 / `pf.plot_seed_band` | `seed_band/seed_band_summary.json` |
| `multid_z3.6/multid_bestworst.{pdf,csv}` | 2D/3D combine best/worst rel-err | `fig:multid_bestworst` | `scripts/regen_multid.py --refit-dir $PROD/sobolev --z 3.6` (tier-2, GP) | the Sobolev 1D eqs + GP |
| `pysr_pred_tau0_Ap.pdf` | τ0/Ap 1D prediction | `fig:tau0_ap_pred` | `$FIGREPO/scripts/regen_fig1.py --refit-dir $PROD/sobolev/refit/z3.6` (tier-2) | `sobolev/refit/z3.6/{refits,payloads}` + GP |
| `pysr_graphs_3.6_dtau0.pdf` | dτ0 1D prediction | `fig:dtau0_p1d_pred` | `$FIGREPO/scripts/regen_fig3.py --param dtau0 --refit-dir $PROD/sobolev/refit/z3.6` (tier-2) | same + GP |
| `2d-denorm-Sobol_dtau0-Ap.pdf` | dτ0-Ap 2D de-norm | `fig:denorm_dtau0-ap` (**dropped**) | `$FIGREPO/scripts/regen_fig4.py …` (tier-2) | same + GP |

## Table fragments (numbers of record)
| file | paper table | source |
|---|---|---|
| `taxonomy_table.{tex,txt}` | `tab:faith_taxonomy` (value/Sobolev knee grad_err + class) | `{value,sobolev}/refit/z3.6/grad_faith_*.csv` via `knee_row` |
| `per_param_equations.{tex,txt}` | `tab:per1d_eqs` (knee equations) | `sobolev/refit/z3.6/pareto_*.csv` (knee) |
| `table2_stats.tex` | `tab:stats_table` (1D rel-err) | the per-z fits |
| `multid_z3.6/multid_bestworst.csv` | `tab:multid` (2D/3D combine error) | `regen_multid.py` |
| `param_priors_table.{tex,txt}` | `tab:param_table` (priors) | `priya_forecast.parameters` |
| `maxsize_sensitivity.csv` | backs `fig:maxsize_sens` + the §4.4 budget numbers | the sweep sidecars |

NOTE: `taxonomy_table.tex` on disk was generated with the **old best-loss** pick; the
paper's `tab:faith_taxonomy` uses the corrected **knee** numbers (recompute via
`priya_forecast.paper_figures.taxonomy(load_run())`). The knee values are the numbers of record.
