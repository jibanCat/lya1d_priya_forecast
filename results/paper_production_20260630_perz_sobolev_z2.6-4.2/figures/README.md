# Figures + table fragments (paper artifacts)

**run-id** `prod-20260630-perz-sobolev` (2026-06-30). The tables and the data files are as
committed at tag `v1.2-paper`. The PDFs of the figures the paper renders are the manuscript's
own, drawn by `scripts/make_figures.py` from those files; `notebooks/reproduce_paper.ipynb`
redraws them. `$PROD` = this run dir.

## Figures
Figure numbers follow the order of the current draft, which renders seven figures.
Anything marked *not in the paper* is commented out in the `.tex`.

| file | shows | label | drawn by | source |
|---|---|---|---|---|
| `pysr_pred_tau0_Ap.pdf` | τ0/Ap equation against the GP | `fig:tau0_ap_pred` (Fig. 1) | `scripts/make_figures.py --only prediction` | `sobolev/refit/z3.6/{refits,payloads}/{tau0,Ap}.pkl` (no GP) |
| `param_error_bars.pdf` | mean and maximum error of each 1D equation | `fig:param_error_bars` (Fig. 2) | `… --only param_errors` | `table2_stats.tex` below |
| `seed_band.pdf` | derivative error across five seeds | `fig:seed_band` (Fig. 5) | `… --only seed_band` | `seed_band/seed_band_summary.json` |
| `pareto_faithfulness.pdf` | 11-panel Pareto fronts, coloured by derivative error | `fig:pareto_faith` (Fig. 6) | `… --only pareto` | `{value,sobolev}/refit/z3.6/grad_faith_*.csv` + `seed_band/z3.6_seed0_budget` |
| `ns_budget_panel.pdf` | the n_P panel with its complexity control | `fig:ns_budget` (Fig. 7) | `… --only ns_budget` | the three n_P score files |
| `multid_z3.6/multid_bestworst.{pdf,csv}` | 2D/3D combine best/worst rel-err | the `.csv` is `tab:multid`; the `.pdf` is an earlier two-panel drawing, **not in the paper** | `scripts/regen_multid.py --refit-dir $PROD/sobolev --z 3.6` (GP) | the Sobolev 1D eqs + GP |
| `resolution_correction.pdf` | the HF/LF ratio R(k) of `eq:rescorr` | **not in the paper** (dropped) | `scripts/regen_rescorr.py --out-dir <dir>` | `sobolev/refit/z3.6/refits/*.pkl` (no GP) |
| `faithfulness_scorecard.pdf` | value vs Sobolev knee grad_err per param | **not in the paper** (duplicated `fig:seed_band`) | `scripts/make_diagnostic_figs.py` | `{value,sobolev}/refit/z3.6/{pareto,grad_faith}_*.csv` |
| `crossz_faithfulness.pdf` | z=2.6/3.6/4.2 robustness (data still valid) | **not in the paper** | same (`--crossz-dirs …`) | `sobolev/refit/z{2.6,3.6,4.2}` |
| `maxsize_sensitivity.pdf` | grad_err vs budget, value vs Sobolev | **not in the paper** (numbers kept in §4.2) | `scripts/regen_maxsize_sensitivity.py --prod $PROD --z 3.6` | `sens_maxsize*/`, `budget35_*/`, `{value,sobolev}/refit/z3.6` |
| `pysr_graphs_3.6_dtau0.pdf` | dτ0 1D prediction | **not in the paper** | a script in the manuscript's repository | same + GP |
| `2d-denorm-Sobol_dtau0-Ap.pdf` | dτ0-Ap 2D de-norm | **not in the paper** | a script in the manuscript's repository | same + GP |

## Table fragments (numbers of record)
| file | paper table | source |
|---|---|---|
| `taxonomy_table.tex` | `tab:faith_taxonomy` (value/Sobolev knee grad_err + class) | `{value,sobolev}/refit/z3.6/grad_faith_*.csv` via `knee_row` |
| `per_param_equations.tex` | `tab:per1d_eqs` (knee equations) | `sobolev/refit/z3.6/pareto_*.csv` (knee) |
| `table2_stats.tex` | `tab:stats_table` (1D rel-err) | the per-z fits |
| `multid_z3.6/multid_bestworst.csv` | `tab:multid` (2D/3D combine error) | `regen_multid.py` |
| `multid_z3.6/multid_bestworst_perk.csv` | per-k mean / p90 / max rel-err curves for all ten combos (the rel-err-vs-k panels); same Sobol sample as `multid_bestworst.csv` | `regen_multid.py` (added after `v1.2-paper`) |
| `param_priors_table.{tex,txt}` | `tab:param_table` (priors) | `priya_forecast.parameters` |
| `maxsize_sensitivity.csv` | backs `fig:maxsize_sens` (not in the paper) + the §4.2 budget numbers, which ARE live | the sweep sidecars |

`taxonomy_table.tex` and `per_param_equations.tex` are the **Pareto-knee** versions and match
the paper's `tab:faith_taxonomy` and `tab:per1d_eqs`.
