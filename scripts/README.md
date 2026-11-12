# scripts/

## Drawing the paper's figures: `make_figures.py`

`make_figures.py` draws the seven figures of the paper exactly as they appear in the
manuscript. It reads only files committed in this repository, so it needs no GP emulator
and no PySR. `figstyle.py` holds the shared style (colours, fonts, sizes).

**Needs:** `pip install -r requirements-figures.txt`, a full `git clone` (each input is
checked against tag `v1.2-paper` before it is plotted), and LaTeX on `PATH`.

### From a terminal

```bash
python scripts/make_figures.py --out my_figures                    # all seven, about 20 s
python scripts/make_figures.py --out my_figures --only seed_band   # one figure
python scripts/make_figures.py --out my_figures --png              # also write PNG previews
python scripts/make_figures.py --out my_figures --dump my_values   # also write the plotted numbers as JSON
```

### From Python, one function per figure

```python
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
import make_figures as mf

out = Path("my_figures"); out.mkdir(exist_ok=True)
mf.S.use()                      # the paper's figure style (needs LaTeX)

mf.fig_prediction(out, {})      # Figure 1     -> pysr_pred_tau0_Ap.pdf
mf.fig_param_errors(out, {})    # Figure 2     -> param_error_bars.pdf
mf.fig_multid(out, {})          # Figures 3, 4 -> multid_bestworst.pdf, multid_3d.pdf
mf.fig_seed_band(out, {})       # Figure 5     -> seed_band.pdf
mf.fig_pareto(out, {})          # Figure 6     -> pareto_faithfulness.pdf
mf.fig_ns_budget(out, {})       # Figure 7     -> ns_budget_panel.pdf
```

The second argument receives the numbers the figure plots:

```python
values = {}
mf.fig_seed_band(out, values)
values["sobolev"]["ns"]         # [median, min, max, number of seeds]
# [0.2116, 0.1225, 0.2456, 5]
```

### What each function draws, and from what

| function | `--only` | paper figure | reads (under `results/paper_production_20260630_perz_sobolev_z2.6-4.2/`) | equation shown |
|---|---|---|---|---|
| `fig_prediction` | `prediction` | Fig. 1, `fig:tau0_ap_pred` | `sobolev/refit/z3.6/{refits,payloads}/{tau0,Ap}.pkl` | lowest-loss |
| `fig_param_errors` | `param_errors` | Fig. 2, `fig:param_error_bars` | `figures/table2_stats.tex` | lowest-loss |
| `fig_multid` | `multid` | Fig. 3, `fig:multid_bestworst`; Fig. 4, `fig:multid_3d` | `figures/multid_z3.6/multid_bestworst_perk.csv`, checked against `multid_bestworst.csv` (Table 5) | lowest-loss valid |
| `fig_seed_band` | `seed_band` | Fig. 5, `fig:seed_band` | `seed_band/seed_band_summary.json` | Pareto knee |
| `fig_pareto` | `pareto` | Fig. 6, `fig:pareto_faith` | `{value,sobolev}/refit/z3.6/grad_faith_*.csv`, `seed_band/z3.6_seed0_budget/refit/z3.6/grad_faith_ns.csv` | every valid candidate |
| `fig_ns_budget` | `ns_budget` | Fig. 7, `fig:ns_budget` | the three `grad_faith_ns.csv` files above | every valid candidate |

The selection rules are defined in `REPRODUCE.md`, "Selection rules and settings".

If an input file differs from the tagged one, the script stops with
`... differs from v1.2-paper; refusing to plot it.` That is deliberate: a figure drawn by
this script always shows the published numbers.

`notebooks/reproduce_paper.ipynb` calls the same functions and shows the figures inline.

## Everything else

The other scripts run fits, score them, or rebuild tables; `REPRODUCE.md` says which one
does what.
