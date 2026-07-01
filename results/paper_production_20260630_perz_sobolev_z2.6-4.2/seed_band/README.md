# Across-seed band — schema + provenance

**code git** `7aa26af` · **run-id** `prod-20260630-perz-sobolev` · 2026-06-30.
Produced at the same git as the rest of this run (see `../README.md` + `../RUN_MANIFEST.md`).

## What this is
`z3.6_seed{0..4}_{value,sobolev,budget}/refit/z3.6/` hold the 5-seed re-fits at z=3.6
(same `pareto_<p>.csv` + `grad_faith_<p>.csv` schema as `../sobolev/README.md`). They are
aggregated into **`seed_band_summary.json`**, the file the paper's seed-band figure
(`fig:seed_band`) and `priya_forecast.paper_figures.plot_seed_band` read.

## `seed_band_summary.json` schema
```
{
  "gate":  0.25,                     # grad_err faithfulness threshold used to classify
  "seeds": [0, 1, 2, 3, 4],          # the 5 PySR random seeds aggregated
  "params": {                        # one entry per PRIYA parameter
    "<param>": {
      "value":   [median, min, max, n_seeds],   # knee grad_err across seeds, value loss
      "sobolev": [median, min, max, n_seeds],   # knee grad_err across seeds, Sobolev loss
      "flips":   "stable" | ...                  # whether the pass/fail verdict is seed-stable
    }, ...
  },
  "ns_budget35": {                   # ns value-loss control at maxsize=35 across seeds
    "median": ..., "min": ..., "max": ..., "n": 5, "verdict": "MIXED"
  }
}
```
`value`/`sobolev` are `[median, min, max, n]` of the **Pareto-knee** `grad_err` over the 5
seeds (knee = lowest complexity within 10% of best loss; `grad_faith_io.knee_row`). A
parameter is seed-robustly faithful when its `sobolev` median **and** max are ≤ `gate`.

## How to read it
```python
import json
sb = json.load(open("seed_band_summary.json"))
sb["params"]["ns"]["sobolev"]      # -> [median, min, max, n] e.g. [0.212, 0.123, 0.246, 5]
```
Or via the reusable module: `pf.plot_seed_band(pf.load_run())` (see `notebooks/figures_tutorial.ipynb`).

## How to reproduce
1. Re-run the 5-seed fits (Tier-3, SLURM) via `scripts/submit_paper_production.sh` (seeds 0–4 at z=3.6).
2. Aggregate:
   ```bash
   scripts/aggregate_seed_band.py \
       --band-dir results/paper_production_20260630_perz_sobolev_z2.6-4.2/seed_band \
       --out      results/paper_production_20260630_perz_sobolev_z2.6-4.2/seed_band/seed_band_summary.json
   ```
`seed_band_summary.json` carries a top-level `"git"` field (the producing code hash;
convention `priya_forecast.provenance.git_stamp`). The per-seed `grad_faith_<p>.csv`
sidecars carry their own `# param= z= tol= log_space= git= source=` provenance header;
see `../sobolev/README.md` for the CSV column schema.
