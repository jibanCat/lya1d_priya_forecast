#!/usr/bin/env python
"""Render the per-parameter Pareto-faithfulness figure from cached CSVs.

Local / emulator-free. Each --series points at a dir of pareto_<param>.csv;
if a matching grad_faith_<param>.csv sidecar sits beside it (or in
--sidecar-dir), points are colored by derivative faithfulness, else gray.

Example (Phase-1 gray first cut, no cluster):
  PYTHONPATH=src python scripts/plot_pareto_faithfulness.py \
    --series value@20=results/paper_production_20260630_perz_sobolev_z2.6-4.2/value/refit/z3.6 \
    --series Sobolev@20=results/paper_production_20260630_perz_sobolev_z2.6-4.2/sobolev/refit/z3.6 \
    --out results/_repro_scratch/pareto_faithfulness.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

from priya_forecast.parameters import PARAM_NAMES
from priya_forecast.pareto_diag import load_front, render_grid

_MARKERS = ["o", "s", "^", "D", "v"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", action="append", required=True,
                    help="LABEL=PARETO_DIR (repeatable)")
    ap.add_argument("--sidecar-dir", action="append", default=None,
                    help="optional LABEL=SIDECAR_DIR overrides (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--y-col", default="auto",
                    choices=["auto", "value_mse", "Loss"],
                    help="y-axis column: 'value_mse' (common, comparable), "
                         "'Loss' (raw PySR loss), or 'auto' (value_mse if any "
                         "sidecar present, else Loss)")
    args = ap.parse_args()

    sidecar_override = {}
    for s in (args.sidecar_dir or []):
        label, d = s.split("=", 1)
        sidecar_override[label] = Path(d)

    series_specs = []
    for i, s in enumerate(args.series):
        label, d = s.split("=", 1)
        series_specs.append((label, Path(d), _MARKERS[i % len(_MARKERS)]))

    fronts_by_param = {}
    for param in PARAM_NAMES:
        rows = []
        for label, pareto_dir, marker in series_specs:
            pareto = pareto_dir / f"pareto_{param}.csv"
            if not pareto.exists():
                continue
            sdir = sidecar_override.get(label, pareto_dir)
            sidecar = sdir / f"grad_faith_{param}.csv"
            rows.append({
                "front": load_front(pareto, sidecar if sidecar.exists() else None),
                "label": label, "marker": marker,
            })
        if rows:
            fronts_by_param[param] = rows

    # Resolve y-axis column. value_mse is the honest, cross-objective-comparable
    # axis; fall back to raw PySR Loss only when no sidecar carries value_mse.
    y_col = args.y_col
    if y_col == "auto":
        has_vmse = any(
            r["front"]["value_mse"].notna().any()
            for rows in fronts_by_param.values() for r in rows
        )
        y_col = "value_mse" if has_vmse else "Loss"
    y_label = ("value MSE vs GP (logP, HF)" if y_col == "value_mse"
               else "PySR training loss")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    render_grid(fronts_by_param, args.out, param_order=list(PARAM_NAMES),
                y_col=y_col, y_label=y_label)
    print(f"wrote {args.out}  ({len(fronts_by_param)} params, y={y_col})")


if __name__ == "__main__":
    main()
