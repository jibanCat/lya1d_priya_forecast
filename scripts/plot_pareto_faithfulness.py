#!/usr/bin/env python
"""Render the per-parameter Pareto-faithfulness figure from cached CSVs.

Local / emulator-free. Each --series points at a dir of pareto_<param>.csv;
if a matching grad_faith_<param>.csv sidecar sits beside it (or in
--sidecar-dir), points are colored by derivative faithfulness, else gray.

Example (Phase-1 gray first cut, no cluster):
  PYTHONPATH=src python scripts/plot_pareto_faithfulness.py \
    --series value@20=results/single_z_stage6_log/refit/z3.6 \
    --series Sobolev@20=results/single_z_stage9/refit/z3.6 \
    --out results/single_z_stage_pareto_diag/pareto_faithfulness.png
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

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    render_grid(fronts_by_param, args.out, param_order=list(PARAM_NAMES))
    print(f"wrote {args.out}  ({len(fronts_by_param)} params)")


if __name__ == "__main__":
    main()
