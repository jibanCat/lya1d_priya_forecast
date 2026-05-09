"""Single-z forecast pipeline entry point.

Usage:
    python scripts/run_pipeline.py --config configs/single_z/example.yaml

Reads one YAML, dispatches to the right mode (gp_only / forecast_only /
refit_and_forecast), writes everything into ``cfg.output_dir``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from priya_forecast.single_z.config import load_config
from priya_forecast.single_z.pipeline import run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path,
                   help="Path to a single-z pipeline YAML.")
    p.add_argument("--mode", type=str, default=None,
                   help="Override mode in the YAML (gp_only / forecast_only / refit_and_forecast).")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override output_dir in the YAML.")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.mode is not None:
        cfg.mode = args.mode
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    result = run(cfg)
    print(f"OK mode={cfg.mode} z={cfg.redshift} out={cfg.output_dir}")
    table = result.get("table_path")
    if table is not None:
        print(f"  wrote {table}")
    sc = result.get("scorecard_path")
    if sc is not None:
        print(f"  wrote {sc}")


if __name__ == "__main__":
    main()
