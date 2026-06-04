#!/usr/bin/env python
"""Run the multi-z forecast pipeline from a YAML config."""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from priya_forecast.multi_z.config import load_config
from priya_forecast.multi_z.pipeline import run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    result = run(cfg)
    print(f"[multi_z] mode={cfg.mode} z in [{cfg.z_min},{cfg.z_max}] -> {cfg.output_dir}")
    if "table_path" in result:
        print(f"  table: {result['table_path']}")


if __name__ == "__main__":
    main()
