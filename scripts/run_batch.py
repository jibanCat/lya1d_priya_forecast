#!/usr/bin/env python
"""Fan the single-z pipeline over all 13 z-bins.

gp_only / forecast_only  — loops the pipeline in-process, then aggregates.
refit_and_forecast       — two phases:
    --phase submit  : submit 13 SLURM array jobs (one per z-bin).
    --phase collect : forecast per z-bin from the refits, then aggregate.

    python scripts/run_batch.py --config configs/single_z/example.yaml
    python scripts/run_batch.py --config c.yaml --mode refit_and_forecast --phase submit
"""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from priya_forecast.single_z.config import PipelineConfig, load_config

Z_BINS_13 = [round(z, 1) for z in np.arange(2.2, 4.601, 0.2)]


def derive_z_configs(base: PipelineConfig) -> list[PipelineConfig]:
    """One PipelineConfig per z-bin: override redshift + output_dir."""
    base_out = base.output_dir.rstrip("/")
    derived = []
    for z in Z_BINS_13:
        c = dataclasses.replace(
            base, redshift=z, output_dir=f"{base_out}/z{z}",
        )
        derived.append(c)
    return derived


def run_inprocess(base: PipelineConfig) -> Path:
    """gp_only / forecast_only: run all 13 z-bins in-process, then aggregate."""
    from priya_forecast.single_z.pipeline import run
    from aggregate_z import aggregate  # type: ignore

    for cfg in derive_z_configs(base):
        print(f"[batch] z={cfg.redshift} ...", flush=True)
        run(cfg)
    out = aggregate(base_dir=base.output_dir.rstrip("/"), z_bins=Z_BINS_13)
    print(f"[batch] aggregated -> {out}")
    return out


def submit_slurm(base: PipelineConfig, repo: Path) -> None:
    """refit_and_forecast --phase submit: one SLURM array job per z-bin."""
    base_out = base.output_dir.rstrip("/")
    for z in Z_BINS_13:
        cmd = [
            "sbatch",
            f"--export=ALL,REPO={repo},BASEDIR={base.gp.basedir},"
            f"OUTPUT_DIR={base_out}/z{z},Z={z}",
            "--array=0-10",
            str(repo / "slurm" / "single_z_refit.slurm"),
        ]
        print("[batch submit]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--mode", default=None, help="Override the YAML mode.")
    p.add_argument("--phase", choices=["submit", "collect"], default=None,
                   help="refit_and_forecast only: submit the SLURM array, "
                        "or collect+forecast+aggregate.")
    args = p.parse_args()

    base = load_config(args.config)
    if args.mode is not None:
        base.mode = args.mode

    repo = Path(__file__).resolve().parent.parent
    if base.mode in ("gp_only", "forecast_only"):
        run_inprocess(base)
    elif base.mode == "refit_and_forecast":
        if args.phase == "submit":
            submit_slurm(base, repo)
        elif args.phase == "collect":
            run_inprocess(base)
        else:
            raise SystemExit(
                "refit_and_forecast needs --phase submit or --phase collect."
            )
    else:
        raise SystemExit(f"unknown mode {base.mode!r}.")


if __name__ == "__main__":
    main()
