#!/usr/bin/env python
"""Refit one (parameter, z-bin) with single-z PySR; write its Pareto CSV.

One SLURM array task runs one parameter. Used by slurm/single_z_refit.slurm.

    python scripts/refit_one_param_single_z.py --param ns --z 3.6 \\
        --basedir data/kodiaq_gp --output-dir results/single_z_run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from priya_forecast.parameters import PARAM_NAMES, PARAMS_11D
from priya_forecast.single_z.config import PipelineConfig, GPConfig, PySRConfig
from priya_forecast.single_z import refit as _refit


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--param", required=True, choices=list(PARAM_NAMES))
    p.add_argument("--z", type=float, required=True)
    p.add_argument("--basedir", default="data/kodiaq_gp")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--kmin", type=float, default=0.001)
    p.add_argument("--kmax", type=float, default=0.04)
    p.add_argument("--niterations", type=int, default=50)
    p.add_argument("--maxsize", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from priya_forecast.models.gp_model import GPModel

    cfg = PipelineConfig(
        mode="refit_and_forecast", redshift=args.z,
        output_dir=args.output_dir, gp=GPConfig(basedir=args.basedir),
        pysr=PySRConfig(niterations=args.niterations, maxsize=args.maxsize,
                        seed=args.seed),
    )
    k_grid = _refit.kodiaq_k_grid(args.kmin, args.kmax, 48)
    refit_dir = Path(args.output_dir) / "refit" / f"z{args.z}"
    fid = np.array([pp.fid for pp in PARAMS_11D], dtype=float)

    print(f"Loading emulators from {args.basedir} ...", flush=True)
    t0 = time.time()
    gp_lf = GPModel(basedir=args.basedir, fidelity="lf", kf=k_grid)
    gp_hf = GPModel(basedir=args.basedir, fidelity="hf", kf=k_grid)
    _ = gp_lf.predict(fid, k_grid, args.z)
    _ = gp_hf.predict(fid, k_grid, args.z)
    print(f"  loaded in {time.time() - t0:.0f}s.", flush=True)

    t0 = time.time()
    result = _refit.refit_one_param_single_z(
        param_name=args.param, z=args.z, cfg=cfg,
        gp_lf=gp_lf, gp_hf=gp_hf, k_grid=k_grid, out_dir=refit_dir,
    )
    print(f"[{time.time() - t0:.0f}s] {args.param} z={args.z} -> "
          f"{refit_dir}/pareto_{args.param}.csv "
          f"(eq complexity={result.pareto_complexity}, "
          f"loss={result.pareto_loss:.4g})", flush=True)


if __name__ == "__main__":
    main()
