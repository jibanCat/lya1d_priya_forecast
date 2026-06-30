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
    p.add_argument("--populations", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-space", choices=("linear", "log"), default="linear")
    p.add_argument("--use-sobolev", action="store_true")
    p.add_argument("--sobolev-lambda", type=float, default=1.0)
    p.add_argument(
        "--anova-loss", action="store_true",
        help="Use the dimension-balanced ANOVA loss (ablation only). Default "
             "OFF: the value baseline trains on plain MSE.",
    )
    args = p.parse_args()

    # Guard: Sobolev matches d(logP)/dtheta, so a linear-P target silently
    # mismatches the gradient. Fail loud rather than corrupt the fit.
    if args.use_sobolev and args.target_space != "log":
        p.error("--use-sobolev requires --target-space log "
                "(the Sobolev loss matches d(logP)/dtheta).")
    if args.use_sobolev and args.anova_loss:
        p.error("--use-sobolev and --anova-loss are mutually exclusive "
                "(Sobolev overrides the training loss).")

    from priya_forecast.models.gp_model import GPModel

    cfg = PipelineConfig(
        mode="refit_and_forecast", redshift=args.z,
        output_dir=args.output_dir, gp=GPConfig(basedir=args.basedir),
        pysr=PySRConfig(niterations=args.niterations, maxsize=args.maxsize,
                        populations=args.populations, seed=args.seed,
                        use_sobolev=args.use_sobolev,
                        sobolev_lambda=args.sobolev_lambda,
                        use_anova_loss=args.anova_loss),
        target_space=args.target_space,
    )
    cfg.validate()
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
