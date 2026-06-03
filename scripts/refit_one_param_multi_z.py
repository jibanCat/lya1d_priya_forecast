#!/usr/bin/env python
"""Refit one parameter over [z_min, z_max] with multi-z PySR; write CSV+norm."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from priya_forecast.parameters import PARAM_NAMES, PARAMS_11D
from priya_forecast.multi_z.config import MultiZPipelineConfig
from priya_forecast.single_z.config import GPConfig, PySRConfig
from priya_forecast.single_z.refit import kodiaq_k_grid
from priya_forecast.multi_z import refit as _refit


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--param", required=True, choices=list(PARAM_NAMES))
    p.add_argument("--z-min", type=float, required=True)
    p.add_argument("--z-max", type=float, required=True)
    p.add_argument("--basedir", default="data/kodiaq_gp")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--kmin", type=float, default=0.001)
    p.add_argument("--kmax", type=float, default=0.04)
    p.add_argument("--n-total", type=int, default=225)
    p.add_argument("--niterations", type=int, default=50)
    p.add_argument("--maxsize", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from priya_forecast.models.gp_model import GPModel
    cfg = MultiZPipelineConfig(
        mode="refit_and_forecast", z_min=args.z_min, z_max=args.z_max,
        output_dir=args.output_dir, gp=GPConfig(basedir=args.basedir),
        pysr=PySRConfig(niterations=args.niterations, maxsize=args.maxsize,
                        seed=args.seed),
    )
    k_grid = kodiaq_k_grid(args.kmin, args.kmax, 48)
    refit_dir = Path(args.output_dir) / "refit" / f"z{args.z_min}-{args.z_max}"
    fid = np.array([pp.fid for pp in PARAMS_11D], dtype=float)

    print(f"Loading emulators from {args.basedir} ...", flush=True)
    t0 = time.time()
    gp_lf = GPModel(basedir=args.basedir, fidelity="lf", kf=k_grid)
    gp_hf = GPModel(basedir=args.basedir, fidelity="hf", kf=k_grid)
    _ = gp_lf.predict(fid, k_grid, args.z_min)
    print(f"  loaded in {time.time()-t0:.0f}s.", flush=True)

    t0 = time.time()
    result = _refit.refit_one_param_multi_z(
        param_name=args.param, z_min=args.z_min, z_max=args.z_max, cfg=cfg,
        gp_lf=gp_lf, gp_hf=gp_hf, k_grid=k_grid, out_dir=refit_dir,
        n_total=args.n_total,
    )
    print(f"[{time.time()-t0:.0f}s] {args.param} z in [{args.z_min},{args.z_max}] "
          f"-> {refit_dir}/pareto_{args.param}.csv "
          f"(complexity={result.pareto_complexity}, loss={result.pareto_loss:.4g})",
          flush=True)


if __name__ == "__main__":
    main()
