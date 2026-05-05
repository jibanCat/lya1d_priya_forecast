"""Phase 1 of option C: pre-compute all 11 multi-z 1pvar payloads on one node.

Saves `payloads/<param>.pkl` for each PRIYA param. Each pkl is a dict
ready to feed `_build_training_matrix_multiz` — the heavy GP work
(loading both LF and HF emulators + ~5,000 GP predicts) is done ONCE
here, then the parallel `refit_one_param.py` jobs each load one tiny pkl
and run only PySR.

Run:
  PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia \\
  PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \\
      python scripts/precompute_payloads.py \\
          --z-min 2.6 --z-max 4.2 --n-total 225 \\
          --output results/refit_kodiaq_multiz_2.6-4.2/payloads
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(Path.home() / ".julia_env"))
os.environ.setdefault("JULIA_DEPOT_PATH", str(Path.home() / ".julia"))

_LYAEMU = Path("/home/mfho/student_projects/lya_emulator_full")
if _LYAEMU.exists():
    sys.path.insert(0, str(_LYAEMU))

from priya_forecast.parameters import PARAM_NAMES, fiducial_vector
from priya_forecast.refit_1d_pysr import (
    _generate_1pvar_multiz_inline,
    compute_local_normalization_multiz,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--z-min", type=float, default=2.6)
    p.add_argument("--z-max", type=float, default=4.2)
    p.add_argument("--k-min", type=float, default=0.005)
    p.add_argument("--k-max", type=float, default=0.064)
    p.add_argument("--n-k", type=int, default=32)
    p.add_argument("--n-total", type=int, default=225,
                   help="Sobol points per param.")
    p.add_argument("--seed", type=int, default=0,
                   help="Sobol seed (deterministic across params; pysr seed differs).")
    p.add_argument("--basedir", type=Path,
                   default=Path("/nfs/turbo/umor-yueyingn/mfho/birdgroup/"
                                "lya_xq100/kodiaq_2_2_4_6-48-48"))
    p.add_argument("--params", nargs="+", default=list(PARAM_NAMES))
    p.add_argument("--output", type=Path, required=True,
                   help="Directory to write payloads/<param>.pkl files.")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    k_grid = np.linspace(args.k_min, args.k_max, args.n_k)
    z_grid_kodiaq = np.array([2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])
    z_grid_in_range = z_grid_kodiaq[
        (z_grid_kodiaq >= args.z_min - 1e-6) & (z_grid_kodiaq <= args.z_max + 1e-6)
    ]
    fid = np.array(fiducial_vector(), dtype=float)

    print(f"Loading kodiaq emulators (LF + HF) at {args.basedir} ...")
    from priya_forecast.models.gp_model import GPModel
    t0 = time.time()
    gp_lf = GPModel(basedir=args.basedir, fidelity="lf", kf=k_grid)
    gp_hf = GPModel(basedir=args.basedir, fidelity="hf", kf=k_grid)
    # warm both: trigger lazy load so first per-param call doesn't pay the cost.
    _ = gp_lf.predict(fid, k_grid, 3.6)
    _ = gp_hf.predict(fid, k_grid, 3.6)
    print(f"  emulators loaded in {time.time()-t0:.0f}s.")

    print(f"\nGenerating multi-z 1pvar payloads for {len(args.params)} params:")
    print(f"  z=[{args.z_min}, {args.z_max}] (snap to {z_grid_in_range.tolist()})")
    print(f"  k=linspace({args.k_min}, {args.k_max}, {args.n_k}) s/km")
    print(f"  n_total={args.n_total} Sobol per param.")

    for pname in args.params:
        out_path = args.output / f"{pname}.pkl"
        if out_path.exists():
            print(f"  [skip-cache] {pname} → {out_path}")
            continue
        t0 = time.time()
        payload = _generate_1pvar_multiz_inline(
            gp_lf=gp_lf, gp_hf=gp_hf, param_name=pname,
            z_min=args.z_min, z_max=args.z_max,
            k_grid=k_grid, n_total=args.n_total, seed=args.seed,
        )
        # Per-z, at-fid-anchored normalization (computed once here so
        # the per-param SLURM job doesn't need the GP again).
        from priya_forecast.parameters import get_param
        p_meta = get_param(pname)
        norm = compute_local_normalization_multiz(
            flux_lf_z=payload["flux_lf_z"], z_per_row=payload["z_per_row"],
            z_grid=z_grid_in_range, k_grid=k_grid,
            gp_lf=gp_lf, fid=fid,
            param_min=float(p_meta.prior[0]),
            param_max=float(p_meta.prior[1]),
        )
        bundle = dict(
            param_name=pname,
            payload=payload,
            norm=norm,
            k_grid=k_grid,
            z_min=float(args.z_min), z_max=float(args.z_max),
            z_grid_in_range=z_grid_in_range,
        )
        with open(out_path, "wb") as fh:
            pickle.dump(bundle, fh)
        print(f"  [{time.time()-t0:.1f}s] {pname} → {out_path}", flush=True)

    print("\nAll payloads written. Submit refit_one_param.py jobs now.")


if __name__ == "__main__":
    main()
