#!/usr/bin/env python
"""Regenerate per-parameter LF/HF 1pvar training data from the emulator.

Replaces the legacy `InferenceLyaData/1pvar/` HDF5s (Martin Fernandez's
k-range) with data sampled from the kodiaq-squad emulator at the GP basedir,
so the PySR refit trains on the same k-grid the forecast scores against.

Writes raw P_F (not k·P/π) to
`<output>/{lf,hf}_<param>_npoints50.hdf5` for all 11 params and 13 z-bins.

    python scripts/regen_1pvar.py --basedir data/kodiaq_gp \\
        --output data/single_z_1pvar
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from priya_forecast.parameters import PARAM_NAMES, fiducial_vector
from priya_forecast.single_z.training_data import regenerate_param, write_1pvar_hdf5

# The 13 kodiaq z-bins — the emulator's trained_mf/zbin* grid, increasing.
# NOT the stale 9-bin `z_grid_kodiaq` constant in refit_1d_pysr.py.
Z_GRID_13 = np.round(np.arange(2.2, 4.601, 0.2), 1)


def kodiaq_k_grid(kmin: float, kmax: float, nk: int) -> np.ndarray:
    """Log-spaced k-grid (s/km) the refit trains on."""
    return np.geomspace(kmin, kmax, nk)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--basedir", default="data/kodiaq_gp",
                   help="GP emulator basedir (cfg.gp.basedir).")
    p.add_argument("--output", default="data/single_z_1pvar",
                   help="Directory for the regenerated HDF5s.")
    p.add_argument("--kmin", type=float, default=0.001,
                   help="Min k (s/km) of the training grid.")
    p.add_argument("--kmax", type=float, default=0.04,
                   help="Max k (s/km) of the training grid.")
    p.add_argument("--nk", type=int, default=48,
                   help="Number of log-spaced k points.")
    p.add_argument("--params", nargs="+", default=list(PARAM_NAMES),
                   help="Subset of parameters (default: all 11).")
    args = p.parse_args()

    from priya_forecast.models.gp_model import GPModel

    k_grid = kodiaq_k_grid(args.kmin, args.kmax, args.nk)
    out_dir = Path(args.output)
    fid = np.asarray(fiducial_vector(), dtype=float)

    print(f"Loading LF + HF emulators from {args.basedir} ...")
    t0 = time.time()
    gp_lf = GPModel(basedir=args.basedir, fidelity="lf", kf=k_grid)
    gp_hf = GPModel(basedir=args.basedir, fidelity="hf", kf=k_grid)
    _ = gp_lf.predict(fid, k_grid, 3.6)  # warm the lazy load
    _ = gp_hf.predict(fid, k_grid, 3.6)
    print(f"  loaded in {time.time() - t0:.0f}s.")

    for pname in args.params:
        t0 = time.time()
        gen = regenerate_param(
            gp_lf=gp_lf, gp_hf=gp_hf, param_name=pname,
            z_grid=Z_GRID_13, k_grid=k_grid,
        )
        for fidelity in ("lf", "hf"):
            write_1pvar_hdf5(
                out_dir / f"{fidelity}_{pname}_npoints50.hdf5",
                params=gen[f"params_{fidelity}"],
                kfkms=gen[f"kfkms_{fidelity}"],
                flux_vectors=gen[f"flux_{fidelity}"],
                zout=gen["zout"],
            )
        print(f"  [{time.time() - t0:.1f}s] {pname} -> "
              f"{out_dir}/lf_{pname}_npoints50.hdf5, "
              f"{out_dir}/hf_{pname}_npoints50.hdf5", flush=True)

    print(f"Done. {len(args.params)} params x 2 fidelities written to {out_dir}.")


if __name__ == "__main__":
    main()
