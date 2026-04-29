"""Run the multi-D PySR diagnostic across all parameter pairs of the real GP.

Sweeps the 11 PRIYA parameters in the 2D_pairs regime and writes the
coupling-matrix heatmap that's the headline science output of the paper.
For each pair (θ_i, θ_j), it computes:

  coupling[i,j] = (MSE_1D_product − MSE_2D_joint) / MSE_1D_product

on a held-out 2D Sobol test set. Pairs with `coupling ≈ 0` mean the GP's
joint response in that subspace is well-approximated by the product of
1D fits. Pairs with `coupling ≫ 0` are where the paper's 1D-factorization
is leaving information on the table.

Backend: polynomial surrogate (fast, ~seconds per pair). Total runtime
on the real GP for all 55 pairs of 11 params is ~minutes.

Usage:
  PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \\
      python scripts/run_coupling_matrix.py \\
          --params dtau0 tau0 ns Ap herei heref alphaq hub omegamh2 hireionz bhfeedback \\
          --order 4 --n-train 96 --n-test 256 \\
          --output results/coupling_matrix/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_LYAEMU = Path("/home/mfho/student_projects/lya_emulator_full")
if _LYAEMU.exists():
    sys.path.insert(0, str(_LYAEMU))

from priya_forecast.data import load_eboss
from priya_forecast.multid_diagnostic import plot_diagnostic, run_diagnostic
from priya_forecast.parameters import PARAM_NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params", nargs="+", default=list(PARAM_NAMES))
    p.add_argument("--z", type=float, default=3.6)
    p.add_argument("--n-train", type=int, default=96)
    p.add_argument("--n-test", type=int, default=256)
    p.add_argument("--order", type=int, default=4)
    p.add_argument("--regimes", nargs="+", default=["1D", "2D_pairs", "full_kD"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading real PRIYA GP emulator...")
    from priya_forecast.models.gp_model import GPModel
    gp = GPModel()
    k_eboss, _, _ = load_eboss(z=args.z)

    n_pairs = len(args.params) * (len(args.params) - 1) // 2
    print(f"Running diagnostic across {len(args.params)} params at z={args.z}.")
    print(f"  1D       : {len(args.params)} fits")
    print(f"  2D_pairs : {n_pairs} fits")
    print(f"  full_kD  : 1 fit")

    results_by_regime: dict[str, list] = {}
    for regime in args.regimes:
        print(f"\n--- {regime} ---")
        out = run_diagnostic(
            gp_model=gp, z=args.z, k_grid=k_eboss,
            param_names=args.params, regime=regime,
            n_train=args.n_train, n_test=args.n_test,
            poly_order=args.order, seed=0,
        )
        results_by_regime[regime] = out
        if regime == "2D_pairs":
            # Print the table sorted by coupling.
            ranked = sorted(out, key=lambda r: -r.extra.get("coupling", 0))
            print(f"  pair                       coupling     1D-prod MSE   2D-joint MSE")
            for r in ranked:
                a, b = r.param_names
                cpl = r.extra["coupling"]
                m1 = r.extra["mse_1D_product"]
                m2 = r.extra["mse_2D_joint"]
                print(f"  {a:>10} × {b:<10}    {cpl:+.3f}     {m1:.3e}     {m2:.3e}")

    plot_diagnostic(results_by_regime, outdir=args.output)
    print(f"\nFigures written to {args.output}/")
    print(f"  diag1_scaling.png        — test MSE vs n_params")
    print(f"  diag2_walltime.png       — fit cost vs n_params")
    print(f"  diag3_coupling_matrix.png — the coupling matrix (the headline)")


if __name__ == "__main__":
    main()
