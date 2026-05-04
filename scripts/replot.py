"""Lightweight local replot — regenerate paper figures from cached results.

Designed to run on a laptop **without** PySR / Julia / lyaemu / GPy.
Reads the cached `refits/*.pkl` and `fisher.npz` from a results dir
and re-renders all the paper figures + markdown deliverables.

What works locally:
  - per_param_summary.md (full equations, prettified)
  - resolution_correction.{md,json,grid_*.{png,pdf}}  — HF/LF ratio at fid
  - resolution_correction_equations.md                — symbolic expressions
  - resolution_correction_param_variation_*.{png,pdf} — R(k; θ) per quantile
  - corner.{png,pdf}                                  — multi-z Fisher overlay

What does NOT work locally (needs the GP emulator on the cluster):
  - holdout_validation_*.{png,pdf}                    — needs gp_lf, gp_hf

Run:
  PYTHONPATH=src python scripts/replot.py \\
      --results-dir results/refit_optionC_z2.6-4.2

Dependencies: numpy, scipy, matplotlib, sympy. NOT pysr/julia/lyaemu.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, required=True,
                   help="A `refit_*.py` / `multi_z_aggregate.py` output dir "
                        "containing `refits/*.pkl` and (optionally) `fisher.npz`.")
    args = p.parse_args()

    refits_dir = args.results_dir / "refits"
    if not refits_dir.exists():
        raise SystemExit(f"refits dir not found: {refits_dir}")

    # Make our package importable. Don't use 'from priya_forecast.X import ...'
    # because some of those modules pull in pysr lazily.
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))

    from priya_forecast.deliverables import (
        per_param_summary_lines,
        write_param_variation_resolution_correction,
        write_resolution_correction_equations,
        write_resolution_correction_outputs,
    )
    from priya_forecast.parameters import PARAM_NAMES, fiducial_vector

    # Load refits.
    refits = {pn: None for pn in PARAM_NAMES}
    for pn in PARAM_NAMES:
        path = refits_dir / f"{pn}.pkl"
        if path.exists():
            with open(path, "rb") as fh:
                refits[pn] = pickle.load(fh)
    refits_loaded = {pn: r for pn, r in refits.items() if r is not None}
    n_loaded = len(refits_loaded)
    print(f"Loaded {n_loaded} refits from {refits_dir}")
    if n_loaded == 0:
        raise SystemExit("No refits.")

    # Pull k_grid from any one refit.
    sample = next(iter(refits_loaded.values()))
    k_grid = np.asarray(sample.k_grid, dtype=float)
    fid = np.array(fiducial_vector(), dtype=float)

    # Multi-z metadata (if applicable) for z_eval default.
    z_eval = None
    if getattr(sample, "is_multiz", False):
        z_eval = float((sample.z_min + sample.z_max) / 2.0)

    # 1. per_param_summary.md (with prettified eqs).
    summary_lines = per_param_summary_lines(refits_loaded, header_z=sample.z)
    (args.results_dir / "per_param_summary.md").write_text(
        "\n".join(summary_lines) + "\n"
    )
    print("  → per_param_summary.md")

    # 2. resolution_correction.{md,json,grid_*.{png,pdf}}.
    write_resolution_correction_outputs(
        refits_loaded, k_grid, fid, args.results_dir, z_eval=z_eval,
    )
    print("  → resolution_correction.{md,json,grid_*.{png,pdf}}")

    # 3. resolution_correction_equations.md
    write_resolution_correction_equations(refits_loaded, args.results_dir)
    print("  → resolution_correction_equations.md")

    # 4. param-variation grid (R(k; θ) at quantiles).
    write_param_variation_resolution_correction(
        refits_loaded, k_grid, args.results_dir, z_eval=z_eval,
    )
    print("  → resolution_correction_param_variation_{cosmo,astro}.{png,pdf}")

    # 5. corner plot from fisher.npz, if present.
    fisher_path = args.results_dir / "fisher.npz"
    if fisher_path.exists():
        from priya_forecast.plotting import plot_fisher_corner
        from priya_forecast.parameters import PARAMS_11D
        data = np.load(fisher_path, allow_pickle=True)
        param_names = list(data["param_names"])
        params_in_fisher = tuple(p for p in PARAMS_11D if p.name in param_names)
        # Re-order to match the npz.
        params_in_fisher = sorted(
            params_in_fisher,
            key=lambda p: param_names.index(p.name),
        )
        params_in_fisher = tuple(params_in_fisher)

        class _FR:
            def __init__(self, sigma, cov, theta_fid, names):
                self.sigma = sigma
                self.cov = cov
                self.theta_fid = theta_fid
                self.param_names = tuple(names)

        # Backwards compat: older fisher.npz may have `theta_fid` key
        # instead of `theta_fid_subset`.
        if "theta_fid_subset" in data.files:
            theta_fid_arr = data["theta_fid_subset"]
        elif "theta_fid" in data.files:
            theta_fid_arr = data["theta_fid"]
        else:
            theta_fid_arr = np.array([fid[PARAM_NAMES.index(n)] for n in param_names])
        fr_gp = _FR(data["sigma_gp"], data["cov_gp"], theta_fid_arr, param_names)
        fr_hy = _FR(data["sigma_hybrid"], data["cov_hybrid"], theta_fid_arr, param_names)
        for ext in ("pdf", "png"):
            plot_fisher_corner(
                fr_gp=fr_gp, fr_hybrid=fr_hy, params=params_in_fisher,
                output_path=args.results_dir / f"corner.{ext}",
                title=f"GP (black) vs hybrid (red) — {len(param_names)}D Fisher",
            )
        print(f"  → corner.{{pdf,png}} ({len(param_names)} params)")
    else:
        print("  (no fisher.npz, skipping corner plot)")

    print("\nDone. Hold-out validation plots require the GP emulator and "
          "are NOT regenerated here — keep the pre-existing "
          "`holdout_validation_*.{png,pdf}` files from the cluster run.")


if __name__ == "__main__":
    main()
