"""End-to-end PySR HPO driver (the student's "tune this PySR run" tool).

Sample a Sobol sweep of the GP for one parameter at one z, split into
train / val, then sweep PySR hyperparameters and rank them. The goal is
to find the (niterations, maxsize, parsimony, ...) combination that
gives the smallest, most-accurate equation for the chosen target.

Designed to be PySR-agnostic: the same script also works on any
external (X, y) numpy file with `--data path/to.npz` (looks for arrays
named `X_train, y_train, X_val, y_val`). That makes the HPO module
reusable for the student's future projects, not just PRIYA P1D.

Run:

    PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env \\
    JULIA_DEPOT_PATH=$HOME/.julia \\
    PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \\
        python scripts/run_pysr_hpo.py \\
            --param ns --n-train 64 --n-val 256 \\
            --space configs/hpo/quick.yaml \\
            --strategy random --n-trials 6 \\
            --metric val_mse \\
            --output results/hpo_ns_$(date +%Y%m%d)/

PySR is slow — start with `quick.yaml` (6 trials × small budgets, ~10
minutes total). For paper-quality runs, swap in `full.yaml` and expect
hours.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(Path.home() / ".julia_env"))
os.environ.setdefault("JULIA_DEPOT_PATH", str(Path.home() / ".julia"))

_LYAEMU = Path("/home/mfho/student_projects/lya_emulator_full")
if _LYAEMU.exists():
    sys.path.insert(0, str(_LYAEMU))

from priya_forecast.config import load_hpo_config
from priya_forecast.data import load_eboss
from priya_forecast.parameters import (
    PARAM_NAMES,
    fiducial_vector,
    get_param,
)
from priya_forecast.pysr_hpo import (
    HPOSearchSpace,
    plot_hpo_results,
    run_hpo,
)


def _build_priya_dataset(*, gp, param: str, n_train: int, n_val: int, k_grid, z, seed=0):
    """Sobol sweep of one parameter at fixed-fiducial-rest. Returns
    (X_train, y_train, X_val, y_val) where X has columns (theta_norm, k_norm)
    and y is raw P_F."""
    from scipy.stats import qmc

    fid = np.array(fiducial_vector(), dtype=float)
    p = get_param(param)
    sampler = qmc.Sobol(d=1, seed=seed)
    n_total = n_train + n_val
    u = sampler.random(n=n_total).ravel()
    thetas = p.prior[0] + (p.prior[1] - p.prior[0]) * u

    flux = np.empty((n_total, k_grid.size))
    for i, t in enumerate(thetas):
        theta_full = fid.copy()
        theta_full[PARAM_NAMES.index(param)] = t
        flux[i] = gp.predict(theta_full, k_grid, z)

    k_min, k_max = float(k_grid.min()), float(k_grid.max())
    rows_X, rows_y = [], []
    for i in range(n_total):
        for ki, k in enumerate(k_grid):
            rows_X.append([(thetas[i] - p.prior[0]) / p.width(), (k - k_min) / (k_max - k_min)])
            rows_y.append(flux[i, ki])
    X = np.asarray(rows_X)
    y = np.asarray(rows_y)
    n_split = n_train * k_grid.size
    return X[:n_split], y[:n_split], X[n_split:], y[n_split:]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--param", help="PRIYA parameter to sweep (single param). "
                                   "Mutually exclusive with --data.")
    p.add_argument("--data", type=Path, help="Path to .npz with X_train/y_train/X_val/y_val.")
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--n-val", type=int, default=256)
    p.add_argument("--z", type=float, default=3.6)
    p.add_argument("--space", type=Path, required=True,
                   help="YAML defining the HPOSearchSpace (configs/hpo/{quick,full}.yaml).")
    p.add_argument("--strategy", choices=["grid", "random", "bayesian"], default="random")
    p.add_argument("--n-trials", type=int, default=6)
    p.add_argument("--metric", default="val_mse")
    p.add_argument("--target-loss", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    space_cfg = load_hpo_config(args.space)
    space = HPOSearchSpace(**space_cfg.space)

    if args.data is not None:
        d = np.load(args.data)
        X_tr, y_tr = d["X_train"], d["y_train"]
        X_va, y_va = d["X_val"], d["y_val"]
    else:
        if args.param is None:
            raise SystemExit("Must pass --param or --data.")
        from priya_forecast.models.gp_model import GPModel
        print("Loading real PRIYA GP emulator...")
        gp = GPModel()
        k_eboss, _, _ = load_eboss(z=args.z)
        print(f"Building Sobol training set for {args.param} at z={args.z}...")
        X_tr, y_tr, X_va, y_va = _build_priya_dataset(
            gp=gp, param=args.param, n_train=args.n_train,
            n_val=args.n_val, k_grid=k_eboss, z=args.z, seed=args.seed,
        )
        print(f"  X_train={X_tr.shape}, X_val={X_va.shape}")

    print(f"Running HPO: strategy={args.strategy}, n_trials={args.n_trials}, "
          f"metric={args.metric}")
    results = run_hpo(
        X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va,
        space=space, strategy=args.strategy, n_trials=args.n_trials,
        metric=args.metric, target_loss=args.target_loss,
        seed=args.seed, cache_dir=args.output / "cache",
    )

    plot_hpo_results(results, outdir=args.output, metric=args.metric,
                     target_loss=args.target_loss)

    # Markdown scorecard.
    lines = ["| rank | val_loss | wall_time | maxsize | niter | parsimony | best_expr |",
             "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(results[:10]):
        lines.append(
            f"| {i+1} | {r.val_loss:.3g} | {r.wall_time_s:.1f}s | "
            f"{r.config['maxsize']} | {r.config['niterations']} | "
            f"{r.config['parsimony']:.0e} | `{r.best_expression[:50]}...` |"
        )
    (args.output / "hpo_top10.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nFigures + cache + scorecard at {args.output}")


if __name__ == "__main__":
    main()
