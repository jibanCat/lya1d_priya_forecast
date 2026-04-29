"""User's Comment 4: refit a residual PySR on top of the 1D-product
combine of already-refit 1D equations, then forecast with
P = P_baseline + g_residual.

Reads the cached refits from `results/refit_target_subset/refits/*.pkl`
(produced by `scripts/refit_all_11_params.py`), trains a residual PySR
on the chosen subspace, and re-runs the 11D Fisher.

Run:
  PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia \\
  PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \\
      python scripts/run_residual_pysr.py \\
          --varying Ap ns tau0 dtau0 \\
          --refits-dir results/refit_target_subset/refits \\
          --output results/residual_4d
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(Path.home() / ".julia_env"))
os.environ.setdefault("JULIA_DEPOT_PATH", str(Path.home() / ".julia"))

_LYAEMU = Path("/home/mfho/student_projects/lya_emulator_full")
if _LYAEMU.exists():
    sys.path.insert(0, str(_LYAEMU))

from priya_forecast.data import load_eboss
from priya_forecast.fisher import fisher_matrix
from priya_forecast.likelihood import GaussianLikelihood
from priya_forecast.models.base import P1DModel
from priya_forecast.parameters import (
    PARAM_NAMES, PARAMS_11D, fiducial_vector,
)
from priya_forecast.refit_residual import fit_residual


class _BaselinePlusResidual(P1DModel):
    def __init__(self, *, gp, fid, refits, residual_fit, k_grid, z):
        self._gp, self._fid = gp, fid
        self._refits, self._residual_fit = refits, residual_fit
        self._k_grid, self._z = k_grid, z
        self._p_fid = gp.predict(fid, k_grid, z)

    def predict(self, theta, k, z):
        return self._residual_fit.predict_with_p_fid(theta, k, self._p_fid)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--varying", nargs="+", required=True)
    p.add_argument("--refits-dir", type=Path, required=True)
    p.add_argument("--n-train", type=int, default=128)
    p.add_argument("--niter", type=int, default=80)
    p.add_argument("--maxsize", type=int, default=20)
    p.add_argument("--z", type=float, default=3.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading GP...")
    from priya_forecast.models.gp_model import GPModel
    gp = GPModel()
    z = args.z
    k_eboss, _, _ = load_eboss(z=z)
    fid = np.array(fiducial_vector(), dtype=float)

    refits = {pn: None for pn in PARAM_NAMES}
    for pn in PARAM_NAMES:
        path = args.refits_dir / f"{pn}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                refits[pn] = pickle.load(f)
    print(f"Loaded {sum(r is not None for r in refits.values())} 1D refits.")

    print(f"\nFitting residual PySR on {args.varying}...")
    residual_fit = fit_residual(
        gp=gp, refits_1d=refits, varying_names=args.varying,
        z=z, k_grid=k_eboss, n_train=args.n_train, seed=args.seed,
        pysr_kwargs=dict(niterations=args.niter, maxsize=args.maxsize),
    )
    print(f"  residual: complexity={residual_fit.residual_complexity}, "
          f"loss={residual_fit.residual_loss:.3g}, "
          f"wall={residual_fit.wall_time_s:.0f}s")
    print(f"  equation: {residual_fit.residual_equation_str[:120]}")

    with open(args.output / "residual_fit.pkl", "wb") as f:
        pickle.dump(residual_fit, f)

    # 11D Fisher: GP, baseline-only (no residual), baseline+residual.
    print("\nRunning 11D Fisher comparisons...")
    lk_gp = GaussianLikelihood(model=gp, z=z, mock_data="gp", theta_fid=fid)
    fr_gp = fisher_matrix(likelihood=lk_gp, theta_fid=fid, params=PARAMS_11D,
                          step_frac=0.02, rel_tol=0.05, max_halvings=2)
    print("  GP done.")

    # Baseline-only (1D-product, no residual).
    from scripts.refit_all_11_params import _RefitMultiplicativeModel
    baseline = _RefitMultiplicativeModel(gp=gp, fid=fid, refits=refits, k_grid=k_eboss, z=z)
    lk_b = GaussianLikelihood(model=baseline, z=z, mock_data="gp", theta_fid=fid)
    fr_b = fisher_matrix(likelihood=lk_b, theta_fid=fid, params=PARAMS_11D,
                          step_frac=0.02, rel_tol=0.05, max_halvings=2)
    print("  baseline (1D-product) done.")

    # Baseline + residual.
    bpr = _BaselinePlusResidual(gp=gp, fid=fid, refits=refits,
                                 residual_fit=residual_fit, k_grid=k_eboss, z=z)
    lk_r = GaussianLikelihood(model=bpr, z=z, mock_data="gp", theta_fid=fid)
    fr_r = fisher_matrix(likelihood=lk_r, theta_fid=fid, params=PARAMS_11D,
                          step_frac=0.02, rel_tol=0.05, max_halvings=2)
    print("  baseline + residual done.")

    target = ("Ap", "ns", "tau0", "dtau0")
    lines = [
        "# Residual-PySR forecast scorecard\n",
        f"varying = {args.varying}, n_train = {args.n_train}, niter = {args.niter}, maxsize = {args.maxsize}",
        f"residual eq complexity = {residual_fit.residual_complexity}, "
        f"loss = {residual_fit.residual_loss:.3g}",
        "",
        "| param | GP σ | baseline σ | bsl/GP | base+resid σ | (b+r)/GP |",
        "|---|---|---|---|---|---|",
    ]
    for i, pname in enumerate(PARAM_NAMES):
        sg = fr_gp.sigma[i]
        sb = fr_b.sigma[i]
        sr = fr_r.sigma[i]
        lines.append(
            f"| {pname} | {sg:.3g} | {sb:.3g} | {sb/sg:.2f}× | {sr:.3g} | {sr/sg:.2f}× |"
        )
    lines.append("")
    lines.append(f"## Target subset {target}")
    for pname in target:
        i = PARAM_NAMES.index(pname)
        lines.append(f"  - **{pname}**: baseline {fr_b.sigma[i]/fr_gp.sigma[i]:.2f}×, "
                     f"base+residual {fr_r.sigma[i]/fr_gp.sigma[i]:.2f}×")

    md = "\n".join(lines) + "\n"
    (args.output / "scorecard.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
