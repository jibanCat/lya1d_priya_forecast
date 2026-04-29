"""Refit a 1D PySR equation for every PRIYA parameter, then run the full
11D Fisher forecast on the multiplicative combine of those equations.

This is the "Original Design" the user asked for, but with our own
freshly-trained equations rather than the published ones (which had a
broken alphaq among other issues). Architecture per the user's
specification:
  - Train PySR on flux_norm = (P_F - mean_k(k)) / std_k(k).
  - Predict in raw P_F via the bundled NormalizationSpec.
  - Combine multiplicatively: P(theta, k) = P_fid · ∏ f_i(θ_i)/f_i(fid).
  - Fisher forecast on all 11 params jointly.

Run:
  PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia \\
  PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \\
      python scripts/refit_all_11_params.py \\
          --niter 100 --maxsize 25 --n-train 128 \\
          --output results/refit_all_11D
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

from priya_forecast.data import load_eboss
from priya_forecast.fisher import fisher_matrix
from priya_forecast.likelihood import GaussianLikelihood
from priya_forecast.models.base import P1DModel
from priya_forecast.parameters import (
    PARAM_NAMES,
    PARAMS_11D,
    fiducial_vector,
)
from priya_forecast.refit_1d_pysr import refit_1d_for_param


class _RefitMultiplicativeModel(P1DModel):
    """11D forward model: each f_i comes from refit_1d_for_param's
    Refit1DResult (which already does the flux_norm round-trip)."""

    def __init__(self, *, gp, fid, refits: dict, k_grid: np.ndarray, z: float):
        self._gp = gp
        self._fid = fid
        self._refits = refits  # dict: param_name -> Refit1DResult or None (use GP slice)
        self._k_grid = k_grid
        self._z = z
        self._p_fid = gp.predict(fid, k_grid, z)
        # Cache f_i_at_fid per param to avoid recomputing every predict.
        self._f_fid_cache = {}
        for pname, r in refits.items():
            if r is None:
                continue
            self._f_fid_cache[pname] = r.predict(theta_phys=r.fid_value, k=k_grid)

    def predict(self, theta: np.ndarray, k: np.ndarray, z: float) -> np.ndarray:
        if not np.allclose(k, self._k_grid):
            raise ValueError("This model is built for a fixed k-grid.")
        out = self._p_fid.copy()
        for pname in PARAM_NAMES:
            i = PARAM_NAMES.index(pname)
            r = self._refits.get(pname)
            if r is not None:
                num = r.predict(theta_phys=float(theta[i]), k=k)
                den = self._f_fid_cache[pname]
            else:
                t_only = self._fid.copy()
                t_only[i] = theta[i]
                num = self._gp.predict(t_only, k, z)
                den = self._p_fid
            with np.errstate(divide="ignore", invalid="ignore"):
                out = out * (num / den)
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--niter", type=int, default=100)
    p.add_argument("--maxsize", type=int, default=25)
    p.add_argument("--n-train", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--z", type=float, default=3.6)
    p.add_argument("--params", nargs="+", default=list(PARAM_NAMES),
                   help="Subset of params to refit (others use GP-slice).")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output / "refits"
    cache_dir.mkdir(exist_ok=True)

    print("Loading real PRIYA GP emulator...")
    from priya_forecast.models.gp_model import GPModel
    gp = GPModel()
    z = args.z
    k_eboss, _, _ = load_eboss(z=z)
    fid = np.array(fiducial_vector(), dtype=float)

    refits: dict = {pn: None for pn in PARAM_NAMES}
    pysr_kwargs = dict(niterations=args.niter, maxsize=args.maxsize)

    print(f"Refitting 1D PySR for {len(args.params)} params (others = GP-slice)...")
    print(f"  per-param config: niter={args.niter}, maxsize={args.maxsize}, "
          f"n_train={args.n_train}, ops include exp/log/square")
    for pname in args.params:
        cache_path = cache_dir / f"{pname}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                refits[pname] = pickle.load(f)
            print(f"  [cache] {pname}: complexity={refits[pname].pareto_complexity}, "
                  f"loss={refits[pname].pareto_loss:.3g}")
            continue
        print(f"  fitting {pname}...", flush=True)
        t0 = time.time()
        result = refit_1d_for_param(
            gp=gp, param_name=pname, z=z, k_grid=k_eboss,
            n_train=args.n_train, seed=args.seed,
            pysr_kwargs=pysr_kwargs,
        )
        elapsed = time.time() - t0
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        refits[pname] = result
        print(f"  [{elapsed:.0f}s] {pname}: complexity={result.pareto_complexity}, "
              f"loss={result.pareto_loss:.3g}", flush=True)

    # ---- 11D Fisher forecast ----
    print("\nRunning 11D Fisher on the GP and on the refit hybrid model...")
    lk_gp = GaussianLikelihood(model=gp, z=z, mock_data="gp", theta_fid=fid)
    fr_gp = fisher_matrix(
        likelihood=lk_gp, theta_fid=fid, params=PARAMS_11D,
        step_frac=0.02, rel_tol=0.05, max_halvings=2,
    )
    print("  GP done.")

    hybrid = _RefitMultiplicativeModel(
        gp=gp, fid=fid, refits=refits, k_grid=k_eboss, z=z,
    )
    # Sanity: at fid, hybrid == GP (multiplicative ratios all = 1).
    p_hy = hybrid.predict(fid, k_eboss, z)
    p_gp = gp.predict(fid, k_eboss, z)
    rel = np.max(np.abs(p_hy - p_gp) / p_gp)
    print(f"  Sanity: hybrid vs GP at fid, max rel diff = {rel:.3g} (should be ~0)")

    lk_hy = GaussianLikelihood(model=hybrid, z=z, mock_data="gp", theta_fid=fid)
    fr_hy = fisher_matrix(
        likelihood=lk_hy, theta_fid=fid, params=PARAMS_11D,
        step_frac=0.02, rel_tol=0.05, max_halvings=2,
    )
    print("  Hybrid (refit) done.")

    # ---- Scorecard ----
    target = ("Ap", "ns", "tau0", "dtau0")
    lines = [
        f"# 11D forecast: refit 1D PySR equations × multiplicative combine\n",
        f"z = {z},  niter = {args.niter},  maxsize = {args.maxsize},  "
        f"n_train = {args.n_train}\n",
        "Train target: flux_norm = (P_F - mean_k)/std_k.  Predict on raw P_F.\n",
        "",
        "| param | GP σ | refit-hybrid σ | hybrid/GP ratio | refit complexity | refit loss (flux_norm) |",
        "|---|---|---|---|---|---|",
    ]
    for i, pname in enumerate(PARAM_NAMES):
        r = refits.get(pname)
        sigma_gp = fr_gp.sigma[i]
        sigma_hy = fr_hy.sigma[i]
        ratio = sigma_hy / sigma_gp if sigma_gp > 0 else float("inf")
        if r is not None:
            cplx = r.pareto_complexity
            ploss = f"{r.pareto_loss:.3g}"
        else:
            cplx = "—"
            ploss = "GP fallback"
        lines.append(
            f"| {pname} | {sigma_gp:.3g} | {sigma_hy:.3g} | "
            f"**{ratio:.2f}×** | {cplx} | {ploss} |"
        )
    lines.append("")
    lines.append(f"## Target subset {target}")
    for pname in target:
        i = PARAM_NAMES.index(pname)
        ratio = fr_hy.sigma[i] / fr_gp.sigma[i] if fr_gp.sigma[i] > 0 else float("inf")
        lines.append(f"  - **{pname}**: ratio = {ratio:.2f}×")

    md = "\n".join(lines) + "\n"
    (args.output / "scorecard.md").write_text(md)
    print(md)
    np.savez(args.output / "fisher.npz",
             param_names=np.array(PARAM_NAMES),
             sigma_gp=fr_gp.sigma, sigma_hybrid=fr_hy.sigma,
             cov_gp=fr_gp.cov, cov_hybrid=fr_hy.cov)
    print(f"Refits cached at {cache_dir}/")


if __name__ == "__main__":
    main()
