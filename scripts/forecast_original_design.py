"""The "Original Design" forecast: published 1D PySR equations × 1st-order
multiplicative combine, scored on the FULL 11D Fisher matrix.

The 4 quoted equations (dtau0, Ap, ns, alphaq) get their published
PySR forms with mode='auto' normalization. The other 7 params get
perfect_1D_slices fallback — i.e. each remaining f_i delegates to the
GP at θ_i with everything else at fid. That isolates the PySR-induced
σ inflation to the 4 published parameters; for the 7 GP-fallback
params σ matches σ_GP by construction.

Expected output: a 11-row scorecard showing σ_pysr / σ_GP for every
parameter. The "Original Design" claim is supported if {Ap, ns, dtau0}
σ ratios are within ~2× of GP. (tau0 has no published equation so
it'll be 1.0 by construction.)

Run:
  PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \\
      python scripts/forecast_original_design.py --output results/original_design_11D
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_LYAEMU = Path("/home/mfho/student_projects/lya_emulator_full")
if _LYAEMU.exists():
    sys.path.insert(0, str(_LYAEMU))

from priya_forecast.config import EqnConfig, EqnParam
from priya_forecast.data import load_eboss
from priya_forecast.fisher import fisher_matrix
from priya_forecast.likelihood import GaussianLikelihood
from priya_forecast.models import PySRModel
from priya_forecast.models.base import P1DModel
from priya_forecast.models.normalization import derive_from_gp
from priya_forecast.models.pysr_model import compile_equation
from priya_forecast.parameters import (
    PARAM_NAMES,
    PARAMS_11D,
    fiducial_vector,
    get_param,
)


PUBLISHED_EQUATIONS = {
    "dtau0":  "(((1.4061172 - k)**(-0.5989224)) * dtau0) - (r * 1.3422583) + dtau0 - 1.3998809",
    "Ap":     "(((2*Ap)**(cos(k))) + ((-0.5290618 - sin(r)) * 1.4107764)) + Ap",
    "ns":     "((ns * k) - r) * 2.3955164",
    "alphaq": "cos(r + 0.7157408 - 1.5351741*k)**4 / 0.47581 - r - 1.04696",
}


class _HybridModel(P1DModel):
    """Multiplicative combine where SOME params delegate to GP per-param-slice
    and OTHERS use a PySR equation. Mathematically:

      P_pysr(theta) = P_fid(k) ·
        ∏_{i ∈ pysr_params} [f_i(theta_i, k) / f_i(theta_fid_i, k)]
        ∏_{i ∈ gp_fallback_params} [GP(theta_i, others=fid) / GP(fid)]

    The first term is the published PySR equations; the second is the
    perfect_1D_slices (GP-delegate) treatment for the other 7 params.
    """

    def __init__(self, *, gp, fid, pysr_compiled, k_grid, z):
        self._gp = gp
        self._fid = fid
        self._compiled = pysr_compiled  # dict: param_name -> CompiledEquation
        self._k_grid = k_grid
        self._z = z
        # Pre-cache GP at fid for the multiplicative reference.
        self._p_fid = gp.predict(fid, k_grid, z)

    def predict(self, theta: np.ndarray, k: np.ndarray, z: float) -> np.ndarray:
        if not np.allclose(k, self._k_grid):
            raise ValueError("This forecast model is built for a fixed k-grid.")
        out = self._p_fid.copy()
        for pname in PARAM_NAMES:
            i = PARAM_NAMES.index(pname)
            if pname in self._compiled:
                ce = self._compiled[pname]
                num = ce.evaluate(theta_i=float(theta[i]), k=k)
                den = ce.evaluate(theta_i=float(ce.fiducial), k=k)
                with np.errstate(divide="ignore", invalid="ignore"):
                    out = out * (num / den)
            else:
                t_only = self._fid.copy()
                t_only[i] = theta[i]
                num = self._gp.predict(t_only, k, z)
                with np.errstate(divide="ignore", invalid="ignore"):
                    out = out * (num / self._p_fid)
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--z", type=float, default=3.6)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-norm-samples", type=int, default=64,
                   help="Number of GP samples for deriving (mean_k, std_k) per param.")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading real PRIYA GP emulator...")
    from priya_forecast.models.gp_model import GPModel
    gp = GPModel()
    z = args.z
    k_eboss, _, cov = load_eboss(z=z)
    fid = np.array(fiducial_vector(), dtype=float)

    # Compile each published equation with GP-derived per-k normalization.
    print("Compiling published equations with auto-normalization...")
    pysr_compiled = {}
    for pname, expr in PUBLISHED_EQUATIONS.items():
        norm = derive_from_gp(
            gp_model=gp, param_name=pname, z=z, k_grid=k_eboss,
            n_samples=args.n_norm_samples, seed=0,
        )
        pysr_compiled[pname] = compile_equation(
            param_name=pname, raw_expression=expr,
            variables=[pname, "k", "r"], fix={"r": 0.8}, norm=norm,
            fiducial=get_param(pname).fid,
        )

    hybrid = _HybridModel(
        gp=gp, fid=fid, pysr_compiled=pysr_compiled, k_grid=k_eboss, z=z,
    )

    # Sanity: hybrid at fid recovers GP at fid (multiplicative ratios all = 1).
    p_hybrid_fid = hybrid.predict(fid, k_eboss, z)
    p_gp_fid = gp.predict(fid, k_eboss, z)
    rel = np.max(np.abs(p_hybrid_fid - p_gp_fid) / p_gp_fid)
    print(f"  Sanity: hybrid vs GP at fid, max relative diff = {rel:.3g}")
    assert rel < 1e-6, "Hybrid model must equal GP at fid by construction."

    # Run 11D Fisher on each model.
    print(f"\nRunning Fisher on the full 11D parameter space at z={z}...")
    lk_gp = GaussianLikelihood(model=gp, z=z, mock_data="gp", theta_fid=fid)
    fr_gp = fisher_matrix(
        likelihood=lk_gp, theta_fid=fid, params=PARAMS_11D,
        step_frac=0.02, rel_tol=0.05, max_halvings=2,
    )
    print("  GP Fisher done.")

    lk_hy = GaussianLikelihood(model=hybrid, z=z, mock_data="gp", theta_fid=fid)
    fr_hy = fisher_matrix(
        likelihood=lk_hy, theta_fid=fid, params=PARAMS_11D,
        step_frac=0.02, rel_tol=0.05, max_halvings=2,
    )
    print("  Hybrid (Original Design) Fisher done.")

    # Build the scorecard.
    lines = ["# Original Design 11D forecast scorecard\n",
             f"z = {z}, mock_data = 'gp' (GP at fid as data), Fisher.\n",
             f"4 published 1D PySR equations (dtau0/Ap/ns/alphaq) × multiplicative combine + ",
             f"GP-slice fallback for the other 7 params.\n",
             "",
             "| param | GP σ | hybrid σ | hybrid/GP ratio | has PySR eq? |",
             "|---|---|---|---|---|"]
    has_pysr = set(PUBLISHED_EQUATIONS)
    for i, pname in enumerate(PARAM_NAMES):
        sigma_gp = fr_gp.sigma[i]
        sigma_hy = fr_hy.sigma[i]
        ratio = sigma_hy / sigma_gp if sigma_gp > 0 else float("inf")
        flag = "✓ published" if pname in has_pysr else "GP-slice fallback"
        lines.append(
            f"| {pname} | {sigma_gp:.3g} | {sigma_hy:.3g} | "
            f"**{ratio:.2f}×** | {flag} |"
        )
    target_subset = ("Ap", "ns", "tau0", "dtau0")
    lines.append("")
    lines.append(f"## Target subset {target_subset} (from user's Comment 1):")
    for pname in target_subset:
        i = PARAM_NAMES.index(pname)
        sigma_gp = fr_gp.sigma[i]
        sigma_hy = fr_hy.sigma[i]
        ratio = sigma_hy / sigma_gp
        flag = "PySR" if pname in has_pysr else "GP-slice"
        lines.append(f"  - **{pname}**: ratio = {ratio:.2f}×  ({flag})")

    md = "\n".join(lines)
    (args.output / "scorecard.md").write_text(md + "\n")
    print(md)

    # Also save raw arrays.
    np.savez(args.output / "fisher.npz",
             param_names=np.array(PARAM_NAMES),
             sigma_gp=fr_gp.sigma, sigma_hybrid=fr_hy.sigma,
             cov_gp=fr_gp.cov, cov_hybrid=fr_hy.cov,
             F_gp=fr_gp.F, F_hybrid=fr_hy.F)
    print(f"\nWrote {args.output}/scorecard.md and fisher.npz")


if __name__ == "__main__":
    main()
