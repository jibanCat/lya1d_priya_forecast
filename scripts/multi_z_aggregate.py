"""Phase 3 of option C: aggregate per-param multi-z refits → multi-z Fisher.

Reads `refits/<param>.pkl` produced by the parallel `refit_one_param.py`
jobs, builds `MultiZAdditiveTaylorModel`, runs per-z Fisher on the HF GP
(truth) and the hybrid model, sums F_phys across z (with z-bin
independence assumption), adds production priors + (optional) Kim
mean-flux prior on tau0, and writes the scorecard + corner plot.

Run:
  PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia \\
  PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \\
      python scripts/multi_z_aggregate.py \\
          --refits-dir results/refit_kodiaq_multiz_2.6-4.2/refits \\
          --output results/refit_kodiaq_multiz_2.6-4.2 \\
          --priors production --kim-tau0
"""

from __future__ import annotations

import argparse
import json
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

from priya_forecast.fisher import (
    combine_fisher_phys_arrays,
    compute_fisher_F_phys,
)
from priya_forecast.likelihood import GaussianLikelihood
from priya_forecast.parameters import (
    PARAM_NAMES,
    PARAMS_11D,
    fiducial_vector,
)
from priya_forecast.refit_taylor import MultiZAdditiveTaylorModel
from priya_forecast.deliverables import (
    per_param_summary_lines,
    write_resolution_correction_equations,
    write_resolution_correction_outputs,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refits-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--basedir", type=Path,
                   default=Path("/nfs/turbo/umor-yueyingn/mfho/birdgroup/"
                                "lya_xq100/kodiaq_2_2_4_6-48-48"))
    p.add_argument("--priors", choices=("production", "none"), default="production")
    p.add_argument("--kim-tau0", action="store_true",
                   help="Add Kim Gaussian prior on tau0: σ ≈ 0.331 (= 0.304 · fid).")
    p.add_argument("--cov-diag-frac", type=float, default=0.05,
                   help="Synthetic diagonal cov σ_k = frac·P_F(fid, k); placeholder "
                        "until the KSData covariance is wired in.")
    p.add_argument("--fix-params", nargs="+", default=["dtau0"],
                   help="Params held fixed at theta_fid in the multi-z Fisher. "
                        "Default ['dtau0'] — production paper's `USE_TAU0_ONLY=true` "
                        "convention; dtau0 is just tau0's z-dependent slope.")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    refits = {pn: None for pn in PARAM_NAMES}
    for pname in PARAM_NAMES:
        path = args.refits_dir / f"{pname}.pkl"
        if path.exists():
            with open(path, "rb") as fh:
                refits[pname] = pickle.load(fh)
    n_loaded = sum(r is not None for r in refits.values())
    print(f"Loaded {n_loaded}/{len(PARAM_NAMES)} refits from {args.refits_dir}")
    if n_loaded == 0:
        raise SystemExit("No refits to aggregate.")

    # Pull k_grid + z range from any one refit (they all share the same).
    sample = next(r for r in refits.values() if r is not None)
    k_grid = np.asarray(sample.k_grid, dtype=float)
    z_min = float(sample.z_min); z_max = float(sample.z_max)
    z_grid_kodiaq = np.array([2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])
    z_grid_use = z_grid_kodiaq[
        (z_grid_kodiaq >= z_min - 1e-6) & (z_grid_kodiaq <= z_max + 1e-6)
    ]
    print(f"  k=[{k_grid[0]:.4g}, {k_grid[-1]:.4g}] s/km, "
          f"z_grid={z_grid_use.tolist()}")

    fid = np.array(fiducial_vector(), dtype=float)
    print("Loading HF emulator...")
    from priya_forecast.models.gp_model import GPModel
    gp_hf = GPModel(basedir=args.basedir, fidelity="hf", kf=k_grid)

    # Build hybrid (only over the params with refits).
    refits_loaded = {p: r for p, r in refits.items() if r is not None}
    hybrid = MultiZAdditiveTaylorModel(
        gp=gp_hf, fid=fid, refits=refits_loaded,
        k_grid=k_grid, z_grid=z_grid_use,
    )
    max_rel = 0.0
    for z_check in z_grid_use:
        p_hy = hybrid.predict(fid, k_grid, float(z_check))
        p_gp = gp_hf.predict(fid, k_grid, float(z_check))
        max_rel = max(max_rel, float(np.max(np.abs(p_hy - p_gp) / p_gp)))
    print(f"  hybrid vs HF GP at fid (max over z): {max_rel*100:.4f}%")

    # Fisher params: drop fixed params (default ['dtau0']) — production paper's
    # `USE_TAU0_ONLY=true` convention treats dtau0 as a slope absorbed into
    # tau0's z-dependence.
    fix_set = set(args.fix_params or [])
    fisher_params = tuple(p for p in PARAMS_11D if p.name not in fix_set)
    fisher_param_names = [p.name for p in fisher_params]
    param_indices = [PARAM_NAMES.index(p.name) for p in fisher_params]
    theta_fid_subset = np.array(
        [fid[PARAM_NAMES.index(p.name)] for p in fisher_params]
    )
    if fix_set:
        print(f"  fixed at theta_fid: "
              f"{sorted({p: fid[PARAM_NAMES.index(p)] for p in fix_set}.items())}")

    print(f"\nRunning per-z Fisher on HF GP (truth) and hybrid for {len(z_grid_use)} z bins...")
    F_gp_list, F_hy_list = [], []
    for z_bin in z_grid_use:
        lk_gp_z = GaussianLikelihood(
            model=gp_hf, z=float(z_bin), mock_data="gp", theta_fid=fid,
            k_grid=k_grid, cov_diag_frac=args.cov_diag_frac,
        )
        F_gp_list.append(compute_fisher_F_phys(
            likelihood=lk_gp_z, theta_fid=fid, params=fisher_params,
            param_indices=param_indices,
            step_frac=0.02, rel_tol=0.05, max_halvings=2,
        ))
        lk_hy_z = GaussianLikelihood(
            model=hybrid, z=float(z_bin), mock_data="gp", theta_fid=fid,
            k_grid=k_grid, cov_diag_frac=args.cov_diag_frac,
        )
        F_hy_list.append(compute_fisher_F_phys(
            likelihood=lk_hy_z, theta_fid=fid, params=fisher_params,
            param_indices=param_indices,
            step_frac=0.02, rel_tol=0.05, max_halvings=2,
        ))
        print(f"  z={z_bin:.1f} done", flush=True)

    priors_sigma = {}
    if args.priors == "production":
        priors_sigma.update({
            "hub": 0.015, "omegamh2": 0.001, "bhfeedback": 0.005,
        })
    if args.kim_tau0:
        priors_sigma["tau0"] = 0.304 * 1.090
    if priors_sigma:
        print(f"\nPriors: {priors_sigma}")

    fr_gp = combine_fisher_phys_arrays(
        F_gp_list, params=fisher_params, theta_fid=theta_fid_subset,
        priors_sigma=priors_sigma if priors_sigma else None,
    )
    fr_hy = combine_fisher_phys_arrays(
        F_hy_list, params=fisher_params, theta_fid=theta_fid_subset,
        priors_sigma=priors_sigma if priors_sigma else None,
    )

    # Scorecard.
    target = ("Ap", "ns", "tau0", "dtau0")
    lines = [
        "# Multi-z Fisher forecast (PySR additive-Taylor combine, mode=local_anchored)",
        f"emulator: {args.basedir}",
        f"z range: [{z_min}, {z_max}] (z_grid={z_grid_use.tolist()})",
        f"k-grid: linspace({k_grid[0]:.4g}, {k_grid[-1]:.4g}, {len(k_grid)}) s/km",
        f"cov: synthetic diagonal, σ_k = {args.cov_diag_frac*100:.1f}%·P_F(fid, k) per z.",
        f"priors: {priors_sigma if priors_sigma else 'none'}",
        f"hybrid vs HF GP at fid (max over z): {max_rel*100:.4f}%",
        "",
        "| param | GP σ | hybrid σ | hybrid/GP ratio | LF rel-err | HF rel-err | x0? | complexity |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, pp in enumerate(fisher_params):
        pname = pp.name
        r = refits.get(pname)
        sigma_gp = fr_gp.sigma[i]
        sigma_hy = fr_hy.sigma[i]
        ratio = sigma_hy / sigma_gp if sigma_gp > 0 else float("inf")
        if r is not None:
            lf = f"{r.lf_train_mean_rel_err*100:.2f}%"
            hf = f"{r.hf_train_mean_rel_err*100:.2f}%"
            cplx = str(r.pareto_complexity)
            has_x0 = "✓" if "x0" in r.equation_str else "✗"
        else:
            lf = hf = "—"; cplx = "—"; has_x0 = "—"
        lines.append(
            f"| {pname} | {sigma_gp:.3g} | {sigma_hy:.3g} | "
            f"**{ratio:.2f}×** | {lf} | {hf} | {has_x0} | {cplx} |"
        )
    lines.append("")
    lines.append(f"## Target subset {target}")
    for pname in target:
        if pname in fix_set:
            lines.append(f"  - **{pname}**: fixed at fid={fid[PARAM_NAMES.index(pname)]:.4g}")
            continue
        i = fisher_param_names.index(pname)
        ratio = fr_hy.sigma[i] / fr_gp.sigma[i] if fr_gp.sigma[i] > 0 else float("inf")
        lines.append(f"  - **{pname}**: ratio = {ratio:.2f}×")

    md = "\n".join(lines) + "\n"
    (args.output / "scorecard.md").write_text(md)
    print("\n" + md)
    np.savez(
        args.output / "fisher.npz",
        param_names=np.array(fisher_param_names),
        sigma_gp=fr_gp.sigma, sigma_hybrid=fr_hy.sigma,
        cov_gp=fr_gp.cov, cov_hybrid=fr_hy.cov,
        theta_fid=theta_fid_subset,
    )

    # Corner plot.
    try:
        from priya_forecast.plotting import plot_fisher_corner
        plot_fisher_corner(
            fr_gp=fr_gp, fr_hybrid=fr_hy, params=fisher_params,
            output_path=args.output / "corner.pdf",
            title=f"Multi-z (z={z_min}-{z_max}) — GP (black) vs hybrid (red)",
        )
        plot_fisher_corner(
            fr_gp=fr_gp, fr_hybrid=fr_hy, params=fisher_params,
            output_path=args.output / "corner.png",
            title=f"Multi-z (z={z_min}-{z_max}) — GP (black) vs hybrid (red)",
        )
        print(f"Corner plots: {args.output / 'corner.{pdf,png}'}")
    except Exception as e:
        print(f"  (corner plot skipped: {e})")

    # Per-param summary + resolution-correction deliverables (paper artifacts).
    refits_loaded_dict = {p: r for p, r in refits.items() if r is not None}
    summary_lines = per_param_summary_lines(refits_loaded_dict)
    (args.output / "per_param_summary.md").write_text("\n".join(summary_lines) + "\n")
    write_resolution_correction_outputs(
        refits_loaded_dict, k_grid, fid, args.output,
        z_eval=float((z_min + z_max) / 2.0),
    )
    write_resolution_correction_equations(refits_loaded_dict, args.output)
    print(f"Deliverables: per_param_summary.md, resolution_correction.md/json/grid, "
          f"resolution_correction_equations.md")


if __name__ == "__main__":
    main()
