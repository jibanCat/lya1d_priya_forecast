"""Multi-D PySR cross-coupled forecast: single eq over a parameter subset.

End-to-end pipeline for the user's "drop priored-out + drop tau0" plan:

  1. Train ONE multi-D PySR equation over the cross-coupled subset
     `{ns, Ap, herei, heref, alphaq, hireionz}` jointly. Inputs:
     `(θ_ns, θ_Ap, ..., θ_hireionz, k, resolution, z)` — 9 features.
     Captures cross-coupling that per-1D + additive Taylor cannot
     (the herei × alphaq positive coupling from the Phase 5 matrix).

  2. Build `MultiDCrossCoupledModel`: multi-D eq for the subset +
     GP-slice for {tau0, hub, omegamh2, bhfeedback}; dtau0 fixed at 0.

  3. Run multi-z Fisher with production priors on hub/Ω/bh, optional
     Kim Gaussian prior on tau0.

  4. Write scorecard + corner + paper deliverables.

Run:
  PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia \\
  PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \\
      python scripts/refit_multid_subset.py \\
          --z-min 2.6 --z-max 4.2 --n-total 256 \\
          --output results/refit_multid_z2.6-4.2
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
from priya_forecast.refit_multi_d import (
    DEFAULT_SUBSET,
    MultiDCrossCoupledModel,
    refit_multi_d,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", nargs="+", default=list(DEFAULT_SUBSET),
                   help="Cross-coupled param names. Default: ns Ap herei heref alphaq hireionz.")
    p.add_argument("--fix-params", nargs="+", default=["dtau0"],
                   help="Held fixed at fid (default: dtau0; production "
                        "USE_TAU0_ONLY=true convention).")
    p.add_argument("--z-min", type=float, default=2.6)
    p.add_argument("--z-max", type=float, default=4.2)
    p.add_argument("--k-min", type=float, default=0.005)
    p.add_argument("--k-max", type=float, default=0.064)
    p.add_argument("--n-k", type=int, default=32)
    p.add_argument("--n-total", type=int, default=256,
                   help="Sobol points over (θ_subset × z) for multi-D training.")
    p.add_argument("--niter", type=int, default=100,
                   help="PySR iterations (multi-D fit needs more than per-1D).")
    p.add_argument("--maxsize", type=int, default=25,
                   help="PySR max equation size (multi-D may need more).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--basedir", type=Path,
                   default=Path("/nfs/turbo/umor-yueyingn/mfho/birdgroup/"
                                "lya_xq100/kodiaq_2_2_4_6-48-48"))
    p.add_argument("--cov-diag-frac", type=float, default=0.05,
                   help="Synthetic diagonal cov σ_k = frac·P_F(fid, k). "
                        "Ignored if --use-ksdata is set.")
    p.add_argument("--use-ksdata", action="store_true",
                   help="Use the real KODIAQ-SQUAD covariance.")
    p.add_argument("--priors", choices=("production", "none"), default="production")
    p.add_argument("--kim-tau0", action="store_true",
                   help="Apply Kim Gaussian prior on tau0 (σ ≈ 0.331).")
    p.add_argument("--no-dim-balanced", action="store_true",
                   help="Disable the dim-balanced loss (use standard MSE).")
    p.add_argument("--procs", type=int, default=4,
                   help="PySR multithreading procs (set to ~ntasks-1 in SLURM).")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    fid = np.array(fiducial_vector(), dtype=float)
    k_grid = np.linspace(args.k_min, args.k_max, args.n_k)
    z_grid_kodiaq = np.array([2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])
    z_grid_use = z_grid_kodiaq[
        (z_grid_kodiaq >= args.z_min - 1e-6) & (z_grid_kodiaq <= args.z_max + 1e-6)
    ]

    print(f"Multi-D cross-coupled refit:")
    print(f"  subset: {args.subset} (n={len(args.subset)})")
    print(f"  fixed:  {args.fix_params}")
    print(f"  z_grid: {z_grid_use.tolist()}")
    print(f"  k-grid: linspace({args.k_min}, {args.k_max}, {args.n_k}) s/km")
    print(f"  Sobol n_total={args.n_total}, niter={args.niter}, maxsize={args.maxsize}")
    print()

    # Step 1: refit multi-D eq.
    refit_cache = args.output / "multid_refit.pkl"
    if refit_cache.exists():
        with open(refit_cache, "rb") as fh:
            multid_refit = pickle.load(fh)
        print(f"[cache] multi-D refit loaded from {refit_cache}")
        print(f"  eq: {multid_refit.equation_str[:140]}")
    else:
        print(f"Loading kodiaq emulators (LF + HF)...")
        from priya_forecast.models.gp_model import GPModel
        gp_lf = GPModel(basedir=args.basedir, fidelity="lf", kf=k_grid)
        gp_hf = GPModel(basedir=args.basedir, fidelity="hf", kf=k_grid)
        # Warm both with one prediction (pulls in the lazy emulator load).
        _ = gp_lf.predict(fid, k_grid, 3.6)
        _ = gp_hf.predict(fid, k_grid, 3.6)
        print(f"\nFitting multi-D PySR...")
        t0 = time.time()
        multid_refit = refit_multi_d(
            gp_lf=gp_lf, gp_hf=gp_hf,
            subset_names=tuple(args.subset),
            z_min=args.z_min, z_max=args.z_max, k_grid=k_grid,
            n_total=args.n_total,
            pysr_kwargs=dict(
                niterations=args.niter, maxsize=args.maxsize,
                procs=args.procs,
            ),
            seed=args.seed,
            use_dim_balanced_loss=(not args.no_dim_balanced),
        )
        print(f"  [{time.time()-t0:.0f}s] complexity={multid_refit.pareto_complexity}, "
              f"flux_norm loss={multid_refit.pareto_loss:.3g}")
        print(f"  LF rel-err={multid_refit.lf_train_mean_rel_err*100:.2f}% "
              f"HF rel-err={multid_refit.hf_train_mean_rel_err*100:.2f}%")
        print(f"  eq: {multid_refit.equation_str[:140]}")
        with open(refit_cache, "wb") as fh:
            pickle.dump(multid_refit, fh)

    # Step 2: build the hybrid model.
    from priya_forecast.models.gp_model import GPModel
    gp_hf = GPModel(basedir=args.basedir, fidelity="hf", kf=k_grid)
    hybrid = MultiDCrossCoupledModel(
        multi_d_refit=multid_refit, gp=gp_hf, fid=fid, k_grid=k_grid,
        z_grid=z_grid_use, fixed_params=tuple(args.fix_params),
    )
    # Sanity: at fid, hybrid == HF GP exactly at every z.
    max_rel = 0.0
    for z_check in z_grid_use:
        p_hy = hybrid.predict(fid, k_grid, float(z_check))
        p_gp = gp_hf.predict(fid, k_grid, float(z_check))
        max_rel = max(max_rel, float(np.max(np.abs(p_hy - p_gp) / p_gp)))
    print(f"\nhybrid vs HF GP at fid (max over z): {max_rel*100:.4f}%")

    # Step 3: per-z Fisher + sum.
    fix_set = set(args.fix_params or [])
    fisher_params = tuple(p for p in PARAMS_11D if p.name not in fix_set)
    fisher_param_names = [p.name for p in fisher_params]
    param_indices = [PARAM_NAMES.index(p.name) for p in fisher_params]
    theta_fid_subset = np.array(
        [fid[PARAM_NAMES.index(p.name)] for p in fisher_params]
    )
    print(f"\nFisher params (n={len(fisher_params)}): {fisher_param_names}")
    if fix_set:
        print(f"  fixed: {sorted(fix_set)}")

    priors_sigma = {}
    if args.priors == "production":
        priors_sigma.update({"hub": 0.015, "omegamh2": 0.001, "bhfeedback": 0.005})
    if args.kim_tau0:
        priors_sigma["tau0"] = 0.304 * 1.090
    priors_sigma_arg = priors_sigma if priors_sigma else None

    if args.use_ksdata:
        from priya_forecast.fisher import fisher_matrix
        from priya_forecast.ksdata_likelihood import KSDataLikelihood
        print("\nMulti-z Fisher with KSData covariance...")
        lk_gp_ks = KSDataLikelihood(
            model=gp_hf, z_min=args.z_min, z_max=args.z_max, k_max=args.k_max,
            mock_data="gp", theta_fid=fid,
        )
        lk_hy_ks = KSDataLikelihood(
            model=hybrid, z_min=args.z_min, z_max=args.z_max, k_max=args.k_max,
            mock_data="gp", theta_fid=fid,
        )
        fr_gp = fisher_matrix(
            likelihood=lk_gp_ks, theta_fid=fid, params=fisher_params,
            param_indices=param_indices,
            step_frac=0.02, rel_tol=0.05, max_halvings=2,
            priors_sigma=priors_sigma_arg,
        )
        fr_hy = fisher_matrix(
            likelihood=lk_hy_ks, theta_fid=fid, params=fisher_params,
            param_indices=param_indices,
            step_frac=0.02, rel_tol=0.05, max_halvings=2,
            priors_sigma=priors_sigma_arg,
        )
        cov_label = (f"KSData(conservative=True), z=[{args.z_min},{args.z_max}], k≤{args.k_max}")
        # Skip per-z loop below.
        _SKIP_PER_Z = True
    else:
        _SKIP_PER_Z = False
        cov_label = f"synthetic {args.cov_diag_frac*100:.1f}%·P_F(fid, k)"

    if not _SKIP_PER_Z:
        print(f"\nPer-z Fisher on {len(z_grid_use)} z bins...")
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
        if priors_sigma:
            print(f"  priors: {priors_sigma}")
        fr_gp = combine_fisher_phys_arrays(
            F_gp_list, params=fisher_params, theta_fid=theta_fid_subset,
            priors_sigma=priors_sigma_arg,
        )
        fr_hy = combine_fisher_phys_arrays(
            F_hy_list, params=fisher_params, theta_fid=theta_fid_subset,
            priors_sigma=priors_sigma_arg,
        )

    # Scorecard.
    target = ("Ap", "ns", "tau0", "dtau0")
    in_subset = lambda pn: pn in set(args.subset)
    lines = [
        "# Multi-D cross-coupled forecast (single PySR eq + GP-slice fallback)",
        f"emulator: {args.basedir}",
        f"subset: {args.subset}  (handled by joint multi-D PySR eq)",
        f"fixed:  {args.fix_params}",
        f"z range: [{args.z_min}, {args.z_max}] (z_grid={z_grid_use.tolist()})",
        f"k-grid: linspace({args.k_min}, {args.k_max}, {args.n_k}) s/km, "
        f"cov: {args.cov_diag_frac*100:.1f}%·P_F(fid, k) per z.",
        f"priors: {priors_sigma if priors_sigma else 'none'}",
        f"hybrid vs HF GP at fid: {max_rel*100:.4f}%",
        f"multi-D eq complexity: {multid_refit.pareto_complexity}, "
        f"flux_norm loss: {multid_refit.pareto_loss:.3g}",
        "",
        "| param | in subset? | GP σ | hybrid σ | hybrid/GP ratio |",
        "|---|---|---|---|---|",
    ]
    for i, pp in enumerate(fisher_params):
        pname = pp.name
        sigma_gp = fr_gp.sigma[i]
        sigma_hy = fr_hy.sigma[i]
        ratio = sigma_hy / sigma_gp if sigma_gp > 0 else float("inf")
        in_sub = "✓ multi-D" if in_subset(pname) else "GP-slice"
        lines.append(
            f"| {pname} | {in_sub} | {sigma_gp:.3g} | {sigma_hy:.3g} | "
            f"**{ratio:.2f}×** |"
        )
    lines.append("")
    lines.append(f"## Target subset {target}")
    for pname in target:
        if pname in fix_set:
            lines.append(f"  - **{pname}**: fixed at fid={fid[PARAM_NAMES.index(pname)]:.4g}")
            continue
        i = fisher_param_names.index(pname)
        ratio = fr_hy.sigma[i] / fr_gp.sigma[i] if fr_gp.sigma[i] > 0 else float("inf")
        lines.append(f"  - **{pname}**: ratio = {ratio:.2f}× ({'multi-D' if in_subset(pname) else 'GP-slice'})")

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
    # Save the multi-D eq as a separate paper-friendly file.
    from priya_forecast.deliverables import _prettify_equation
    pretty_names = [f"θ_{name}" for name in args.subset]
    pretty_names += ["k", "r", "z"]
    eq_md = [
        "# Multi-D cross-coupled PySR equation",
        "",
        f"**Subset**: {args.subset}",
        f"**Inputs** (in order): "
        + ", ".join([f"x{i}={n}" for i, n in enumerate(pretty_names)]),
        "",
        "**Trained equation** (raw):",
        "```",
        multid_refit.equation_str,
        "```",
        "",
        "**Trained equation (variables prettified)**:",
        "```",
        _prettify_equation(multid_refit.equation_str, names=pretty_names),
        "```",
        "",
        f"complexity = {multid_refit.pareto_complexity}",
        f"flux_norm loss = {multid_refit.pareto_loss:.3g}",
        f"LF rel-err = {multid_refit.lf_train_mean_rel_err*100:.2f}%, "
        f"HF rel-err = {multid_refit.hf_train_mean_rel_err*100:.2f}%",
    ]
    (args.output / "multid_equation.md").write_text("\n".join(eq_md) + "\n")
    print(f"Multi-D equation: {args.output / 'multid_equation.md'}")
    print(f"Scorecard: {args.output / 'scorecard.md'}")
    print(f"Fisher: {args.output / 'fisher.npz'}")

    # Corner.
    try:
        from priya_forecast.plotting import plot_fisher_corner
        for ext in ("pdf", "png"):
            plot_fisher_corner(
                fr_gp=fr_gp, fr_hybrid=fr_hy, params=fisher_params,
                output_path=args.output / f"corner.{ext}",
                title=f"Multi-D cross-coupled (z={args.z_min}-{args.z_max})",
            )
        print(f"Corner: {args.output / 'corner.{pdf,png}'}")
    except Exception as e:
        print(f"  (corner plot skipped: {e})")


if __name__ == "__main__":
    main()
