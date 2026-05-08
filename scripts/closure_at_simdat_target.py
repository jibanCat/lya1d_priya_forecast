"""Phase 2 step 0: closure of Phase-1 hybrid at theta_target_simdat.

Diagnostic-only — no new PySR fits. Compares σ_GP_Fisher,
σ_PySR_Fisher (both with KSData covariance), and σ_MCMC_simdat at
θ_target = simdat-ind15 truth (with dtau0 → 0 per Kim convention).

If σ_PySR_at_target ≈ σ_GP_at_target ≈ σ_MCMC_simdat the per-1D +
additive-Taylor extrapolation is faithful off-fid → pair fits may not be
needed for those params. If σ_PySR_at_target diverges sharply, the
discovered eqs have wrong fid-curvature → need Phase 1.5 smart refit
or pair coupling.

Run (login node, needs lyaemu/GPy):
  PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia \\
  PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \\
      python scripts/closure_at_simdat_target.py \\
          --refits-dir results/refit_optionC_z2.6-4.2/refits \\
          --truth results/simdat_ind15_truth.npz \\
          --output results/closure_at_simdat_ind15_ksdata
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

from priya_forecast.fisher import fisher_matrix
from priya_forecast.parameters import (
    PARAM_NAMES,
    PARAMS_11D,
    fiducial_vector,
)
from priya_forecast.refit_taylor import MultiZAdditiveTaylorModel


REL_ERR_THRESHOLD = 0.05  # mirrors multi_z_aggregate.py


def _load_refits_with_gate(refits_dir: Path) -> dict:
    """Load and gate refits the same way the aggregator does."""
    refits_raw = {pn: None for pn in PARAM_NAMES}
    for pname in PARAM_NAMES:
        path = refits_dir / f"{pname}.pkl"
        if path.exists():
            with open(path, "rb") as fh:
                refits_raw[pname] = pickle.load(fh)
    refits = dict(refits_raw)
    dropped = []
    for pname, r in list(refits_raw.items()):
        if r is None:
            continue
        has_x0 = "x0" in r.equation_str
        lf_ok = (np.isfinite(r.lf_train_mean_rel_err)
                 and r.lf_train_mean_rel_err < REL_ERR_THRESHOLD)
        hf_ok = (np.isfinite(r.hf_train_mean_rel_err)
                 and r.hf_train_mean_rel_err < REL_ERR_THRESHOLD)
        if not (has_x0 and lf_ok and hf_ok):
            refits[pname] = None
            dropped.append(pname)
    if dropped:
        print(f"[gate] dropped {dropped}; routing via GP-slice.")
    return refits


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--refits-dir", type=Path, required=True)
    p.add_argument("--pair-refits-dir", type=Path, default=None,
                   help="Optional: dir of Phase-2 pair refits (.pkl). If set, "
                        "wrap the base hybrid with MultiZPairCoupledModel "
                        "before computing the off-fid Fisher.")
    p.add_argument("--truth", type=Path, required=True,
                   help="NPZ with theta_target, mcmc_sigma, mcmc_corr, param_names.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--basedir", type=Path,
                   default=Path("/nfs/turbo/umor-yueyingn/mfho/birdgroup/"
                                "lya_xq100/kodiaq_2_2_4_6-48-48"))
    p.add_argument("--k-max", type=float, default=0.064)
    p.add_argument("--fix-params", nargs="+", default=["dtau0"])
    p.add_argument("--priors", choices=("production", "none"), default="production")
    p.add_argument("--kim-tau0", action="store_true",
                   help="Kim Gaussian prior on tau0: σ ≈ 0.331 (= 0.304·fid).")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # 1) Load truth (θ_target + σ_MCMC).
    truth = np.load(args.truth)
    theta_target_full = np.asarray(truth["theta_target"], dtype=float).copy()
    mcmc_sigma_full = np.asarray(truth["mcmc_sigma"], dtype=float).copy()
    truth_names = list(truth["param_names"])
    assert truth_names == list(PARAM_NAMES), (
        f"truth param_names mismatch: {truth_names} vs {PARAM_NAMES}"
    )
    # Unit conversion: the MCMC chain stores Ap in raw units (~1.9e-9),
    # but our pipeline (and the GP emulator's prior cube) uses Ap × 10^9
    # (~1.9; see GPModel._theta_15d). Apply the same scaling to the
    # marginal sigma. Other params share units (no conversion).
    ap_idx = PARAM_NAMES.index("Ap")
    if theta_target_full[ap_idx] < 1e-3:
        theta_target_full[ap_idx] *= 1e9
        mcmc_sigma_full[ap_idx] *= 1e9
        print(f"[units] Converted Ap from raw to internal "
              f"(×1e9): θ_Ap={theta_target_full[ap_idx]:.4f}, "
              f"σ_MCMC[Ap]={mcmc_sigma_full[ap_idx]:.4f}")
    # Force dtau0 -> 0 (Kim USE_TAU0_ONLY convention; dtau0 fixed in our forecast).
    dtau0_idx = PARAM_NAMES.index("dtau0")
    theta_target = theta_target_full.copy()
    theta_target[dtau0_idx] = 0.0
    print("θ_target (with dtau0 → 0 for Kim convention):")
    for n, v in zip(PARAM_NAMES, theta_target):
        print(f"  {n:>11s} = {v:+.4e}")

    # 2) Load refits + gate.
    refits = _load_refits_with_gate(args.refits_dir)
    n_loaded = sum(r is not None for r in refits.values())
    print(f"Loaded {n_loaded}/{len(PARAM_NAMES)} refits (post-gate).")
    sample = next(r for r in refits.values() if r is not None)
    k_grid = np.asarray(sample.k_grid, dtype=float)
    z_min, z_max = float(sample.z_min), float(sample.z_max)
    z_grid_kodiaq = np.array([2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])
    z_grid_use = z_grid_kodiaq[
        (z_grid_kodiaq >= z_min - 1e-6) & (z_grid_kodiaq <= z_max + 1e-6)
    ]
    print(f"k=[{k_grid[0]:.4g}, {k_grid[-1]:.4g}] s/km; z={z_grid_use.tolist()}")

    # 3) Load HF GP, build hybrid (anchored at the default fid — that's
    # how the per-1D refits were trained; we only change WHERE Fisher
    # evaluates).
    print("Loading HF emulator...")
    from priya_forecast.models.gp_model import GPModel
    gp_hf = GPModel(basedir=args.basedir, fidelity="hf", kf=k_grid)
    fid_default = np.array(fiducial_vector(), dtype=float)

    # 4) KSData covariance + likelihoods evaluating at θ_target.
    from priya_forecast.ksdata_likelihood import KSDataLikelihood
    from lyaemu.lyman_data import KSData
    _ks = KSData(conservative=True)
    ks_k_grid = np.sort(np.unique(_ks.kf[_ks.kf <= args.k_max + 1e-6]))
    print(f"KSData k-grid: {len(ks_k_grid)} bins ≤ {args.k_max} s/km.")
    base_hybrid_ks = MultiZAdditiveTaylorModel(
        gp=gp_hf, fid=fid_default, refits=refits,
        k_grid=ks_k_grid, z_grid=z_grid_use,
    )
    if args.pair_refits_dir is not None:
        from priya_forecast.refit_pair import (
            MultiZPairCoupledModel, Refit2DPairResult,  # noqa: F401
        )
        pair_refits_loaded = []
        for path in sorted(args.pair_refits_dir.glob("*.pkl")):
            with open(path, "rb") as fh:
                pair_refits_loaded.append(pickle.load(fh))
        if pair_refits_loaded:
            hybrid_ks = MultiZPairCoupledModel(
                base=base_hybrid_ks, pairs=pair_refits_loaded,
            )
            print(f"Phase-2 hybrid wraps base with "
                  f"{len(pair_refits_loaded)} pair(s).")
        else:
            hybrid_ks = base_hybrid_ks
            print("(no pair refits found — using base Phase-1 hybrid.)")
    else:
        hybrid_ks = base_hybrid_ks
    # Sanity: at θ=fid the hybrid still matches the GP.
    fid_check = np.max(np.abs(
        hybrid_ks.predict(fid_default, ks_k_grid, float(z_grid_use[len(z_grid_use)//2]))
        - gp_hf.predict(fid_default, ks_k_grid, float(z_grid_use[len(z_grid_use)//2]))
    ))
    print(f"hybrid vs GP at fid (sanity, mid-z): max |Δ| = {fid_check:.2e}")
    # Off-fid sanity: how big is the hybrid-vs-GP gap at θ_target itself?
    z_mid = float(z_grid_use[len(z_grid_use)//2])
    p_gp_at_target = gp_hf.predict(theta_target, ks_k_grid, z_mid)
    p_hy_at_target = hybrid_ks.predict(theta_target, ks_k_grid, z_mid)
    rel_at_target = np.max(np.abs((p_hy_at_target - p_gp_at_target) / p_gp_at_target))
    print(f"hybrid vs GP at θ_target (mid-z): max |Δ/P_F| = {rel_at_target*100:.2f}%")

    # 5) Fisher at θ_target with KSData covariance.
    fix_set = set(args.fix_params or [])
    fisher_params = tuple(p for p in PARAMS_11D if p.name not in fix_set)
    fisher_param_names = [p.name for p in fisher_params]
    param_indices = [PARAM_NAMES.index(p.name) for p in fisher_params]
    priors_sigma = {}
    if args.priors == "production":
        priors_sigma.update({"hub": 0.015, "omegamh2": 0.001, "bhfeedback": 0.005})
    if args.kim_tau0:
        priors_sigma["tau0"] = 0.304 * 1.090
    if priors_sigma:
        print(f"Priors: {priors_sigma}")

    print("Fisher: GP at θ_target...")
    lk_gp = KSDataLikelihood(
        model=gp_hf, z_min=z_min, z_max=z_max, k_max=args.k_max,
        mock_data="gp", theta_fid=theta_target,
    )
    fr_gp = fisher_matrix(
        likelihood=lk_gp, theta_fid=theta_target, params=fisher_params,
        param_indices=param_indices,
        # step_frac=0.005 instead of 0.02: at θ_target_simdat several
        # params (notably hireionz=7.946 vs upper 8.0) are close to their
        # prior boundaries, so the centered 5-point stencil with the
        # default step_frac=0.02 falls off the GP's training range.
        # max_halvings=4 lets adaptive halving recover if 0.005 is too
        # tight for any single param.
        step_frac=0.005, rel_tol=0.05, max_halvings=4,
        priors_sigma=priors_sigma if priors_sigma else None,
    )
    print("Fisher: hybrid at θ_target...")
    lk_hy = KSDataLikelihood(
        model=hybrid_ks, z_min=z_min, z_max=z_max, k_max=args.k_max,
        mock_data="gp", theta_fid=theta_target,
    )
    fr_hy = fisher_matrix(
        likelihood=lk_hy, theta_fid=theta_target, params=fisher_params,
        param_indices=param_indices,
        # step_frac=0.005 instead of 0.02: at θ_target_simdat several
        # params (notably hireionz=7.946 vs upper 8.0) are close to their
        # prior boundaries, so the centered 5-point stencil with the
        # default step_frac=0.02 falls off the GP's training range.
        # max_halvings=4 lets adaptive halving recover if 0.005 is too
        # tight for any single param.
        step_frac=0.005, rel_tol=0.05, max_halvings=4,
        priors_sigma=priors_sigma if priors_sigma else None,
    )

    # 6) Build the closure scorecard.
    sigma_gp = fr_gp.sigma
    sigma_hy = fr_hy.sigma
    # mcmc_sigma_full is over PARAM_NAMES (length 11); pull the matching subset.
    sigma_mcmc = np.array(
        [mcmc_sigma_full[PARAM_NAMES.index(n)] for n in fisher_param_names],
        dtype=float,
    )
    routes = {pn: ("PySR" if refits[pn] is not None else "GP-slice (gated)")
              for pn in fisher_param_names}

    rows = []
    for i, pn in enumerate(fisher_param_names):
        sg, sh, sm = float(sigma_gp[i]), float(sigma_hy[i]), float(sigma_mcmc[i])
        ratio_hy_gp = sh / sg if sg > 0 else float("nan")
        ratio_hy_mcmc = sh / sm if sm > 0 else float("nan")
        ratio_gp_mcmc = sg / sm if sm > 0 else float("nan")
        rows.append(dict(
            param=pn, route=routes[pn],
            sigma_gp=sg, sigma_hy=sh, sigma_mcmc=sm,
            ratio_hy_gp=ratio_hy_gp,
            ratio_hy_mcmc=ratio_hy_mcmc,
            ratio_gp_mcmc=ratio_gp_mcmc,
        ))
    md_lines = [
        "# Off-fid closure at θ_target_simdat (Data Index 15)",
        f"emulator: {args.basedir}",
        f"θ_target_simdat (with dtau0 → 0): see fisher_at_target.npz",
        f"z range: [{z_min}, {z_max}]; KSData k-grid: {len(ks_k_grid)} bins",
        f"hybrid-vs-GP at θ_target (mid-z): max |Δ/P_F| = {rel_at_target*100:.2f}%",
        f"priors: {priors_sigma if priors_sigma else 'none'}",
        "",
        "| param | route | σ_GP | σ_PySR | σ_MCMC | PySR/GP | PySR/MCMC | GP/MCMC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['param']} | {r['route']} | "
            f"{r['sigma_gp']:.3g} | {r['sigma_hy']:.3g} | {r['sigma_mcmc']:.3g} | "
            f"**{r['ratio_hy_gp']:.2f}×** | "
            f"**{r['ratio_hy_mcmc']:.2f}×** | "
            f"{r['ratio_gp_mcmc']:.2f}× |"
        )
    md_lines += [
        "",
        "## Interpretation",
        "- `PySR/GP` ≈ 1: hybrid Fisher matches the GP Fisher at θ_target (faithful off-fid).",
        "- `PySR/MCMC` ≈ 1: hybrid σ matches the truth (final closure target).",
        "- `GP/MCMC` ≠ 1 → Gaussianity / boundary effects in the simdat MCMC, "
        "  not a pipeline failure. Useful sanity check for the Cramer-Rao bound.",
        "",
        "Diverging `PySR/GP` flags params where the per-1D + additive-Taylor "
        "extrapolation has wrong fid-curvature. Candidates for Phase 1.5 smart "
        "refit (ANOVA loss + restricted operators) or Phase 2 pair coupling.",
    ]
    scorecard_path = args.output / "scorecard.md"
    scorecard_path.write_text("\n".join(md_lines) + "\n")
    print(f"\nWrote {scorecard_path}")

    # 7) Cache Fisher results for replot.
    np.savez(
        args.output / "fisher_at_target.npz",
        param_names=np.array(fisher_param_names),
        theta_target=theta_target,
        sigma_gp=sigma_gp, sigma_hy=sigma_hy, sigma_mcmc=sigma_mcmc,
        cov_gp=fr_gp.cov, cov_hy=fr_hy.cov,
    )
    print(f"Wrote {args.output / 'fisher_at_target.npz'}")

    # 8) Make a simple bar-plot scorecard figure.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping bar plot.")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(fisher_param_names))
    w = 0.27
    ratios_pysr_mcmc = np.array([r["ratio_hy_mcmc"] for r in rows])
    ratios_gp_mcmc = np.array([r["ratio_gp_mcmc"] for r in rows])
    ratios_pysr_gp = np.array([r["ratio_hy_gp"] for r in rows])
    ax.bar(x - w, np.log10(np.maximum(ratios_gp_mcmc, 1e-3)),
           width=w, color="0.5", label="σ_GP / σ_MCMC")
    ax.bar(x, np.log10(np.maximum(ratios_pysr_mcmc, 1e-3)),
           width=w, color="#d62728", label="σ_PySR / σ_MCMC")
    ax.bar(x + w, np.log10(np.maximum(ratios_pysr_gp, 1e-3)),
           width=w, color="#1f77b4", label="σ_PySR / σ_GP")
    ax.axhline(0, color="black", lw=0.7, alpha=0.6)
    ax.axhline(np.log10(2.0), ls="--", color="0.4", lw=0.6, alpha=0.5)
    ax.axhline(np.log10(0.5), ls="--", color="0.4", lw=0.6, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(fisher_param_names, rotation=40, ha="right")
    ax.set_ylabel(r"$\log_{10}(\sigma\ \mathrm{ratio})$ at $\theta_\mathrm{target,simdat}$")
    ax.set_title("Off-fid closure: σ ratios at θ_target_simdat (KSData covariance)")
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.output / f"closure_at_target.{ext}")
    print(f"Wrote {args.output / 'closure_at_target.pdf'}")


if __name__ == "__main__":
    main()
