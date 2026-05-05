"""Multi-D Sobol hold-out validation: P̂_hybrid vs P_GP at all-θ-varied points.

The existing `deliverables.write_holdout_validation` is **per-1D** — for
each param p, samples θ_p across its prior and fixes everything else at
fid. That tests the per-1D PySR fits in their training regime and misses
the cross-coupling regime entirely. The Phase-1 hybrid (per-1D + Taylor)
is structurally a *first-order ANOVA* approximation, so a multi-D Sobol
test is the natural diagnostic for higher-order interactions.

This script Sobol-samples all 11 PRIYA params jointly within their
priors (with `dtau0 → 0`), evaluates the GP truth and the hybrid model
at each point + a fixed z, and reports per-row rel-err vs k. The mean
and 90th-percentile across rows are the headline numbers.

Run:
  PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia \\
  PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \\
      python scripts/holdout_multid.py \\
          --refits-dir results/refit_optionC_z2.6-4.2_phase1_5_v2/refits \\
          --pair-refits-dir results/refit_pair_z2.6-4.2/refits \\
          --output results/holdout_multid_phase2_v2 \\
          --n-sobol 64 --z-eval 3.6
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

from priya_forecast.parameters import (
    PARAM_NAMES,
    PARAMS_11D,
    fiducial_vector,
    get_param,
)
from priya_forecast.refit_taylor import MultiZAdditiveTaylorModel


REL_ERR_THRESHOLD = 0.05


def _load_and_gate_refits(refits_dir: Path) -> dict:
    refits = {pn: None for pn in PARAM_NAMES}
    for pname in PARAM_NAMES:
        path = refits_dir / f"{pname}.pkl"
        if path.exists():
            with open(path, "rb") as fh:
                refits[pname] = pickle.load(fh)
    # Apply same gate as multi_z_aggregate.py.
    for pname, r in list(refits.items()):
        if r is None:
            continue
        has_x0 = "x0" in r.equation_str
        lf_ok = (np.isfinite(r.lf_train_mean_rel_err)
                 and r.lf_train_mean_rel_err < REL_ERR_THRESHOLD)
        hf_ok = (np.isfinite(r.hf_train_mean_rel_err)
                 and r.hf_train_mean_rel_err < REL_ERR_THRESHOLD)
        if not (has_x0 and lf_ok and hf_ok):
            refits[pname] = None
    return refits


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--refits-dir", type=Path, required=True)
    p.add_argument("--pair-refits-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--basedir", type=Path,
                   default=Path("/nfs/turbo/umor-yueyingn/mfho/birdgroup/"
                                "lya_xq100/kodiaq_2_2_4_6-48-48"))
    p.add_argument("--z-eval", type=float, default=3.6)
    p.add_argument("--n-sobol", type=int, default=64,
                   help="Sobol points in the 11D θ cube (use a power of 2).")
    p.add_argument("--seed", type=int, default=123,
                   help="Sobol seed (different from training).")
    p.add_argument("--k-min", type=float, default=0.005)
    p.add_argument("--k-max", type=float, default=0.064)
    p.add_argument("--n-k", type=int, default=32)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    fid = np.array(fiducial_vector(), dtype=float)
    dtau0_idx = PARAM_NAMES.index("dtau0")
    fid[dtau0_idx] = 0.0  # Kim convention.
    k_grid = np.linspace(args.k_min, args.k_max, args.n_k)

    # Sobol over all 11 params. Each param mapped to its prior; dtau0 fixed at 0.
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=len(PARAMS_11D), seed=args.seed)
    u = sampler.random(n=args.n_sobol)
    theta_samples = np.empty((args.n_sobol, len(PARAMS_11D)), dtype=float)
    for i, pp in enumerate(PARAMS_11D):
        if pp.name == "dtau0":
            theta_samples[:, i] = 0.0  # fixed
        else:
            theta_samples[:, i] = pp.prior[0] + u[:, i] * (pp.prior[1] - pp.prior[0])

    print(f"Multi-D hold-out: n_sobol={args.n_sobol}, z={args.z_eval}, "
          f"k=[{args.k_min}, {args.k_max}] ({args.n_k} bins).")

    print("Loading kodiaq HF emulator...")
    from priya_forecast.models.gp_model import GPModel
    gp_hf = GPModel(basedir=args.basedir, fidelity="hf", kf=k_grid)

    # Load refits + build base hybrid.
    refits = _load_and_gate_refits(args.refits_dir)
    n_kept = sum(r is not None for r in refits.values())
    print(f"  refits kept (post-gate): {n_kept}/{len(PARAM_NAMES)}")
    z_grid = np.array([float(args.z_eval)])
    base = MultiZAdditiveTaylorModel(
        gp=gp_hf, fid=fid, refits=refits,
        k_grid=k_grid, z_grid=z_grid,
    )

    pair_refits_loaded = []
    if args.pair_refits_dir is not None:
        from priya_forecast.refit_pair import (
            MultiZPairCoupledModel, Refit2DPairResult,  # noqa: F401
        )
        for path in sorted(args.pair_refits_dir.glob("*.pkl")):
            with open(path, "rb") as fh:
                pair_refits_loaded.append(pickle.load(fh))
            print(f"  [pair] {pair_refits_loaded[-1].pair_names} loaded.")
    if pair_refits_loaded:
        from priya_forecast.refit_pair import MultiZPairCoupledModel
        hybrid = MultiZPairCoupledModel(base=base, pairs=pair_refits_loaded)
    else:
        hybrid = base

    rel_err = np.empty((args.n_sobol, len(k_grid)), dtype=float)
    for i in range(args.n_sobol):
        theta = theta_samples[i]
        truth = np.asarray(gp_hf.predict(theta, k_grid, args.z_eval), dtype=float)
        pred = np.asarray(hybrid.predict(theta, k_grid, args.z_eval), dtype=float)
        rel_err[i] = np.abs(pred - truth) / np.abs(truth)
        if (i + 1) % 8 == 0:
            print(f"  [{i+1}/{args.n_sobol}] mean rel-err this batch: "
                  f"{rel_err[max(0,i-7):i+1].mean()*100:.2f}%", flush=True)

    mean_rel = rel_err.mean(axis=0)
    p90_rel = np.percentile(rel_err, 90, axis=0)
    p99_rel = np.percentile(rel_err, 99, axis=0)
    max_rel = rel_err.max(axis=0)

    np.savez(
        args.output / "holdout_multid.npz",
        k_grid=k_grid, theta_samples=theta_samples, rel_err=rel_err,
        mean_rel=mean_rel, p90_rel=p90_rel, p99_rel=p99_rel, max_rel=max_rel,
        z_eval=float(args.z_eval), n_sobol=args.n_sobol,
        param_names=np.array(PARAM_NAMES),
        has_pair=int(bool(pair_refits_loaded)),
    )

    md = [
        "# Multi-D Sobol hold-out validation (all-θ-varied)",
        f"emulator: {args.basedir}",
        f"z={args.z_eval}, n_sobol={args.n_sobol} (11D, dtau0 fixed at 0)",
        f"k=[{args.k_min}, {args.k_max}] s/km, {args.n_k} bins",
        f"refits gated: {n_kept}/{len(PARAM_NAMES)} kept",
        f"pairs: {len(pair_refits_loaded)} loaded ({[p.pair_names for p in pair_refits_loaded]})",
        "",
        "| k_idx | k | mean rel-err | p90 | p99 | max |",
        "|---|---|---|---|---|---|",
    ]
    for ki in range(len(k_grid)):
        md.append(
            f"| {ki} | {k_grid[ki]:.4f} | "
            f"{mean_rel[ki]*100:.2f}% | "
            f"{p90_rel[ki]*100:.2f}% | "
            f"{p99_rel[ki]*100:.2f}% | "
            f"{max_rel[ki]*100:.2f}% |"
        )
    md += [
        "",
        f"**Aggregate** (across all k bins):",
        f"- mean rel-err: {mean_rel.mean()*100:.2f}%",
        f"- p90:          {p90_rel.mean()*100:.2f}%",
        f"- p99:          {p99_rel.mean()*100:.2f}%",
        f"- max:          {max_rel.max()*100:.2f}%",
        "",
        "Per-row mean rel-err sorted (worst 10 rows by mean rel-err):",
    ]
    row_means = rel_err.mean(axis=1)
    worst_order = np.argsort(-row_means)[:10]
    md.append("| rank | row | mean rel-err | θ_Ap | θ_alphaq | θ_herei |")
    md.append("|---|---|---|---|---|---|")
    ap_i = PARAM_NAMES.index("Ap")
    aq_i = PARAM_NAMES.index("alphaq")
    hi_i = PARAM_NAMES.index("herei")
    for rk, r in enumerate(worst_order):
        md.append(
            f"| {rk+1} | {r} | {row_means[r]*100:.2f}% | "
            f"{theta_samples[r, ap_i]:.3f} | "
            f"{theta_samples[r, aq_i]:.3f} | "
            f"{theta_samples[r, hi_i]:.3f} |"
        )

    (args.output / "holdout_multid.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {args.output / 'holdout_multid.md'}")
    print(f"Aggregate mean rel-err: {mean_rel.mean()*100:.3f}%")
    print(f"Aggregate p99 rel-err:  {p99_rel.mean()*100:.3f}%")

    # Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(k_grid, 0, p99_rel * 100, color="0.85",
                    label=f"p99 = {p99_rel.mean()*100:.2f}%")
    ax.fill_between(k_grid, 0, p90_rel * 100, color="0.7",
                    label=f"p90 = {p90_rel.mean()*100:.2f}%")
    ax.plot(k_grid, mean_rel * 100, color="#d62728", lw=1.5,
            label=f"mean = {mean_rel.mean()*100:.2f}%")
    ax.axhline(1.0, ls="--", color="0.4", lw=0.6, alpha=0.5,
               label="1% reference")
    ax.set_xlabel("k [s/km]")
    ax.set_ylabel("rel-err (%)")
    ax.set_title(
        f"Multi-D Sobol hold-out (n={args.n_sobol}, z={args.z_eval}, "
        f"{'with' if pair_refits_loaded else 'no'} pair coupling)"
    )
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.set_yscale("log")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.output / f"holdout_multid.{ext}")
    print(f"Wrote {args.output / 'holdout_multid.pdf'}")


if __name__ == "__main__":
    main()
