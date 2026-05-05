"""Run all PySR hypothesis experiments + produce diagnostic figures.

Outputs to `docs/figures/pysr_hypothesis/`:
  fig_h1_loss_function.png   — Fisher σ ratio: full prior vs near-fid training
  fig_h2_parsimony.png       — terms surviving + test MSE vs parsimony
  fig_h3_normalization.png   — raw P_F vs flux_norm fit quality (THE BIG ONE)
  fig_h4_operators.png       — polynomial-only vs polynomial+exp basis
  fig_h6_covariance.png      — 1D-product vs 1D-product+cross-terms
  fig_published_diagnosis.png — published-equation vs GP at fid + perturbation

This is the script behind `docs/PYSR_HYPOTHESIS.md`. Re-run when you
want to update the figures with a fresh seed or new data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_LYAEMU = Path("/home/mfho/student_projects/lya_emulator_full")
if _LYAEMU.exists():
    sys.path.insert(0, str(_LYAEMU))

from priya_forecast.pysr_hypothesis import (
    experiment_h1_loss_function,
    experiment_h2_parsimony,
    experiment_h3_normalization,
    experiment_h4_operators,
    experiment_h6_covariance_combine,
)

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures" / "pysr_hypothesis"


def _setup():
    OUT.mkdir(parents=True, exist_ok=True)


def fig_h1():
    r = experiment_h1_loss_function(seed=0)
    truth = r.extra["fisher_sigma_truth"]
    full_ratios = {p: r.fisher_sigma[p] / truth[p] for p in truth}
    near_ratios = {p: r.extra["fisher_sigma_near_fid_training"][p] / truth[p] for p in truth}
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    x = np.arange(len(truth))
    w = 0.35
    ax.bar(x - w/2, list(full_ratios.values()), w, label="full-prior training")
    ax.bar(x + w/2, list(near_ratios.values()), w, label="near-fid training")
    ax.set_xticks(x); ax.set_xticklabels(list(truth))
    ax.axhline(1.0, color="black", lw=0.5, ls="--")
    ax.set_ylabel(r"$\sigma_{\rm fit}\,/\,\sigma_{\rm truth}$")
    ax.set_title("H1: Fisher σ ratio — concentration of training near fid")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_h1_loss_function.png"); plt.close(fig)
    print(f"H1: full-prior σ_ns/σ_truth_ns = {full_ratios['ns']:.4f}, near-fid = {near_ratios['ns']:.4f}")


def fig_h2():
    r = experiment_h2_parsimony(seed=0)
    parsimonies = []
    kept = []
    losses = []
    for k, v in r.extra.items():
        # k like "parsimony_1e-03"
        p_str = k.replace("parsimony_", "")
        parsimonies.append(float(p_str))
        kept.append(v["kept_total"])
        losses.append(v["test_mse"])
    order = np.argsort(parsimonies)
    parsimonies = np.array(parsimonies)[order]
    kept = np.array(kept)[order]
    losses = np.array(losses)[order]
    fig, ax1 = plt.subplots(figsize=(6, 3.5), dpi=120)
    fig.patch.set_facecolor("white"); ax1.set_facecolor("white")
    ax1.plot(parsimonies + 1e-12, kept, "o-", color="C0", label="terms kept")
    ax1.set_xscale("symlog", linthresh=1e-4)
    ax1.set_xlabel("parsimony"); ax1.set_ylabel("# polynomial terms surviving", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    ax2.plot(parsimonies + 1e-12, losses, "s-", color="C3", label="test MSE")
    ax2.set_yscale("log"); ax2.set_ylabel("test MSE  [P_F²]", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax1.set_title("H2: parsimony pruning — mild is fine, aggressive breaks fit")
    ax1.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_h2_parsimony.png"); plt.close(fig)
    print(f"H2: parsimony 0 → MSE {losses[0]:.3g};  parsimony 1e-2 → MSE {losses[2]:.3g};  "
          f"parsimony 1e-1 → MSE {losses[3]:.3g}")


def fig_h3():
    r = experiment_h3_normalization(seed=0)
    raw = r.extra["test_mse_raw"]
    norm = r.extra["test_mse_norm_then_denorm"]
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.bar(["raw P_F\n(my framework's\n`mode: identity`)",
            "flux_norm\n(student's\nconvention)"],
           [raw, norm + 1e-30], color=["C3", "C2"])
    ax.set_yscale("log")
    ax.set_ylabel("test MSE  [P_F²]")
    ax.set_title("H3: normalization choice — 28 orders of magnitude difference")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "fig_h3_normalization.png"); plt.close(fig)
    print(f"H3: raw MSE {raw:.3g}  vs  flux_norm MSE {norm:.3g}")


def fig_h4():
    r = experiment_h4_operators(seed=0)
    poly = r.extra["test_mse_polynomial_only"]
    aug = r.extra["test_mse_with_exp_basis"]
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.bar(["polynomial-only\n(no exp/log)",
            "polynomial + exp(-c*k)\n(the right basis)"],
           [poly, aug + 1e-30], color=["C3", "C2"])
    ax.set_yscale("log")
    ax.set_ylabel("test MSE  [P_F²]")
    ax.set_title("H4: operator set — exp(-k) basis is critical for Lyα-shape P_F")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "fig_h4_operators.png"); plt.close(fig)
    print(f"H4: polynomial-only MSE {poly:.3g}  vs  with exp basis {aug:.3g}")


def fig_h6():
    r = experiment_h6_covariance_combine(seed=0)
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.bar(["1D-product\n(no ns × Ap terms)",
            "1D-product + cross-terms\n(explicit ns × Ap)"],
           [r.extra["mse_M0_no_cross_terms"],
            r.extra["mse_M1_with_cross_terms"]], color=["C3", "C2"])
    ax.set_ylabel("test MSE on non-separable truth  [P_F²]")
    ax.set_title(f"H6: covariance-aware combine — modest "
                 f"{r.extra['improvement_factor']:.2f}× improvement")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "fig_h6_covariance.png"); plt.close(fig)
    print(f"H6: M0 (no cross) {r.extra['mse_M0_no_cross_terms']:.3g}, "
          f"M1 (with cross) {r.extra['mse_M1_with_cross_terms']:.3g}, "
          f"factor {r.extra['improvement_factor']:.2f}")


def fig_published_diagnosis():
    """The smoking-gun figure: published ns equation vs GP across the prior range."""
    from priya_forecast.models.gp_model import GPModel
    from priya_forecast.models.pysr_model import compile_equation
    from priya_forecast.models.normalization import derive_from_gp
    from priya_forecast.parameters import fiducial_vector, get_param

    print("Loading real PRIYA GP (this takes a minute)...")
    gp = GPModel()
    fid = np.array(fiducial_vector(), dtype=float)
    k = np.linspace(0.001, 0.02, 35)
    z = 3.6
    norm = derive_from_gp(gp_model=gp, param_name="ns", z=z, k_grid=k, n_samples=64)
    ce = compile_equation(
        param_name="ns",
        raw_expression="((ns * k) - r) * 2.3955164",
        variables=["ns", "k", "r"], fix={"r": 0.8}, norm=norm,
        fiducial=get_param("ns").fid,
    )
    ns_grid = np.linspace(0.8, 1.05, 11)
    gp_at_kmid = []
    pysr_at_kmid = []
    k_idx = len(k) // 2
    for ns in ns_grid:
        theta = fid.copy(); theta[2] = ns
        gp_at_kmid.append(gp.predict(theta, k, z)[k_idx])
        pysr_at_kmid.append(ce.evaluate(theta_i=ns, k=k)[k_idx])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0), dpi=120)
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white"); ax.grid(alpha=0.3)
    ax1.plot(ns_grid, gp_at_kmid, "o-", label="GP (truth)", color="black")
    ax1.plot(ns_grid, pysr_at_kmid, "s--", label="PySR published eq", color="C3")
    ax1.axvline(0.983, color="grey", lw=0.5, ls=":")
    ax1.set_xlabel("ns"); ax1.set_ylabel(r"$P_F(k=k_{\rm mid})$")
    ax1.set_title("Published ns equation vs GP")
    ax1.legend()
    # Right: relative response.
    gp_rel = np.array(gp_at_kmid) / gp_at_kmid[len(ns_grid) // 2]
    pysr_rel = np.array(pysr_at_kmid) / pysr_at_kmid[len(ns_grid) // 2]
    ax2.plot(ns_grid, gp_rel, "o-", label="GP", color="black")
    ax2.plot(ns_grid, pysr_rel, "s--", label="PySR", color="C3")
    ax2.axvline(0.983, color="grey", lw=0.5, ls=":")
    ax2.axhline(1.0, color="grey", lw=0.5)
    ax2.set_xlabel("ns"); ax2.set_ylabel("P_F / P_F(fid)")
    ax2.set_title("Relative response — PySR slope is 3× too shallow")
    ax2.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig_published_diagnosis.png"); plt.close(fig)
    print(f"published diagnosis: at ns=1.04, GP relative = {gp_rel[-2]:.3f}, "
          f"PySR relative = {pysr_rel[-2]:.3f}")


def main():
    _setup()
    fig_h1()
    fig_h2()
    fig_h3()
    fig_h4()
    fig_h6()
    fig_published_diagnosis()
    print(f"\nFigures written to {OUT}/")


if __name__ == "__main__":
    main()
