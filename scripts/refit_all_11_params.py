"""KODIAQ-emulator MF-PySR pipeline with Option B normalization.

Updates from the previous student-replication run:
  - Emulator: KODIAQ `kodiaq_2_2_4_6-48-48` (production, paper
    arXiv:2509.18271). k-grid 0.005 → 0.064 s/km (production range).
  - PySR operators: NO sin/cos (oscillatory derivatives wreck Fisher
    conditioning, see `feedback_pysr_operators.md`). Use exp/log/square/
    sqrt/inv only.
  - Normalization: Option B (per-param 1D-local std). Each per-param
    refit normalizes with `(mean_global_or_local, std_local_per_param)`
    so weak-coupled params (hub, bhfeedback, ...) get ~1σ signal in
    flux_norm space.
  - Combine: `mode="local_anchored"` — `P_F(θ, k) = P_GP(fid, k) +
    Σ_i [r_i.predict(θ_i, k, 0.8) − r_i.predict(fid_i_phys, k, 0.8)]`.
    Each per-param contribution is in P_F units. Exact at fid.
  - 1pvar data: generated INLINE from LF + HF emulators (no HDF5
    dependency). Allows arbitrary k_grid.

Run:
  PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia \\
  PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \\
      python scripts/refit_all_11_params.py --z 3.6 \\
          --output results/refit_kodiaq_optionB_z3.6

Use `--validate-only` to stop after the per-param 1D fits + rel-err
report (no Fisher).
"""

from __future__ import annotations

import argparse
import json
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
from priya_forecast.parameters import (
    PARAM_NAMES,
    PARAMS_11D,
    fiducial_vector,
)
from priya_forecast.refit_1d_pysr import (
    DEFAULT_1PVAR_DIR,
    HF_RESOLUTION,
    LF_RESOLUTION,
    refit_1d_for_param,
    refit_1d_multiz_for_param,
)
from priya_forecast.refit_taylor import (
    AdditiveTaylorModel,
    HF_RESOLUTION_FOR_COMBINE,
    MultiZAdditiveTaylorModel,
    STUDENT_FID_NORM,
    compute_global_normalization,
)

# KODIAQ production emulator (KODIAQ-SQUAD + XQ-100, arXiv:2509.18271).
# Replaces the priya/InferenceLyaData emulator the previous run used.
DEFAULT_KODIAQ_BASEDIR = Path(
    "/nfs/turbo/umor-yueyingn/mfho/birdgroup/lya_xq100/kodiaq_2_2_4_6-48-48"
)
# Production k range: 0.005 - 0.064 s/km (per
# /home/mfho/lya_emulator_full/docs/SLURM_COMMANDS.md).
DEFAULT_K_MIN = 0.005
DEFAULT_K_MAX = 0.064
DEFAULT_N_K = 64


def _per_param_summary(refits: dict) -> list[str]:
    """Markdown summary: stats table + FULL equations (one per code block)."""
    lines = [
        "## Per-param fit statistics",
        "",
        "| param | complexity | flux_norm loss | LF rel-err | HF rel-err | LF max | HF max | x0? | x3? |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for pname in PARAM_NAMES:
        r = refits.get(pname)
        if r is None:
            lines.append(f"| {pname} | — | — | — | — | — | — | — | — |")
            continue
        eq = r.equation_str
        has_x0 = "✓" if "x0" in eq else "✗"
        has_x3 = "✓" if "x3" in eq else "—"
        lines.append(
            f"| {pname} | {r.pareto_complexity} | {r.pareto_loss:.3g} | "
            f"{r.lf_train_mean_rel_err*100:.2f}% | "
            f"{r.hf_train_mean_rel_err*100:.2f}% | "
            f"{r.lf_train_max_rel_err*100:.2f}% | "
            f"{r.hf_train_max_rel_err*100:.2f}% | {has_x0} | {has_x3} |"
        )
    # Full equations as separate code blocks.
    lines.append("")
    lines.append("## Full equations")
    lines.append("")
    lines.append(
        "Inputs: `x0 = (theta - prior_lo)/(prior_hi - prior_lo)`, "
        "`x1 = (k - k_min)/(k_max - k_min)`, "
        "`x2 = resolution` (LF=0.4, HF=0.8), "
        "`x3 = (z - z_min)/(z_max - z_min)` (multi-z fits only)."
    )
    lines.append("")
    for pname in PARAM_NAMES:
        r = refits.get(pname)
        if r is None:
            continue
        lines.append(f"### `{pname}`  (θ_fid = {r.fid_value:.4g}, complexity = {r.pareto_complexity}, flux_norm loss = {r.pareto_loss:.3g})")
        lines.append("")
        lines.append("```")
        lines.append(r.equation_str)
        lines.append("```")
        lines.append("")
    return lines


def _resolution_correction_equations(
    refits: dict,
    fid_full: np.ndarray,
    output_dir: Path,
) -> None:
    """Symbolic HF/LF ratio per param — paper-form.

    For each per-param eq `f(x0, x1, x2, x3)`, report:
        R_i(x0, x1, x3) = [f(x0, x1, 0.8, x3) * std + mean]
                        / [f(x0, x1, 0.4, x3) * std + mean]

    where (mean, std) come from the per-(z, k) MultiZNormalizationSpec
    (or per-k for single-z). When evaluated at θ = fid_norm and
    z = z_norm_specific, we get the (k, z) shape of the resolution
    lift; varying θ shows the θ-coupling of the correction.

    Saves `resolution_correction_equations.md` with sympy-simplified
    expressions for each parameter.
    """
    import sympy as sp
    md = [
        "# Resolution correction equations per dimension",
        "",
        "For each parameter $i$, the LF→HF resolution correction is the",
        "multiplicative ratio of the per-param emulator's HF and LF",
        "predictions at the same $(\\theta, k, z)$:",
        "",
        "$$R_i(\\theta, k, z) \\;=\\; \\frac{P_F^{HF}(\\theta_i, k, z)}"
        "{P_F^{LF}(\\theta_i, k, z)} \\;=\\; \\frac{f_i(x_0, x_1, 0.8, x_3)\\,"
        "\\sigma_F(k, z) + \\mu_F(k, z)}{f_i(x_0, x_1, 0.4, x_3)\\,"
        "\\sigma_F(k, z) + \\mu_F(k, z)}$$",
        "",
        "where $f_i$ is the trained PySR equation, $(\\mu_F, \\sigma_F)$ is the",
        "per-(k, z) anchor / std from the LF emulator, and the inputs are",
        "min-max-normalized: $x_0 = (\\theta - \\theta_\\mathrm{lo})/(\\theta_\\mathrm{hi}-\\theta_\\mathrm{lo})$,",
        "$x_1 = (k - k_\\mathrm{min})/(k_\\mathrm{max}-k_\\mathrm{min})$,",
        "$x_3 = (z - z_\\mathrm{min})/(z_\\mathrm{max}-z_\\mathrm{min})$.",
        "",
        "Below we substitute $x_2 = 0.8$ (HF) and $x_2 = 0.4$ (LF) and report",
        "the **simplified PySR expression in flux_norm space at each",
        "fidelity** so the reader can see exactly how each parameter's",
        "equation depends on $x_2$:",
        "",
    ]
    for pname in PARAM_NAMES:
        r = refits.get(pname)
        if r is None:
            continue
        try:
            expr = sp.sympify(r.equation_str)
            x2 = sp.Symbol("x2")
            expr_hf = sp.simplify(expr.subs(x2, sp.Float(HF_RESOLUTION)))
            expr_lf = sp.simplify(expr.subs(x2, sp.Float(LF_RESOLUTION)))
            ratio_norm = sp.simplify(expr_hf - expr_lf)  # additive form in flux_norm
        except Exception as e:
            md.append(f"### {pname}: failed to sympify ({e})")
            md.append("")
            continue
        md.append(f"### `{pname}` (θ_fid = {r.fid_value:.4g})")
        md.append("")
        md.append("Trained equation (4-input):")
        md.append("```")
        md.append(r.equation_str)
        md.append("```")
        md.append("")
        md.append(f"At HF (x₂ = 0.8):")
        md.append("```")
        md.append(str(expr_hf))
        md.append("```")
        md.append(f"At LF (x₂ = 0.4):")
        md.append("```")
        md.append(str(expr_lf))
        md.append("```")
        md.append("Resolution correction in flux_norm space (HF − LF, simplified):")
        md.append("```")
        md.append(str(ratio_norm))
        md.append("```")
        md.append("")
    (output_dir / "resolution_correction_equations.md").write_text(
        "\n".join(md) + "\n"
    )


def _resolution_correction_per_dim(
    refits: dict,
    k_grid: np.ndarray,
    fid_full: np.ndarray,
) -> dict:
    """Per-param LF→HF resolution correction at θ=fid_phys (paper-form).

    Reports the correction in the **paper-friendly multiplicative form**:

        R_i(k) = P_F^HF(fid_i, k) / P_F^LF(fid_i, k) = predict(fid, k, 0.8)
                                                       / predict(fid, k, 0.4)

    R is the multiplicative factor that lifts an LF prediction to HF at
    θ = fid_i_phys. Numerically R ∈ [≈0.95, ≈1.05] (order-1 ratio).

    Also reports the additive form (Δ_PF = HF − LF) and the underlying
    flux_norm difference, for users who want absolute numbers.
    """
    out = {}
    for pname, r in refits.items():
        if r is None:
            continue
        i = PARAM_NAMES.index(pname)
        fid_phys = float(fid_full[i])
        eq_hf_norm = r.predict_normalized(theta_phys=fid_phys, k=k_grid,
                                          resolution=HF_RESOLUTION)
        eq_lf_norm = r.predict_normalized(theta_phys=fid_phys, k=k_grid,
                                          resolution=LF_RESOLUTION)
        p_hf = r.predict(theta_phys=fid_phys, k=k_grid, resolution=HF_RESOLUTION)
        p_lf = r.predict(theta_phys=fid_phys, k=k_grid, resolution=LF_RESOLUTION)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(p_lf) > 0, p_hf / p_lf, np.nan)
        out[pname] = dict(
            fid_phys=fid_phys,
            ratio_hf_over_lf=ratio.tolist(),
            delta_pf=(p_hf - p_lf).tolist(),
            delta_flux_norm=(eq_hf_norm - eq_lf_norm).tolist(),
            p_lf=p_lf.tolist(),
            p_hf=p_hf.tolist(),
        )
    return out


def _write_resolution_correction_summary(
    res_corr: dict,
    k_grid: np.ndarray,
    output_dir: Path,
) -> None:
    """Markdown table + grid plot of the LF→HF resolution correction per-D.

    Paper-friendly multiplicative form: `R_i(k) = P_F^HF / P_F^LF`.
    Both ratio and additive views are tabulated; the figure plots the
    HF/LF ratio.
    """
    md_lines = [
        "# Resolution correction per dimension",
        "",
        "Per-param LF→HF lift evaluated at each parameter's physical "
        "fiducial value, with all other params held at fid.",
        "",
        "**Paper form (multiplicative)**:",
        "",
        "    R_i(k) = P_F^HF(fid_i, k) / P_F^LF(fid_i, k)",
        "          = predict(fid_i, k, x2=0.8) / predict(fid_i, k, x2=0.4)",
        "",
        "R is the multiplicative correction that lifts the LF emulator's",
        "prediction to the HF emulator's prediction. Numerically R ≈ 0.95–1.05.",
        "",
        "**Additive form** (Δ_PF = HF − LF in physical P_F units) is also",
        "exported in `resolution_correction.json`.",
        "",
        "| param | R_min | R_max | R_mean | Δ_PF abs-max | Δ_PF mean (signed) |",
        "|---|---|---|---|---|---|",
    ]
    for pname, info in res_corr.items():
        ratio = np.asarray(info["ratio_hf_over_lf"], dtype=float)
        d_pf = np.asarray(info["delta_pf"], dtype=float)
        finite = np.isfinite(ratio)
        md_lines.append(
            f"| {pname} | {ratio[finite].min():.4f} | "
            f"{ratio[finite].max():.4f} | {ratio[finite].mean():.4f} "
            f"| {np.abs(d_pf).max():.3g} | {d_pf.mean():+.3g} |"
        )
    (output_dir / "resolution_correction.md").write_text(
        "\n".join(md_lines) + "\n"
    )

    # Grid figure: HF/LF ratio per param. 3 × 4 layout, last empty.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(res_corr)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.4 * rows),
                             sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, (pname, info) in enumerate(res_corr.items()):
        ax = axes[i // cols][i % cols]
        ax.set_visible(True)
        ratio = np.asarray(info["ratio_hf_over_lf"])
        ax.plot(k_grid, ratio, color="#d62728", lw=1.5)
        ax.axhline(1.0, color="gray", lw=0.5, alpha=0.7)
        ax.set_title(f"{pname}  (θ_fid={info['fid_phys']:.4g})", fontsize=9)
        ax.set_xlabel("k [s/km]", fontsize=8)
        ax.set_ylabel(r"$R(k)\,=\,P_F^{HF}/P_F^{LF}$", fontsize=8)
        ax.set_xscale("log")
        ax.tick_params(axis="both", which="major", labelsize=7)
    fig.suptitle(
        "Per-dim resolution correction at θ=fid (HF/LF ratio)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "resolution_correction_grid.png", dpi=160,
                bbox_inches="tight")
    fig.savefig(output_dir / "resolution_correction_grid.pdf",
                bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--niter", type=int, default=20,
                   help="PySR iterations (student default 20).")
    p.add_argument("--maxsize", type=int, default=20,
                   help="PySR max equation size (student default 20).")
    p.add_argument("--maxdepth", type=int, default=10)
    p.add_argument("--n-points", type=int, default=50,
                   help="Inline 1pvar sweep size per param (student uses 50).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (student uses 42).")
    p.add_argument("--z", type=float, default=3.6)
    p.add_argument("--k-min", type=float, default=DEFAULT_K_MIN,
                   help=f"Min k (s/km). Default {DEFAULT_K_MIN} (kodiaq production).")
    p.add_argument("--k-max", type=float, default=DEFAULT_K_MAX,
                   help=f"Max k (s/km). Default {DEFAULT_K_MAX} (kodiaq production).")
    p.add_argument("--n-k", type=int, default=DEFAULT_N_K,
                   help=f"Number of k bins. Default {DEFAULT_N_K} (linear in k).")
    p.add_argument("--basedir", type=Path, default=DEFAULT_KODIAQ_BASEDIR,
                   help="Emulator basedir (KODIAQ production by default).")
    p.add_argument("--params", nargs="+", default=list(PARAM_NAMES),
                   help="Subset of params to refit (others skipped).")
    p.add_argument(
        "--fix-params",
        nargs="+",
        default=["dtau0"],
        help="Params held fixed at theta_fid in the Fisher comparison. "
             "Default ['dtau0'] — at single z, the (dtau0, tau0) "
             "mean-flux pair is degenerate; fixing dtau0 breaks it. "
             "Use [] to vary all 11.",
    )
    p.add_argument(
        "--fix-dtau0-to-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set dtau0 fid to 0 (Kim's mean-flux slope, an EXTERNAL "
             "measurement). Default True — avoids using the upstream "
             "best_par dtau0 = -0.009 which was fit against eBOSS DR14 "
             "P1D and would re-use the same data statistics in our "
             "forecast. Pass --no-fix-dtau0-to-zero to keep -0.009.",
    )
    p.add_argument(
        "--priors",
        choices=("production", "none"),
        default="production",
        help="Gaussian priors on the Fisher. 'production' applies the "
             "lya_emulator_full likelihood.py priors: hub σ=0.015, "
             "omegamh2 σ=0.001, bhfeedback σ=0.005. Default 'production'.",
    )
    p.add_argument(
        "--multi-z",
        action="store_true",
        help="Multi-z mode: 4-input PySR (θ, k, resolution, z) over a Sobol "
             "scatter in (θ, z); per-z normalization. Multi-z Fisher = Σ_z F_z.",
    )
    p.add_argument(
        "--z-min", type=float, default=2.6,
        help="Min z for multi-z mode (kodiaq production: 2.6).",
    )
    p.add_argument(
        "--z-max", type=float, default=4.2,
        help="Max z for multi-z mode (kodiaq production: 4.2).",
    )
    p.add_argument(
        "--n-total", type=int, default=225,
        help="Sobol points for multi-z 1pvar (default 225 = 25 sims/z × 9 z-bins).",
    )
    p.add_argument(
        "--rel-err-target",
        type=float,
        default=0.01,
        help="Target mean rel-err for each fidelity. Reported in the "
             "summary; not enforced as a hard fail.",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Stop after per-param 1D fits + rel-err report; skip Fisher.",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output / "refits"
    cache_dir.mkdir(exist_ok=True)

    z = args.z
    k_grid = np.linspace(args.k_min, args.k_max, args.n_k)
    fid = np.array(fiducial_vector(), dtype=float)
    if args.fix_dtau0_to_zero:
        fid[PARAM_NAMES.index("dtau0")] = 0.0

    print(f"Loading KODIAQ emulator at {args.basedir} (LF + HF) ...")
    from priya_forecast.models.gp_model import GPModel
    gp_lf = GPModel(basedir=args.basedir, fidelity="lf", kf=k_grid)
    gp_hf = GPModel(basedir=args.basedir, fidelity="hf", kf=k_grid)
    print(f"  k_grid: linspace({args.k_min}, {args.k_max}, {args.n_k}) s/km")

    # Step 2: per-param 1D PySR refits (inline 1pvar gen, Option B local std).
    refits: dict = {pn: None for pn in PARAM_NAMES}
    pysr_kwargs = dict(
        niterations=args.niter, maxsize=args.maxsize, maxdepth=args.maxdepth,
    )
    if args.multi_z:
        print(f"\nMulti-z refit: {len(args.params)} params, z=[{args.z_min}, {args.z_max}], "
              f"n_total={args.n_total} Sobol scatter, niter={args.niter}, "
              f"maxsize={args.maxsize}.")
    else:
        print(f"\nRefitting 1D PySR for {len(args.params)} params: "
              f"niter={args.niter}, maxsize={args.maxsize}, maxdepth={args.maxdepth}, "
              f"n_points={args.n_points}, ops without sin/cos.")
    for pname in args.params:
        cache_path = cache_dir / f"{pname}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as fh:
                refits[pname] = pickle.load(fh)
            r = refits[pname]
            print(f"  [cache] {pname}: complexity={r.pareto_complexity}, "
                  f"LF rel-err={r.lf_train_mean_rel_err*100:.2f}%, "
                  f"HF rel-err={r.hf_train_mean_rel_err*100:.2f}%")
            continue
        print(f"  fitting {pname}...", flush=True)
        t0 = time.time()
        if args.multi_z:
            r = refit_1d_multiz_for_param(
                param_name=pname, z_min=args.z_min, z_max=args.z_max,
                k_grid=k_grid, gp_lf=gp_lf, gp_hf=gp_hf,
                n_total=args.n_total,
                pysr_kwargs=pysr_kwargs, seed=args.seed,
            )
        else:
            r = refit_1d_for_param(
                param_name=pname, z=z, k_grid=k_grid,
                gp_lf=gp_lf, gp_hf=gp_hf,
                n_points=args.n_points,
                pysr_kwargs=pysr_kwargs, seed=args.seed,
            )
        elapsed = time.time() - t0
        with open(cache_path, "wb") as fh:
            pickle.dump(r, fh)
        refits[pname] = r
        print(f"  [{elapsed:.0f}s] {pname}: complexity={r.pareto_complexity}, "
              f"flux_norm loss={r.pareto_loss:.3g}, "
              f"LF rel-err={r.lf_train_mean_rel_err*100:.2f}%, "
              f"HF rel-err={r.hf_train_mean_rel_err*100:.2f}%", flush=True)

    # Step 3: per-param summary + rel-err gating report. Use the shared
    # deliverables module (same as multi_z_aggregate.py) so single-z and
    # multi-z runs produce identical artifact formats with prettified
    # equation strings (θ, k, r, z instead of x0, x1, x2, x3).
    from priya_forecast.deliverables import (
        per_param_summary_lines,
        write_resolution_correction_outputs,
        write_resolution_correction_equations,
    )
    target_rel = args.rel_err_target
    n_below = sum(
        1 for r in refits.values()
        if r is not None and max(r.lf_train_mean_rel_err, r.hf_train_mean_rel_err) < target_rel
    )
    n_total = sum(1 for r in refits.values() if r is not None)
    gate_lines = [
        f"## Per-param 1D PySR fits at z = {z}",
        "",
        f"Target: mean rel-err < {target_rel*100:.1f}% on each fidelity.",
        f"Pass: {n_below} / {n_total} params.",
        "",
        *per_param_summary_lines({pn: r for pn, r in refits.items() if r is not None}),
        "",
    ]
    (args.output / "per_param_summary.md").write_text("\n".join(gate_lines) + "\n")
    print("\n".join(gate_lines))

    refits_loaded = {pn: r for pn, r in refits.items() if r is not None}
    write_resolution_correction_outputs(refits_loaded, k_grid, fid, args.output)
    write_resolution_correction_equations(refits_loaded, args.output)
    from priya_forecast.deliverables import (
        write_holdout_validation,
        write_param_variation_resolution_correction,
    )
    write_param_variation_resolution_correction(refits_loaded, k_grid, args.output)
    try:
        write_holdout_validation(
            refits_loaded, gp_lf=gp_lf, gp_hf=gp_hf,
            k_grid=k_grid, output_dir=args.output, n_holdout=50,
        )
    except Exception as e:
        print(f"  (hold-out validation skipped: {e})")

    if args.validate_only:
        print(f"\n--validate-only: stopping after per-param fits. "
              f"{n_below}/{n_total} below {target_rel*100:.1f}% target.")
        return

    if n_below < n_total:
        print(f"\nWarning: {n_total - n_below}/{n_total} params miss the "
              f"<{target_rel*100:.1f}% target. Continuing to Fisher anyway "
              f"(use --validate-only to gate this).")

    # Step 4: build the hybrid combine.
    if args.multi_z:
        z_grid_full = np.array([2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])
        z_grid_use = z_grid_full[
            (z_grid_full >= args.z_min - 1e-6) & (z_grid_full <= args.z_max + 1e-6)
        ]
        hybrid = MultiZAdditiveTaylorModel(
            gp=gp_hf, fid=fid, refits=refits,
            k_grid=k_grid, z_grid=z_grid_use,
        )
        # Sanity: at fid, hybrid == HF GP exactly at every z.
        max_rel = 0.0
        for z_check in z_grid_use:
            p_hy = hybrid.predict(fid, k_grid, float(z_check))
            p_gp = gp_hf.predict(fid, k_grid, float(z_check))
            max_rel = max(max_rel, float(np.max(np.abs(p_hy - p_gp) / p_gp)))
        print(f"  hybrid vs HF GP at fid (max over z): {max_rel*100:.4f}%")
    else:
        hybrid = AdditiveTaylorModel(
            gp=gp_hf, fid=fid, refits=refits, global_norm=None,
            k_grid=k_grid, z=z, mode="local_anchored",
        )
        p_hy = hybrid.predict(fid, k_grid, z)
        p_gp = gp_hf.predict(fid, k_grid, z)
        rel = np.max(np.abs(p_hy - p_gp) / p_gp)
        print(f"  hybrid vs HF GP at fid: max rel diff = {rel*100:.4f}%  "
              f"(local_anchored mode: should be ~0)")

    # Step 5: Fisher with synthetic 5%-of-P_F diagonal cov on the kodiaq
    # k-grid. Drop fixed params (default `dtau0`) — at single z the
    # (dtau0, tau0) mean-flux pair is degenerate; fixing dtau0 breaks it.
    fix_set = set(args.fix_params or [])
    fisher_params = tuple(p for p in PARAMS_11D if p.name not in fix_set)
    fisher_param_names = [p.name for p in fisher_params]
    param_indices = [PARAM_NAMES.index(p.name) for p in fisher_params]
    print(f"\nFisher params (n={len(fisher_params)}): {fisher_param_names}")
    if fix_set:
        fix_summary = ", ".join(
            f"{p}={fid[PARAM_NAMES.index(p)]:.4g}" for p in sorted(fix_set)
        )
        print(f"  fixed at theta_fid: {fix_summary}")
    print(f"  cov: synthetic diagonal, σ_k = 5% · P_F(fid, k) on k_grid.")
    # Gaussian priors per `~/lya_emulator_full/lyaemu/likelihood.py`.
    if args.priors == "production":
        priors_sigma = {
            "hub": 0.015,         # centered at 0.7 (cosmic-variance prior)
            "omegamh2": 0.001,    # Planck 2018, arxiv 1807.06209
            "bhfeedback": 0.005,  # centered at 0.05
        }
        # Filter to only varying params (skip if held fixed).
        priors_sigma = {p: s for p, s in priors_sigma.items() if p not in fix_set}
        print(f"  priors (production): {priors_sigma}")
    else:
        priors_sigma = None
    if args.multi_z:
        from priya_forecast.fisher import (
            compute_fisher_F_phys, combine_fisher_phys_arrays,
        )
        F_gp_list, F_hy_list = [], []
        for z_bin in z_grid_use:
            lk_gp_z = GaussianLikelihood(
                model=gp_hf, z=float(z_bin), mock_data="gp", theta_fid=fid,
                k_grid=k_grid, cov_diag_frac=0.05,
            )
            F_gp_list.append(compute_fisher_F_phys(
                likelihood=lk_gp_z, theta_fid=fid, params=fisher_params,
                param_indices=param_indices,
                step_frac=0.02, rel_tol=0.05, max_halvings=2,
            ))
            lk_hy_z = GaussianLikelihood(
                model=hybrid, z=float(z_bin), mock_data="gp", theta_fid=fid,
                k_grid=k_grid, cov_diag_frac=0.05,
            )
            F_hy_list.append(compute_fisher_F_phys(
                likelihood=lk_hy_z, theta_fid=fid, params=fisher_params,
                param_indices=param_indices,
                step_frac=0.02, rel_tol=0.05, max_halvings=2,
            ))
        theta_fid_subset = np.array(
            [fid[PARAM_NAMES.index(p.name)] for p in fisher_params]
        )
        fr_gp = combine_fisher_phys_arrays(
            F_gp_list, params=fisher_params, theta_fid=theta_fid_subset,
            priors_sigma=priors_sigma,
        )
        fr_hy = combine_fisher_phys_arrays(
            F_hy_list, params=fisher_params, theta_fid=theta_fid_subset,
            priors_sigma=priors_sigma,
        )
    else:
        lk_gp = GaussianLikelihood(
            model=gp_hf, z=z, mock_data="gp", theta_fid=fid,
            k_grid=k_grid, cov_diag_frac=0.05,
        )
        fr_gp = fisher_matrix(
            likelihood=lk_gp, theta_fid=fid, params=fisher_params,
            param_indices=param_indices,
            step_frac=0.02, rel_tol=0.05, max_halvings=2,
            priors_sigma=priors_sigma,
        )
        lk_hy = GaussianLikelihood(
            model=hybrid, z=z, mock_data="gp", theta_fid=fid,
            k_grid=k_grid, cov_diag_frac=0.05,
        )
        fr_hy = fisher_matrix(
            likelihood=lk_hy, theta_fid=fid, params=fisher_params,
            param_indices=param_indices,
            step_frac=0.02, rel_tol=0.05, max_halvings=2,
            priors_sigma=priors_sigma,
        )

    # Scorecard. Iterate over fisher_params (varying); fixed params are
    # noted at the bottom.
    target = ("Ap", "ns", "tau0", "dtau0")
    lines = [
        "# Forecast: refit 1D PySR × additive-Taylor combine (mode=local_anchored)",
        f"emulator: {args.basedir}",
        f"z = {z}, niter = {args.niter}, maxsize = {args.maxsize}, "
        f"resolution feature = (LF={LF_RESOLUTION}, HF={HF_RESOLUTION}).",
        f"k-grid: linspace({args.k_min}, {args.k_max}, {args.n_k}) s/km, "
        f"cov: 5%·P_F(fid, k) diagonal.",
        f"fixed params: {sorted(fix_set) if fix_set else '(none — all 11 vary)'}.",
        "",
        "| param | GP σ | hybrid σ | hybrid/GP ratio | LF rel-err | HF rel-err | complexity |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, p in enumerate(fisher_params):
        pname = p.name
        r = refits.get(pname)
        sigma_gp = fr_gp.sigma[i]
        sigma_hy = fr_hy.sigma[i]
        ratio = sigma_hy / sigma_gp if sigma_gp > 0 else float("inf")
        if r is not None:
            lf = f"{r.lf_train_mean_rel_err*100:.2f}%"
            hf = f"{r.hf_train_mean_rel_err*100:.2f}%"
            cplx = str(r.pareto_complexity)
        else:
            lf = hf = "—"; cplx = "—"
        lines.append(
            f"| {pname} | {sigma_gp:.3g} | {sigma_hy:.3g} | "
            f"**{ratio:.2f}×** | {lf} | {hf} | {cplx} |"
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
        theta_fid_full=fid,
        theta_fid_subset=fr_gp.theta_fid,
    )

    # Corner plot: GP (truth) vs hybrid Gaussian Fisher posteriors.
    try:
        from priya_forecast.plotting import plot_fisher_corner
        plot_fisher_corner(
            fr_gp=fr_gp, fr_hybrid=fr_hy,
            params=fisher_params,
            output_path=args.output / "corner.pdf",
        )
        print(f"Corner plot: {args.output / 'corner.pdf'}")
    except Exception as e:
        print(f"  (corner plot skipped: {e})")

    print(f"\nRefits cached at {cache_dir}/")
    print(f"Per-param summary: {args.output / 'per_param_summary.md'}")
    print(f"Resolution correction: {args.output / 'resolution_correction_per_dim.json'}")
    print(f"Scorecard: {args.output / 'scorecard.md'}")


if __name__ == "__main__":
    main()
