"""Forecast paper deliverables — per-param summary, resolution correction
markdown / JSON / grid figure / symbolic equations.

Used by both the single-z (`scripts/refit_all_11_params.py`) and multi-z
(`scripts/multi_z_aggregate.py`) drivers so both produce the same
paper-grade outputs:

  - `per_param_summary.md`              : stats table + full equations.
  - `resolution_correction.md`          : HF/LF ratio table.
  - `resolution_correction.json`        : full per-k arrays.
  - `resolution_correction_grid.{png,pdf}` : 11-panel HF/LF ratio figure.
  - `resolution_correction_equations.md`: symbolic eq at HF and LF +
                                          simplified HF − LF.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import re

from priya_forecast.parameters import PARAM_NAMES
from priya_forecast.refit_1d_pysr import HF_RESOLUTION, LF_RESOLUTION


# Pretty names for the per-1D PySR inputs:
#   x0 → θ  (the parameter being fit)
#   x1 → k  (k_norm)
#   x2 → r  (resolution; LF=0.4, HF=0.8)
#   x3 → z  (z_norm; multi-z fits only)
PRETTY_NAMES_1D = ["θ", "k", "r", "z"]


def _prettify_equation(eq: str, names: list[str] = PRETTY_NAMES_1D) -> str:
    """Replace `xN` tokens with human-readable names. Substring-safe via
    word-boundary regex (won't corrupt e.g. `x10` if such a token ever
    appears).

    `names[i]` is the replacement for `xi`. Indices not in `names` are
    left as-is.
    """
    out = eq
    for i, name in enumerate(names):
        out = re.sub(rf"\bx{i}\b", name, out)
    return out


def per_param_summary_lines(refits: dict, *, header_z: float | None = None) -> list[str]:
    """Markdown summary: stats table + FULL equations (one per code block)."""
    lines = []
    if header_z is not None:
        lines.append(f"## Per-param 1D PySR fits at z = {header_z}")
        lines.append("")
    lines.extend([
        "## Per-param fit statistics",
        "",
        "| param | complexity | flux_norm loss | LF rel-err | HF rel-err | LF max | HF max | x0? | x3? |",
        "|---|---|---|---|---|---|---|---|---|",
    ])
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
        lines.append(
            f"### `{pname}`  (θ_fid = {r.fid_value:.4g}, complexity = "
            f"{r.pareto_complexity}, flux_norm loss = {r.pareto_loss:.3g})"
        )
        lines.append("")
        lines.append("```")
        lines.append(_prettify_equation(r.equation_str))
        lines.append("```")
        lines.append("")
    return lines


def resolution_correction_per_dim(
    refits: dict, k_grid: np.ndarray, fid_full: np.ndarray,
    *, z_eval: float | None = None,
) -> dict:
    """Per-param HF/LF ratio at θ=fid_phys (paper form).

    Reports:
      - `ratio_hf_over_lf`: P_F^HF / P_F^LF per k (the multiplicative correction).
      - `delta_pf`: HF − LF per k.
      - `delta_flux_norm`: eq value HF − LF (raw PySR output, before de-norm).

    For multi-z refits, evaluate at `z_eval` (default: refit's z midpoint).
    For single-z refits, `z_eval` is ignored.
    """
    out = {}
    for pname, r in refits.items():
        if r is None:
            continue
        i = PARAM_NAMES.index(pname)
        fid_phys = float(fid_full[i])
        # Both predict_normalized and predict accept optional z (ignored for single-z).
        z_arg = z_eval if (getattr(r, "is_multiz", False) and z_eval is not None) else None
        eq_hf_norm = r.predict_normalized(
            theta_phys=fid_phys, k=k_grid, resolution=HF_RESOLUTION, z=z_arg,
        )
        eq_lf_norm = r.predict_normalized(
            theta_phys=fid_phys, k=k_grid, resolution=LF_RESOLUTION, z=z_arg,
        )
        p_hf = r.predict(theta_phys=fid_phys, k=k_grid, resolution=HF_RESOLUTION, z=z_arg)
        p_lf = r.predict(theta_phys=fid_phys, k=k_grid, resolution=LF_RESOLUTION, z=z_arg)
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


def write_resolution_correction_outputs(
    refits: dict, k_grid: np.ndarray, fid_full: np.ndarray,
    output_dir: Path,
    *, z_eval: float | None = None,
) -> None:
    """Generate all 4 resolution-correction deliverables in `output_dir`.

    `z_eval` is the z at which to evaluate the HF/LF ratio for multi-z
    refits (default = z_min of the first available refit's range).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if z_eval is None:
        for r in refits.values():
            if r is not None and getattr(r, "is_multiz", False):
                z_eval = float((r.z_min + r.z_max) / 2.0)
                break
    res_corr = resolution_correction_per_dim(
        refits, k_grid, fid_full, z_eval=z_eval,
    )

    # JSON dump of full per-k arrays.
    with open(output_dir / "resolution_correction.json", "w") as fh:
        json.dump({
            "k_grid": np.asarray(k_grid, dtype=float).tolist(),
            "x2_lf": LF_RESOLUTION,
            "x2_hf": HF_RESOLUTION,
            "z_eval": z_eval,
            "evaluated_at": "fid_i_phys (the actual physical fiducial of "
                            "each parameter; not the student's 0.5-norm "
                            "approximation).",
            "per_param": res_corr,
        }, fh, indent=2)

    # Markdown summary table.
    md_lines = [
        "# Resolution correction per dimension",
        "",
        "Per-param LF→HF lift evaluated at each parameter's physical "
        "fiducial value, with all other params held at fid.",
        "",
        f"z_eval = {z_eval}" if z_eval is not None else "single-z refit",
        "",
        "**Paper form (multiplicative)**:",
        "",
        "    R_i(k) = P_F^HF(fid_i, k) / P_F^LF(fid_i, k)",
        "",
        "R is the multiplicative correction that lifts the LF emulator's",
        "prediction to the HF emulator's prediction at θ = fid_i_phys.",
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

    # Grid figure (HF/LF ratio per param). Two-block layout:
    #   - cosmology + mean-flux block: dtau0, tau0, ns, Ap, hub, omegamh2
    #   - astro / IGM-thermal block:    herei, heref, alphaq, hireionz, bhfeedback
    # Per-panel y-axis (no sharey) so each ratio's own dynamic range is visible.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cosmo_block = ("dtau0", "tau0", "ns", "Ap", "hub", "omegamh2")
    astro_block = ("herei", "heref", "alphaq", "hireionz", "bhfeedback")

    def _make_grid_figure(block_names: tuple[str, ...], title: str, suffix: str) -> None:
        present = [(n, res_corr[n]) for n in block_names if n in res_corr]
        if not present:
            return
        cols = min(3, len(present))
        rows = (len(present) + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols, figsize=(3.0 * cols, 2.4 * rows),
            sharex=True, sharey=False, squeeze=False,
        )
        for ax in axes.flat:
            ax.set_visible(False)
        for i, (pname, info) in enumerate(present):
            ax = axes[i // cols][i % cols]
            ax.set_visible(True)
            ratio = np.asarray(info["ratio_hf_over_lf"])
            ax.plot(k_grid, ratio, color="#d62728", lw=1.5)
            ax.axhline(1.0, color="gray", lw=0.5, alpha=0.7)
            # Per-panel y-axis: tighten around the actual ratio range.
            finite = np.isfinite(ratio)
            if finite.any():
                rmin, rmax = float(ratio[finite].min()), float(ratio[finite].max())
                pad = max(1e-3, 0.1 * (rmax - rmin))
                ax.set_ylim(rmin - pad, rmax + pad)
            ax.set_title(f"{pname}  (θ_fid={info['fid_phys']:.4g})", fontsize=9)
            if i // cols == rows - 1:
                ax.set_xlabel("k [s/km]", fontsize=8)
            if i % cols == 0:
                ax.set_ylabel(r"$R(k)\,=\,P_F^{HF}/P_F^{LF}$", fontsize=8)
            ax.set_xscale("log")
            ax.tick_params(axis="both", which="major", labelsize=7)
        suptitle = title
        if z_eval is not None:
            suptitle += f" — z = {z_eval:.2f}"
        fig.suptitle(suptitle, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(output_dir / f"resolution_correction_grid_{suffix}.png",
                    dpi=160, bbox_inches="tight")
        fig.savefig(output_dir / f"resolution_correction_grid_{suffix}.pdf",
                    bbox_inches="tight")
        plt.close(fig)

    _make_grid_figure(
        cosmo_block, "HF/LF resolution correction — cosmology + mean-flux", "cosmo",
    )
    _make_grid_figure(
        astro_block, "HF/LF resolution correction — IGM thermal / astro", "astro",
    )


def write_param_variation_resolution_correction(
    refits: dict,
    k_grid: np.ndarray,
    output_dir: Path,
    *,
    z_eval: float | None = None,
    quantile_levels: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
) -> None:
    """For each param: HF/LF ratio R(k) at several θ-quantiles in its prior.

    Same cosmo/astro split as `write_resolution_correction_outputs`. Each
    panel shows multiple curves — one per chosen quantile of the prior —
    so the reader can see how the resolution correction depends on θ
    itself.

    Saves `resolution_correction_param_variation_{cosmo,astro}.{png,pdf}`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from priya_forecast.parameters import get_param

    cosmo_block = ("dtau0", "tau0", "ns", "Ap", "hub", "omegamh2")
    astro_block = ("herei", "heref", "alphaq", "hireionz", "bhfeedback")

    if z_eval is None:
        for r in refits.values():
            if r is not None and getattr(r, "is_multiz", False):
                z_eval = float((r.z_min + r.z_max) / 2.0)
                break

    cmap = plt.get_cmap("plasma")
    n_q = len(quantile_levels)
    colors = [cmap(i / max(1, n_q - 1)) for i in range(n_q)]

    def _make_grid(block_names: tuple[str, ...], title: str, suffix: str) -> None:
        present = [n for n in block_names if n in refits and refits[n] is not None]
        if not present:
            return
        cols = min(3, len(present))
        rows = (len(present) + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols, figsize=(3.2 * cols, 2.5 * rows),
            sharex=True, sharey=False, squeeze=False,
        )
        for ax in axes.flat:
            ax.set_visible(False)
        for i, pname in enumerate(present):
            r = refits[pname]
            ax = axes[i // cols][i % cols]
            ax.set_visible(True)
            p = get_param(pname)
            theta_values = [
                p.prior[0] + q * (p.prior[1] - p.prior[0])
                for q in quantile_levels
            ]
            for q, theta_phys, color in zip(quantile_levels, theta_values, colors):
                z_arg = z_eval if (getattr(r, "is_multiz", False) and z_eval is not None) else None
                p_hf = r.predict(theta_phys=theta_phys, k=k_grid, resolution=HF_RESOLUTION, z=z_arg)
                p_lf = r.predict(theta_phys=theta_phys, k=k_grid, resolution=LF_RESOLUTION, z=z_arg)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(np.abs(p_lf) > 0, p_hf / p_lf, np.nan)
                ax.plot(k_grid, ratio, color=color, lw=1.2,
                        label=f"q={q:.1f} (θ={theta_phys:.3g})")
            ax.axhline(1.0, color="gray", lw=0.5, alpha=0.7)
            ax.set_title(f"{pname}", fontsize=9)
            ax.set_xscale("log")
            if i // cols == rows - 1:
                ax.set_xlabel("k [s/km]", fontsize=8)
            if i % cols == 0:
                ax.set_ylabel(r"$R(k;\theta)$", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=7)
            if i == 0:
                ax.legend(fontsize=6, loc="best")
        suptitle = title
        if z_eval is not None:
            suptitle += f" — z = {z_eval:.2f}"
        fig.suptitle(suptitle, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(
            output_dir / f"resolution_correction_param_variation_{suffix}.png",
            dpi=160, bbox_inches="tight",
        )
        fig.savefig(
            output_dir / f"resolution_correction_param_variation_{suffix}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

    _make_grid(cosmo_block,
               "HF/LF ratio vs θ-quantile — cosmology + mean-flux", "cosmo")
    _make_grid(astro_block,
               "HF/LF ratio vs θ-quantile — IGM thermal / astro", "astro")


def write_holdout_validation(
    refits: dict,
    *,
    gp_lf,
    gp_hf,
    k_grid: np.ndarray,
    output_dir: Path,
    n_holdout: int = 50,
    z_eval: float | None = None,
    seed: int = 9999,
) -> None:
    """Per-param hold-out validation: relative error vs k on unseen Sobol θ.

    Generates a fresh Sobol sweep (different seed from training) over
    each param's prior, evaluates the GP at each θ for both LF and HF,
    compares to the per-param refit's prediction, and plots mean
    |pred − truth|/|truth| as a function of k for each fidelity.

    Same cosmo/astro split. Saves
    `holdout_validation_{cosmo,astro}.{png,pdf}`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import qmc
    from priya_forecast.parameters import (
        PARAM_NAMES, fiducial_vector, get_param,
    )

    cosmo_block = ("dtau0", "tau0", "ns", "Ap", "hub", "omegamh2")
    astro_block = ("herei", "heref", "alphaq", "hireionz", "bhfeedback")
    fid = np.array(fiducial_vector(), dtype=float)
    if z_eval is None:
        for r in refits.values():
            if r is not None and getattr(r, "is_multiz", False):
                z_eval = float((r.z_min + r.z_max) / 2.0)
                break
        if z_eval is None:
            z_eval = 3.6

    def _holdout_for_param(pname: str) -> dict[str, np.ndarray] | None:
        r = refits.get(pname)
        if r is None:
            return None
        p = get_param(pname)
        # Sobol over [prior_lo, prior_hi]; different seed from training (default 0).
        sampler = qmc.Sobol(d=1, seed=seed)
        u = sampler.random(n=n_holdout).ravel()
        theta_samples = p.prior[0] + u * (p.prior[1] - p.prior[0])
        idx = PARAM_NAMES.index(pname)

        rel_err_lf = np.empty((n_holdout, len(k_grid)), dtype=float)
        rel_err_hf = np.empty((n_holdout, len(k_grid)), dtype=float)
        for i, t in enumerate(theta_samples):
            theta = fid.copy()
            theta[idx] = t
            truth_lf = np.asarray(gp_lf.predict(theta, k_grid, z_eval), dtype=float)
            truth_hf = np.asarray(gp_hf.predict(theta, k_grid, z_eval), dtype=float)
            z_arg = z_eval if (getattr(r, "is_multiz", False) and z_eval is not None) else None
            pred_lf = r.predict(theta_phys=float(t), k=k_grid, resolution=LF_RESOLUTION, z=z_arg)
            pred_hf = r.predict(theta_phys=float(t), k=k_grid, resolution=HF_RESOLUTION, z=z_arg)
            rel_err_lf[i] = np.abs(pred_lf - truth_lf) / np.abs(truth_lf)
            rel_err_hf[i] = np.abs(pred_hf - truth_hf) / np.abs(truth_hf)
        return dict(
            mean_lf=rel_err_lf.mean(axis=0),
            mean_hf=rel_err_hf.mean(axis=0),
            max_lf=rel_err_lf.max(axis=0),
            max_hf=rel_err_hf.max(axis=0),
        )

    def _make_grid(block_names: tuple[str, ...], title: str, suffix: str) -> None:
        present_pairs = []
        for pname in block_names:
            info = _holdout_for_param(pname)
            if info is not None:
                present_pairs.append((pname, info))
        if not present_pairs:
            return
        cols = min(3, len(present_pairs))
        rows = (len(present_pairs) + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols, figsize=(3.2 * cols, 2.5 * rows),
            sharex=True, sharey=False, squeeze=False,
        )
        for ax in axes.flat:
            ax.set_visible(False)
        for i, (pname, info) in enumerate(present_pairs):
            ax = axes[i // cols][i % cols]
            ax.set_visible(True)
            ax.plot(k_grid, info["mean_lf"] * 100, color="#1f77b4", lw=1.5,
                    label=f"LF mean ({info['mean_lf'].mean()*100:.2f}%)")
            ax.plot(k_grid, info["mean_hf"] * 100, color="#d62728", lw=1.5,
                    label=f"HF mean ({info['mean_hf'].mean()*100:.2f}%)")
            ax.fill_between(k_grid, 0, info["max_lf"] * 100, color="#1f77b4", alpha=0.15)
            ax.fill_between(k_grid, 0, info["max_hf"] * 100, color="#d62728", alpha=0.15)
            ax.axhline(1.0, color="gray", lw=0.5, alpha=0.7, ls="--")  # 1% gate
            ax.set_title(pname, fontsize=9)
            ax.set_xscale("log")
            if i // cols == rows - 1:
                ax.set_xlabel("k [s/km]", fontsize=8)
            if i % cols == 0:
                ax.set_ylabel(r"$|\Delta P_F / P_F|$ [%]", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=7)
            ax.legend(fontsize=6, loc="best")
        suptitle = title + f" — hold-out n={n_holdout} (seed={seed})"
        if z_eval is not None:
            suptitle += f", z = {z_eval:.2f}"
        fig.suptitle(suptitle, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(output_dir / f"holdout_validation_{suffix}.png",
                    dpi=160, bbox_inches="tight")
        fig.savefig(output_dir / f"holdout_validation_{suffix}.pdf",
                    bbox_inches="tight")
        plt.close(fig)

    _make_grid(cosmo_block,
               "Hold-out validation — cosmology + mean-flux", "cosmo")
    _make_grid(astro_block,
               "Hold-out validation — IGM thermal / astro", "astro")


def write_resolution_correction_equations(
    refits: dict, output_dir: Path,
) -> None:
    """Symbolic HF/LF expressions per param — paper-grade text deliverable."""
    import sympy as sp

    output_dir = Path(output_dir)
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
        "per-(k, z) anchor / std from the LF emulator. Below we report",
        "the equation evaluated at $x_2=0.8$ (HF) and $x_2=0.4$ (LF), and",
        "the simplified `HF − LF` in flux_norm space.",
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
            ratio_norm = sp.simplify(expr_hf - expr_lf)
        except Exception as e:
            md.append(f"### {pname}: failed to sympify ({e})")
            md.append("")
            continue
        md.append(f"### `{pname}` (θ_fid = {r.fid_value:.4g})")
        md.append("")
        md.append("Trained equation (variables: θ, k, r=resolution, z):")
        md.append("```")
        md.append(_prettify_equation(r.equation_str))
        md.append("```")
        md.append("")
        md.append("At HF (r = 0.8):")
        md.append("```")
        md.append(_prettify_equation(str(expr_hf)))
        md.append("```")
        md.append("At LF (r = 0.4):")
        md.append("```")
        md.append(_prettify_equation(str(expr_lf)))
        md.append("```")
        md.append("Resolution correction in flux_norm space (HF − LF, simplified):")
        md.append("```")
        md.append(_prettify_equation(str(ratio_norm)))
        md.append("```")
        md.append("")
    (output_dir / "resolution_correction_equations.md").write_text(
        "\n".join(md) + "\n"
    )
