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

    # Grid figure (HF/LF ratio per param).
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
    suptitle = "Per-dim resolution correction at θ=fid (HF/LF ratio)"
    if z_eval is not None:
        suptitle += f"\nevaluated at z = {z_eval:.2f}"
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "resolution_correction_grid.png", dpi=160,
                bbox_inches="tight")
    fig.savefig(output_dir / "resolution_correction_grid.pdf",
                bbox_inches="tight")
    plt.close(fig)


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
