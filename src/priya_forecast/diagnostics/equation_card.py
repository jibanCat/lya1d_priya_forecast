"""Render a PySR equation set as a "card" figure.

The card shows:
- The set's name + redshift + combine rule
- Each per-parameter equation, LaTeX-rendered
- The Pareto-front pick (complexity, loss, rule)
- A small inset with the mathematical combine recipe

Used as a one-shot summary of "what PySR equations am I forecasting on?"
which makes diagnostic loops over different equation sets easy to
audit visually.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import sympy as sp


def _equation_to_latex(expr_str: str, variables: list[str]) -> str:
    """Convert a PySR equation string to a sympy LaTeX string with x0/x1/...
    aliased to the named variables."""
    from priya_forecast.models.pysr_model import _parse_safely

    allowed = {n: sp.Symbol(n) for n in variables}
    for i in range(len(variables)):
        allowed[f"x{i}"] = sp.Symbol(f"x{i}")
    try:
        expr = _parse_safely(expr_str, allowed)
        # Alias x0 → variables[0], etc.
        for i, name in enumerate(variables):
            expr = expr.subs(sp.Symbol(f"x{i}"), sp.Symbol(name))
        return sp.latex(expr)
    except Exception:
        return expr_str  # fallback to raw string


def plot_equation_card(
    *,
    name: str,
    redshift: float,
    combine: str,
    parameters: dict[str, dict[str, Any]],
    outpath: str | Path,
) -> Path:
    """Render an equation-set summary card.

    Parameters
    ----------
    name : str
        Equation-set label (matches the YAML's `name:`).
    redshift : float
    combine : str
        "multiplicative" | "additive" | "joint".
    parameters : dict
        Mapping `param_name -> {"raw_expression": str, "variables": list[str],
        "complexity": int | None, "loss": float | None, "fiducial": float}`.
    outpath : Path
        Output PNG.
    """
    import matplotlib.pyplot as plt

    n = len(parameters)
    fig_h = 1.0 + 0.55 * n
    fig, ax = plt.subplots(figsize=(9, fig_h), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    # Header
    header = (
        rf"$\bf{{Equation\ set:}}$ {name}    "
        rf"$\bf{{z}}$ = {redshift}    "
        rf"$\bf{{combine}}$ = {combine}"
    )
    ax.text(0.01, 0.98, header, transform=ax.transAxes, fontsize=11, va="top")

    # Combine recipe
    recipes = {
        "multiplicative": r"$P(\theta,k) = P_{\rm fid}(k)\,\prod_i\,\frac{f_i(\theta_i,k)}{f_i(\theta_{i,{\rm fid}},k)}$",
        "additive":       r"$P(\theta,k) = P_{\rm fid}(k)\,+\,\sum_i\,\bigl[f_i(\theta_i,k) - f_i(\theta_{i,{\rm fid}},k)\bigr]$",
        "joint":          r"$P(\theta,k) = f(\theta_1,\dots,\theta_n,k)$  (single joint equation)",
    }
    ax.text(0.01, 0.93, recipes.get(combine, ""), transform=ax.transAxes, fontsize=10, va="top")

    # Per-parameter rows
    y = 0.88
    dy = 0.85 / max(n, 1)
    for pname, meta in parameters.items():
        raw = meta.get("raw_expression", "?")
        vars_ = meta.get("variables", [pname, "k"])
        complexity = meta.get("complexity")
        loss = meta.get("loss")
        fid = meta.get("fiducial")
        latex = _equation_to_latex(raw, vars_)

        meta_txt = []
        if complexity is not None:
            meta_txt.append(f"complexity={complexity}")
        if loss is not None:
            meta_txt.append(f"loss={loss:.3g}")
        if fid is not None:
            meta_txt.append(rf"$\theta_{{\rm fid}}$={fid:g}")
        meta_label = "  •  ".join(meta_txt)

        ax.text(0.01, y, rf"$\bf{{{pname}}}$:", transform=ax.transAxes, fontsize=10, va="top")
        ax.text(0.13, y, rf"${latex}$", transform=ax.transAxes, fontsize=10, va="top")
        if meta_label:
            ax.text(0.01, y - 0.4 * dy, meta_label, transform=ax.transAxes,
                    fontsize=8, color="grey", va="top")
        y -= dy

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return outpath
