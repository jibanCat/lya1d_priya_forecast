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
    aliased to the named variables. Strips \\left / \\right because
    matplotlib's mathtext can't render them."""
    from priya_forecast.models.pysr_model import _parse_safely

    allowed = {n: sp.Symbol(n) for n in variables}
    for i in range(len(variables)):
        allowed[f"x{i}"] = sp.Symbol(f"x{i}")
    try:
        expr = _parse_safely(expr_str, allowed)
        for i, name in enumerate(variables):
            expr = expr.subs(sp.Symbol(f"x{i}"), sp.Symbol(name))
        latex = sp.latex(expr)
    except Exception:
        return expr_str  # fallback to raw string
    # mathtext doesn't support \left/\right — drop them.
    latex = latex.replace(r"\left(", "(").replace(r"\right)", ")")
    latex = latex.replace(r"\left[", "[").replace(r"\right]", "]")
    latex = latex.replace(r"\left|", "|").replace(r"\right|", "|")
    return latex


def plot_equation_card(
    *,
    name: str,
    redshift: float,
    combine: str,
    parameters: dict[str, dict[str, Any]],
    outpath: str | Path,
) -> Path:
    """Render an equation-set summary card.

    Per-parameter equations render as a table of one row per param.
    Joint sets — denoted by the special `__joint__` key in `parameters` —
    render as a single equation block with line-wrapping.
    """
    import matplotlib.pyplot as plt
    import textwrap

    is_joint = "__joint__" in parameters
    n = 1 if is_joint else len(parameters)
    fig_h = 1.6 + (0.6 if is_joint else 0.55) * max(n, 1)
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    # Header (split across two lines so it never overlaps the recipe).
    ax.text(0.01, 0.97, rf"$\bf{{Equation\ set:}}$ {name}",
            transform=ax.transAxes, fontsize=11, va="top")
    ax.text(0.55, 0.97, rf"$\bf{{z}}$ = {redshift}    $\bf{{combine}}$ = {combine}",
            transform=ax.transAxes, fontsize=11, va="top")

    recipes = {
        "multiplicative": r"$P(\theta,k) = P_{\rm fid}(k)\,\prod_i\,\frac{f_i(\theta_i,k)}{f_i(\theta_{i,{\rm fid}},k)}$",
        "additive":       r"$P(\theta,k) = P_{\rm fid}(k)\,+\,\sum_i\,\bigl[f_i(\theta_i,k) - f_i(\theta_{i,{\rm fid}},k)\bigr]$",
        "joint":          r"$P(\theta,k) = f(\theta_1,\dots,\theta_n,k)$  (single joint equation)",
    }
    ax.text(0.01, 0.89, recipes.get(combine, ""), transform=ax.transAxes,
            fontsize=10, va="top")

    if is_joint:
        meta = parameters["__joint__"]
        raw = meta.get("raw_expression", "?")
        complexity = meta.get("complexity")
        loss = meta.get("loss")
        # Render long joint equations as monospace code, soft-wrapped at
        # ` + ` / ` - ` boundaries. mathtext can't span newlines so any
        # \frac{...} that wraps would break — code rendering sidesteps it.
        wrap_at = 95
        chunks, buf = [], ""
        for tok in raw.replace(" - ", "  - ").split(" + "):
            if len(buf) + len(tok) > wrap_at:
                chunks.append(buf.rstrip())
                buf = tok
            else:
                buf = (buf + " + " + tok) if buf else tok
        if buf:
            chunks.append(buf.rstrip())
        rendered = "\n  ".join(chunks)
        ax.text(
            0.01, 0.80, "f(theta, k) =", transform=ax.transAxes,
            fontsize=10, va="top",
        )
        ax.text(
            0.04, 0.74, rendered, transform=ax.transAxes,
            fontsize=8.5, family="monospace", va="top",
        )
        if complexity is not None or loss is not None:
            tag = []
            if complexity is not None:
                tag.append(f"complexity={complexity}")
            if loss is not None:
                tag.append(f"loss={loss:.3g}")
            ax.text(
                0.01, 0.10, "  •  ".join(tag),
                transform=ax.transAxes, fontsize=9, color="grey", va="bottom",
            )
    else:
        # Per-parameter rows.
        y = 0.83
        dy = 0.78 / max(n, 1)
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
