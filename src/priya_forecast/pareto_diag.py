"""Render the per-parameter Pareto-faithfulness diagnostic figure.

Pure-plotting: reads PySR Pareto CSVs + grad-faith sidecars, no emulator.
A front whose sidecar is missing is drawn gray (value-only) so the layout
can be iterated before the cluster gradient eval lands.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from priya_forecast.grad_faith_io import read_grad_faith_sidecar

GATE_TOL = 0.25


def load_front(pareto_csv, sidecar_csv=None) -> pd.DataFrame:
    """Return a DataFrame[Complexity, Loss, grad_err, gate_pass, value_mse].

    `Loss` is the PySR training loss (NOT comparable across runs trained with
    different objectives); `value_mse` is the emulator-grounded common value
    loss from the sidecar (comparable). grad_err/gate_pass/value_mse are NaN/NA
    when no sidecar is supplied or a complexity has no sidecar row (left join).
    """
    pareto = pd.read_csv(pareto_csv)[["Complexity", "Loss"]].copy()
    if sidecar_csv is not None and Path(sidecar_csv).exists():
        side = read_grad_faith_sidecar(sidecar_csv)
        cols = ["Complexity", "grad_err", "gate_pass"]
        if "value_mse" in side.columns:
            cols.append("value_mse")
        out = pareto.merge(side[cols], on="Complexity", how="left")
        if "value_mse" not in out.columns:
            out["value_mse"] = np.nan
        return out
    return pareto.assign(grad_err=np.nan, gate_pass=pd.NA, value_mse=np.nan)


def render_grid(fronts_by_param, out_path, *, gate_tol=GATE_TOL,
                param_order=None, ncol=4, y_col="Loss", y_label=None,
                annotate=None):
    """Render one panel per parameter; colour = grad_err (slope error vs the GP).

    fronts_by_param: {param: [ {front: DataFrame, label: str, marker: str}, ... ]}
    y_col: which front column on the (log) y-axis. "value_mse" is the common,
    cross-objective-comparable value loss; "Loss" is the raw PySR training loss.
    Gate legibility: markers that clear the gate (grad_err <= gate_tol) get a
    **bold black ring**, failing markers a thin grey edge, and not-Fisher-safe
    candidates a grey fill -- so pass/fail is readable on each panel, not only on
    the colorbar. `annotate`: optional {param: dict(text=, xy=, xytext=)}.
    """
    params = list(param_order) if param_order else list(fronts_by_param)
    annotate = annotate or {}
    nrow = int(np.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             squeeze=False, layout="constrained")
    cmap = plt.get_cmap("RdYlGn_r")
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=gate_tol, vmax=1.0)
    last_sc = None

    for i, p in enumerate(params):
        ax = axes[i // ncol][i % ncol]
        for series in fronts_by_param.get(p, []):
            df = series["front"]
            marker = series.get("marker", "o")
            label = series.get("label")
            ge = df["grad_err"].to_numpy(dtype=float)
            cx = df["Complexity"].to_numpy(dtype=float)
            cy = df[y_col].to_numpy(dtype=float)
            gv = np.clip(ge, 0.0, 1.0)
            seen = ~np.isnan(ge)
            used_label = False
            fail = seen & (ge > gate_tol)
            if fail.any():
                last_sc = ax.scatter(
                    cx[fail], cy[fail], c=gv[fail], cmap=cmap, norm=norm,
                    marker=marker, edgecolor="0.45", linewidth=0.4, s=44,
                    zorder=3, label=label)
                used_label = True
            passg = seen & (ge <= gate_tol)
            if passg.any():
                sc2 = ax.scatter(
                    cx[passg], cy[passg], c=gv[passg], cmap=cmap, norm=norm,
                    marker=marker, edgecolor="k", linewidth=1.6, s=80,
                    zorder=4, label=None if used_label else label)
                last_sc = last_sc if last_sc is not None else sc2
                used_label = True
            if (~seen).any():
                ax.scatter(cx[~seen], cy[~seen], color="0.78", marker=marker,
                           s=44, zorder=2, label=None if used_label else label)
        ax.set_yscale("log")
        ax.set_title(p)
        ax.set_xlabel("complexity")
        ax.set_ylabel(y_label or y_col)
        ax.grid(True, which="both", alpha=0.2)
        if any(s.get("label") for s in fronts_by_param.get(p, [])):
            ax.legend(fontsize=7, loc="best")
        if p in annotate:
            a = annotate[p]
            ax.annotate(a["text"], xy=a["xy"], xytext=a["xytext"], fontsize=8,
                        color="#6e0b0b", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#6e0b0b", lw=1.4))

    for j in range(len(params), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes.ravel().tolist(),
                            fraction=0.025, pad=0.01)
        cbar.set_label("grad_err = slope error vs GP  (median |∂P_eq/∂P_GP − 1|)\n"
                       "bold ring = clears the 0.25 gate (faithful)")
        cbar.ax.axhline(gate_tol, color="k", lw=1.2)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
    return out_path
