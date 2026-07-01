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

    NOTE — the sidecar only scores **Fisher-safe** rows (those whose equation
    depends on the parameter), so the absolute lowest-`Loss` row can have NaN
    `grad_err` (e.g. ns). For the *value-optimal faithful* equation, drop the
    unscored rows first::

        front.dropna(subset=["grad_err"]).sort_values("Loss").iloc[0]
    """
    pareto = pd.read_csv(pareto_csv, comment="#")[["Complexity", "Loss"]].copy()
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
                param_order=None, ncol=4, y_col="value_mse", y_label=None,
                annotate=None):
    """Render one panel per parameter; colour = grad_err (slope error vs the GP).

    fronts_by_param: {param: [ {front: DataFrame, label: str, marker: str}, ... ]}
    y_col: which front column on the (log) y-axis. "value_mse" is the common,
    cross-objective-comparable value loss; "Loss" is the raw PySR training loss.
    Gate legibility: hard two-tone fill -- green at/below the gate (faithful),
    red above it (Mirage) -- plus a **bold black ring** on markers that clear the
    gate (thin grey edge on failing markers, grey fill on not-Fisher-safe ones),
    so colour and ring agree and pass/fail is readable on each panel, not only on
    the colorbar. `annotate`: optional {param: dict(text=, xy=, xytext=)}.
    """
    params = list(param_order) if param_order else list(fronts_by_param)
    annotate = annotate or {}
    nrow = int(np.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             squeeze=False, layout="constrained")
    # Hard two-tone so colour and the pass/fail ring always agree: green at or
    # below the gate (faithful), red above it (the Mirage). The old diverging map
    # centred on the gate rendered near-gate points an ambiguous cream on *both*
    # sides, so a barely-failing point looked as faithful as a passing one.
    cmap = mcolors.ListedColormap(["#1a9850", "#d6604d"])
    norm = mcolors.BoundaryNorm([0.0, gate_tol + 1e-12, 1.0], cmap.N)
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
        # One shared y-label (fig.supylabel below) instead of repeating the long
        # label on every panel, where it collided with the next column's ticks.
        ax.tick_params(axis="both", labelsize=15)
        ax.tick_params(axis="y", which="minor", labelsize=12)
        ax.grid(True, which="both", alpha=0.2)
        if any(s.get("label") for s in fronts_by_param.get(p, [])):
            ax.legend(fontsize=14, loc="best")
        if p in annotate:
            a = annotate[p]
            ax.annotate(a["text"], xy=a["xy"], xytext=a["xytext"], fontsize=15,
                        color="#6e0b0b", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#6e0b0b", lw=1.4))

    for j in range(len(params), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.supylabel(y_label or y_col, fontsize=22)

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes.ravel().tolist(),
                            fraction=0.025, pad=0.01, ticks=[gate_tol])
        cbar.set_label(r"derivative faithfulness vs GP ($\mathrm{grad\_err}$)"
                       "\n"
                       r"green = faithful ($\leq 0.25$)   red = Mirage ($> 0.25$)"
                       "\n"
                       r"bold ring = clears the gate", fontsize=16)
        cbar.ax.tick_params(labelsize=15)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
