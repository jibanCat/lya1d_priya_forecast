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
    """Return a DataFrame[Complexity, Loss, grad_err, gate_pass].

    grad_err/gate_pass are NaN/NA when no sidecar is supplied or a given
    complexity has no sidecar row (left join).
    """
    pareto = pd.read_csv(pareto_csv)[["Complexity", "Loss"]].copy()
    if sidecar_csv is not None and Path(sidecar_csv).exists():
        side = read_grad_faith_sidecar(sidecar_csv)[
            ["Complexity", "grad_err", "gate_pass"]]
        return pareto.merge(side, on="Complexity", how="left")
    return pareto.assign(grad_err=np.nan, gate_pass=pd.NA)


def render_grid(fronts_by_param, out_path, *, gate_tol=GATE_TOL,
                param_order=None, ncol=4):
    """Render one panel per parameter; color = grad_err (clipped to [0,1]).

    fronts_by_param: {param: [ {front: DataFrame, label: str, marker: str}, ... ]}
    """
    params = list(param_order) if param_order else list(fronts_by_param)
    nrow = int(np.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             squeeze=False)
    cmap = plt.get_cmap("RdYlGn_r")
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=gate_tol, vmax=1.0)
    last_sc = None

    for i, p in enumerate(params):
        ax = axes[i // ncol][i % ncol]
        for series in fronts_by_param.get(p, []):
            df = series["front"]
            marker = series.get("marker", "o")
            ge = df["grad_err"].to_numpy(dtype=float)
            seen = ~np.isnan(ge)
            if seen.any():
                last_sc = ax.scatter(
                    df["Complexity"][seen], df["Loss"][seen],
                    c=np.clip(ge[seen], 0.0, 1.0), cmap=cmap, norm=norm,
                    marker=marker, edgecolor="k", linewidth=0.4, s=44,
                    zorder=3, label=series.get("label"))
            if (~seen).any():
                ax.scatter(df["Complexity"][~seen], df["Loss"][~seen],
                           color="0.75", marker=marker, s=44, zorder=2,
                           label=series.get("label") if not seen.any() else None)
        ax.set_yscale("log")
        ax.set_title(p)
        ax.set_xlabel("complexity")
        ax.set_ylabel("loss")
        ax.grid(True, which="both", alpha=0.2)
        if any(s.get("label") for s in fronts_by_param.get(p, [])):
            ax.legend(fontsize=7, loc="best")

    for j in range(len(params), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes.ravel().tolist(),
                            fraction=0.025, pad=0.01)
        cbar.set_label("grad_err  (median |d_eq / d_GP - 1|, clipped at 1)")
        cbar.ax.axhline(gate_tol, color="k", lw=1.2)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
