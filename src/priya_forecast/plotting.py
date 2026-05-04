"""Forecast plotting helpers — Fisher corner plot.

`plot_fisher_corner` overlays two Gaussian Fisher posteriors (GP "truth" vs
PySR hybrid) using matplotlib. Each posterior is centered at `theta_fid`
and has covariance `fr.cov`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; tests / cluster runs.
import matplotlib.pyplot as plt
import numpy as np

from priya_forecast.parameters import Param


def plot_fisher_corner(
    *,
    fr_gp,
    fr_hybrid,
    params: tuple[Param, ...],
    output_path: str | Path,
    width_sigma: float = 4.0,
    title: str | None = None,
) -> Path:
    """Overlaid Gaussian Fisher corner plot — GP (truth) vs hybrid (PySR).

    Parameters
    ----------
    fr_gp, fr_hybrid : FisherResult
        Output of `priya_forecast.fisher.fisher_matrix`. Must share the
        same `param_names` and `theta_fid`.
    params : tuple of Param
        Parameter metadata (used for axis labels, prior bounds).
    output_path : path
        Where to write the PDF/PNG. Extension determines format.
    width_sigma : float
        Axis range for each panel: `theta_fid_i ± width_sigma · σ_GP_i`.
    title : str | None
        Optional super-title.
    """
    output_path = Path(output_path)
    n = len(params)
    if fr_gp.cov.shape != (n, n) or fr_hybrid.cov.shape != (n, n):
        raise ValueError(
            f"Cov shape mismatch: GP {fr_gp.cov.shape}, hybrid "
            f"{fr_hybrid.cov.shape}, expected ({n},{n})."
        )
    theta = np.asarray(fr_gp.theta_fid, dtype=float)
    if theta.shape != (n,):
        raise ValueError(f"theta_fid shape {theta.shape} != ({n},).")

    sigma_gp = np.sqrt(np.diag(fr_gp.cov))
    sigma_hy = np.sqrt(np.diag(fr_hybrid.cov))
    panel_sigma = np.maximum(sigma_gp, sigma_hy)
    lows = np.array([
        max(p.prior[0], theta[i] - width_sigma * panel_sigma[i])
        for i, p in enumerate(params)
    ])
    highs = np.array([
        min(p.prior[1], theta[i] + width_sigma * panel_sigma[i])
        for i, p in enumerate(params)
    ])

    fig, axes = plt.subplots(n, n, figsize=(1.4 * n, 1.4 * n), squeeze=False)
    color_gp = "#000000"
    color_hy = "#d62728"

    def gauss_1d(x, mean, sigma):
        return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

    def ellipse_2d(ax, mean_xy, cov2, color, n_sigma=(1, 2)):
        from matplotlib.patches import Ellipse
        vals, vecs = np.linalg.eigh(cov2)
        order = np.argsort(vals)[::-1]
        vals = np.maximum(vals[order], 0.0)
        vecs = vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        for ns in n_sigma:
            width = 2 * ns * np.sqrt(vals[0])
            height = 2 * ns * np.sqrt(vals[1])
            e = Ellipse(
                xy=mean_xy, width=width, height=height, angle=angle,
                edgecolor=color, facecolor="none", lw=1.0,
                alpha=0.9 if ns == 1 else 0.5,
            )
            ax.add_patch(e)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            if i == j:
                xx = np.linspace(lows[i], highs[i], 200)
                ax.plot(xx, gauss_1d(xx, theta[i], sigma_gp[i]),
                        color=color_gp, lw=1.5, label="GP")
                ax.plot(xx, gauss_1d(xx, theta[i], sigma_hy[i]),
                        color=color_hy, lw=1.5, ls="--", label="hybrid")
                ax.axvline(theta[i], color="gray", lw=0.5, alpha=0.7)
                ax.set_xlim(lows[i], highs[i])
                ax.set_yticks([])
                if i == 0:
                    ax.legend(fontsize=7, loc="upper right")
            else:
                cov_gp_2 = np.array([
                    [fr_gp.cov[j, j], fr_gp.cov[j, i]],
                    [fr_gp.cov[i, j], fr_gp.cov[i, i]],
                ])
                cov_hy_2 = np.array([
                    [fr_hybrid.cov[j, j], fr_hybrid.cov[j, i]],
                    [fr_hybrid.cov[i, j], fr_hybrid.cov[i, i]],
                ])
                ellipse_2d(ax, (theta[j], theta[i]), cov_gp_2, color_gp)
                ellipse_2d(ax, (theta[j], theta[i]), cov_hy_2, color_hy)
                ax.axhline(theta[i], color="gray", lw=0.3, alpha=0.5)
                ax.axvline(theta[j], color="gray", lw=0.3, alpha=0.5)
                ax.set_xlim(lows[j], highs[j])
                ax.set_ylim(lows[i], highs[i])
            if i == n - 1:
                ax.set_xlabel(f"${params[j].latex}$", fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(f"${params[i].latex}$", fontsize=8)
            elif j > 0:
                ax.set_yticklabels([])
            ax.tick_params(axis="both", which="major", labelsize=6)

    if title is not None:
        fig.suptitle(title, fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
