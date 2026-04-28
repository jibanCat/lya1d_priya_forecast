"""Forecast diagnostic figures.

Five figure helpers, each writing one file. Conventions:

- Every function takes an `outpath: Path` and returns it on success.
- Default style: white background, modest tight_layout, dpi=120.
- All figures include a small `params/data hash` footer so the student can
  identify which run produced which image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from priya_forecast.fisher import FisherResult
from priya_forecast.models.base import P1DModel
from priya_forecast.parameters import PARAMS_11D, fiducial_vector, get_param


def _setup_axes_white():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    return fig, ax


# ---------------------------------------------------------------------------
# 1. PySR prediction vs GP at fiducial + a few off-fiducial points
# ---------------------------------------------------------------------------


def plot_pysr_vs_gp(
    *,
    pysr_model: P1DModel,
    gp_model: P1DModel,
    k: np.ndarray,
    z: float,
    theta_fid: np.ndarray,
    perturbations: dict[str, list[float]] | None = None,
    outpath: str | Path,
) -> Path:
    """Overlay PySR and GP P_F at fiducial and at one-parameter perturbations.

    `perturbations` maps parameter names to lists of physical values to plot.
    By default we sweep ns, Ap, hub at +/- 1 prior sigma each.
    """
    import matplotlib.pyplot as plt

    if perturbations is None:
        perturbations = {}
        for pname in ("ns", "Ap", "hub"):
            p = get_param(pname)
            mid = p.fid
            half = 0.25 * p.width()
            perturbations[pname] = [mid - half, mid + half]

    fig, ax = _setup_axes_white()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$k$ [s/km]")
    ax.set_ylabel(r"$P_F(k)$")
    ax.set_title(f"PySR vs GP at z={z}")

    # Fiducial
    ax.plot(k, gp_model.predict(theta_fid, k, z), "k-", lw=2, label="GP fiducial")
    ax.plot(k, pysr_model.predict(theta_fid, k, z), "r--", lw=1.5, label="PySR fiducial")

    # Perturbations
    color_idx = 0
    palette = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    for pname, values in perturbations.items():
        from priya_forecast.parameters import PARAM_NAMES

        idx = PARAM_NAMES.index(pname)
        for v in values:
            theta = theta_fid.copy()
            theta[idx] = v
            color = palette[color_idx % len(palette)]
            color_idx += 1
            ax.plot(k, gp_model.predict(theta, k, z), color=color, lw=1.0, alpha=0.7,
                    label=f"GP {pname}={v:.4g}")
            ax.plot(k, pysr_model.predict(theta, k, z), color=color, lw=1.0,
                    alpha=0.7, ls="--", label=f"PySR {pname}={v:.4g}")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(alpha=0.3)

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


# ---------------------------------------------------------------------------
# 2. Per-parameter sensitivity dP/dtheta_i / P at fiducial
# ---------------------------------------------------------------------------


def plot_per_parameter_sensitivity(
    *,
    model: P1DModel,
    k: np.ndarray,
    z: float,
    theta_fid: np.ndarray,
    outpath: str | Path,
    params=PARAMS_11D,
) -> Path:
    """For each of the 11 params, plot d(log P)/dθ_i normalized by prior width.

    This is the "what does the data care about, parameter by parameter"
    summary that motivates which equations matter and where they'll matter.
    """
    import matplotlib.pyplot as plt

    fig, ax = _setup_axes_white()
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ [s/km]")
    ax.set_ylabel(r"$d\,\ln P_F / d\,\hat\theta$  (per prior-width)")
    ax.set_title(f"Per-parameter sensitivity at fiducial, z={z}")

    p_fid = model.predict(theta_fid, k, z)
    palette = plt.get_cmap("tab20").colors
    for i, p in enumerate(params):
        h = 0.005 * p.width()
        t_plus = theta_fid.copy()
        t_plus[i] += h
        t_minus = theta_fid.copy()
        t_minus[i] -= h
        dp = (model.predict(t_plus, k, z) - model.predict(t_minus, k, z)) / (2 * h)
        # Normalize by prior width for an apples-to-apples comparison.
        sens = dp * p.width() / p_fid
        ax.plot(k, sens, lw=1.2, color=palette[i % len(palette)], label=p.name)
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=7, ncol=3, loc="best")
    ax.grid(alpha=0.3)
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


# ---------------------------------------------------------------------------
# 3. Residual at fiducial: (P_PySR - P_GP) / sqrt(diag(C_eBOSS))
# ---------------------------------------------------------------------------


def plot_residuals_at_fiducial(
    *,
    pysr_model: P1DModel,
    gp_model: P1DModel,
    k: np.ndarray,
    z: float,
    theta_fid: np.ndarray,
    cov_diag: np.ndarray,
    outpath: str | Path,
    perturbation_amplitude: float = 0.3,
    n_perturbations: int = 5,
    seed: int = 0,
    perturbed_param_names: list[str] | None = None,
) -> Path:
    """Residual at fid AND at perturbations along the forecast subspace.

    For multiplicative combine, the residual at fid is identically zero
    (all ratios = 1), so we draw `n_perturbations` random off-fid points
    in (perturbation_amplitude * prior_width / 2) along `perturbed_param_names`
    only — keeping non-forecast params at fid.
    """
    import matplotlib.pyplot as plt
    from priya_forecast.parameters import PARAMS_11D, PARAM_NAMES

    fig, ax = _setup_axes_white()
    sigma = np.sqrt(cov_diag)
    rng = np.random.default_rng(seed)
    if perturbed_param_names is None:
        # Default: perturb the four "well-constrained" params.
        perturbed_param_names = ["ns", "Ap", "hub", "omegamh2"]
    perturbed_idx = [PARAM_NAMES.index(n) for n in perturbed_param_names]
    widths = np.array([PARAMS_11D[i].width() for i in perturbed_idx])

    diff_fid = pysr_model.predict(theta_fid, k, z) - gp_model.predict(theta_fid, k, z)
    ax.plot(k, diff_fid / sigma, "k-", lw=2, label="θ = fid")

    palette = plt.get_cmap("tab10").colors
    for i in range(n_perturbations):
        theta = theta_fid.copy()
        delta = rng.uniform(-perturbation_amplitude, perturbation_amplitude, size=len(perturbed_idx)) * widths
        for di, idx in zip(delta, perturbed_idx):
            theta[idx] = theta_fid[idx] + di
        try:
            diff = pysr_model.predict(theta, k, z) - gp_model.predict(theta, k, z)
            ax.plot(k, diff / sigma, color=palette[i % len(palette)], alpha=0.7, lw=1,
                    label=f"perturbation {i+1}")
        except Exception:
            continue
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(1, color="grey", lw=0.5, ls="--")
    ax.axhline(-1, color="grey", lw=0.5, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ [s/km]")
    ax.set_ylabel(r"$(P_{\mathrm{PySR}} - P_{\mathrm{GP}}) / \sigma_{\mathrm{eBOSS}}$")
    ax.set_title(f"Residual in eBOSS-sigma units (fid + perturbations), z={z}")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(alpha=0.3)
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


# ---------------------------------------------------------------------------
# 4. Fisher 1-sigma bar chart (PySR vs GP overlay)
# ---------------------------------------------------------------------------


def plot_fisher_sigma_table(
    *,
    fisher_results: dict[str, FisherResult],
    outpath: str | Path,
) -> Path:
    """Bar chart: 1σ marginalized error per parameter, one bar group per Fisher.

    `fisher_results` maps a label → FisherResult. The labels appear in the
    legend (e.g. {"GP": fr_gp, "PySR-v1": fr_pysr}).
    """
    import matplotlib.pyplot as plt

    if not fisher_results:
        raise ValueError("fisher_results is empty.")
    sample = next(iter(fisher_results.values()))
    names = sample.param_names
    n = len(names)

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bar_w = 0.8 / len(fisher_results)
    x = np.arange(n)
    for i, (label, fr) in enumerate(fisher_results.items()):
        ax.bar(x + i * bar_w, fr.sigma, width=bar_w, label=label)
    ax.set_xticks(x + 0.4 - bar_w / 2)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("1σ marginalized")
    ax.set_yscale("log")
    ax.set_title("Fisher forecast: 1σ per parameter")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


# ---------------------------------------------------------------------------
# 5. Fisher Gaussian-corner overlay (1D + 2D contour)
# ---------------------------------------------------------------------------


def plot_fisher_corner(
    *,
    fisher_results: dict[str, FisherResult],
    outpath: str | Path,
    param_subset: list[str] | None = None,
    axis_reference: str | None = None,
) -> Path:
    """1D Gaussian posterior overlays + 2D 1σ confidence ellipses.

    Lightweight Fisher-Gaussian corner — no MCMC required. Use `param_subset`
    to focus on a few params; full 11D corners are too dense to read.
    `axis_reference` names which result determines axis limits (default: the
    *tightest* σ across all results, so GP-like reference contours are
    visible even when other sets are far looser).
    """
    import matplotlib.pyplot as plt

    if not fisher_results:
        raise ValueError("fisher_results is empty.")
    sample = next(iter(fisher_results.values()))
    all_names = list(sample.param_names)
    subset = param_subset if param_subset is not None else all_names[:5]
    idx = [all_names.index(n) for n in subset]
    n = len(idx)

    fig, axes = plt.subplots(n, n, figsize=(2 * n, 2 * n), dpi=120)
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = np.array([[axes]])

    # Pick the reference whose σ sets axis bounds — default = tightest.
    if axis_reference is None:
        sigmas_total = {label: float(np.sum(fr.sigma)) for label, fr in fisher_results.items()
                        if np.all(np.isfinite(fr.sigma))}
        axis_reference = min(sigmas_total, key=sigmas_total.get) if sigmas_total else next(iter(fisher_results))
    ref_fr = fisher_results[axis_reference]

    palette = plt.get_cmap("tab10").colors
    for ai, i in enumerate(idx):
        for aj, j in enumerate(idx):
            ax = axes[ai, aj]
            ax.set_facecolor("white")
            if aj > ai:
                ax.set_visible(False)
                continue
            for c, (label, fr) in enumerate(fisher_results.items()):
                color = palette[c % len(palette)]
                if ai == aj:
                    s = fr.sigma[i]
                    fid = fr.theta_fid[i]
                    xs = np.linspace(fid - 4 * s, fid + 4 * s, 200)
                    ys = np.exp(-0.5 * ((xs - fid) / s) ** 2) / (s * np.sqrt(2 * np.pi))
                    ax.plot(xs, ys, color=color, label=label)
                else:
                    sub = np.array([[fr.cov[i, i], fr.cov[i, j]], [fr.cov[j, i], fr.cov[j, j]]])
                    fid_i, fid_j = fr.theta_fid[i], fr.theta_fid[j]
                    _draw_ellipse(ax, mean=(fid_j, fid_i), cov=sub[::-1, ::-1], color=color,
                                  set_limits=False)
                    ax.scatter(fid_j, fid_i, marker="x", color=color, s=20)
            # Set axis limits from the reference's σ (so GP-tight contours are visible).
            if ai == aj:
                s_ref = ref_fr.sigma[i]; mu = ref_fr.theta_fid[i]
                ax.set_xlim(mu - 4 * s_ref, mu + 4 * s_ref)
            else:
                s_i, s_j = ref_fr.sigma[i], ref_fr.sigma[j]
                ax.set_xlim(ref_fr.theta_fid[j] - 4 * s_j, ref_fr.theta_fid[j] + 4 * s_j)
                ax.set_ylim(ref_fr.theta_fid[i] - 4 * s_i, ref_fr.theta_fid[i] + 4 * s_i)
            if ai == n - 1:
                ax.set_xlabel(all_names[j])
            if aj == 0 and ai != aj:
                ax.set_ylabel(all_names[i])
            ax.tick_params(labelsize=7)

    # Single legend on top-right
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=9)
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


def _draw_ellipse(
    ax, *, mean: tuple[float, float], cov: np.ndarray, color: str, set_limits: bool = True
) -> None:
    """1σ confidence ellipse from a 2x2 cov."""
    import matplotlib.patches as mpatches

    eigvals, eigvecs = np.linalg.eigh(cov)
    if np.any(eigvals <= 0):
        return
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * np.sqrt(eigvals)
    ell = mpatches.Ellipse(
        xy=mean, width=width, height=height, angle=angle,
        edgecolor=color, facecolor="none", lw=1.2,
    )
    ax.add_patch(ell)
    if set_limits:
        pad = 1.3 * np.sqrt(max(eigvals))
        ax.set_xlim(mean[0] - pad, mean[0] + pad)
        ax.set_ylim(mean[1] - pad, mean[1] + pad)
