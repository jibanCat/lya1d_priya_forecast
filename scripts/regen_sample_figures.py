"""Regenerate the sample figures committed under `docs/figures/`.

Uses the *real* PRIYA GP emulator at fiducial as ground truth and the
eBOSS DR14 covariance for the noise model (the standard cosmology-forecast
setup). We progressively scale up dimensionality so each step is
diagnosable: first 1D (vary one parameter), then 2D, then 3D, then
"the four well-constrained" subset (ns, Ap, hub, omegamh2).

Outputs (under `docs/figures/`):

  fig01_gp_at_fiducial.png        GP prediction at fid + eBOSS data overlay
  fig02_gp_param_sensitivity.png  d(log P)/d theta_i for the four sensitive params
  fig03_fisher_1d.png             1D forecast: vary one parameter at a time
  fig04_fisher_2d.png             2D forecast on (ns, Ap)
  fig05_fisher_3d.png             3D forecast on (ns, Ap, hub)
  fig06_fisher_4d.png             4D forecast on (ns, Ap, hub, omegamh2)
  fig07_fisher_corner.png         4D Gaussian-corner overlay
  fig08_pysr_vs_gp.png            (optional) sample PySR equation vs GP

Run from repo root with the upstream lyaemu repo on PYTHONPATH:

    PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \\
        python scripts/regen_sample_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running from repo root with the upstream lyaemu added.
_LYAEMU = Path("/home/mfho/student_projects/lya_emulator_full")
if _LYAEMU.exists():
    sys.path.insert(0, str(_LYAEMU))

from priya_forecast.data import load_eboss
from priya_forecast.fisher import fisher_matrix
from priya_forecast.likelihood import GaussianLikelihood
from priya_forecast.models.base import P1DModel
from priya_forecast.parameters import (
    PARAM_NAMES,
    PARAMS_11D,
    Param,
    fiducial_vector,
    get_param,
)

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def _setup():
    """Build the real GP, prefetch eBOSS data."""
    from priya_forecast.models.gp_model import GPModel

    print("Building real GPModel (loads emulator pickles for all 13 z-bins)...")
    gp = GPModel()
    z = 3.6
    k_eboss, pf_eboss, cov = load_eboss(z=z)
    fid = np.array(fiducial_vector(), dtype=float)
    p_fid = gp.predict(fid, k_eboss, z)
    return gp, z, k_eboss, pf_eboss, cov, fid, p_fid


def _proj_model(base: P1DModel, fid_full: np.ndarray, sub_idx: list[int]) -> P1DModel:
    """Wrap a model so it accepts a sub-vector of length len(sub_idx); the
    other params are pinned to fid_full."""

    class _Proj:
        def predict(self, theta_sub, k, z):
            full = fid_full.copy()
            for i, idx in enumerate(sub_idx):
                full[idx] = theta_sub[i]
            return base.predict(full, k, z)

    return _Proj()


# ---------------------------------------------------------------------------
# Figure 1: GP at fiducial vs eBOSS data
# ---------------------------------------------------------------------------


def fig_gp_at_fiducial(*, k, p_fid, pf_eboss, cov, z, outpath):
    import matplotlib.pyplot as plt

    sigma = np.sqrt(np.diag(cov))
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.errorbar(k, pf_eboss, yerr=sigma, fmt=".", capsize=2, color="black",
                label=f"eBOSS DR14 (z={z})")
    ax.plot(k, p_fid, "C0-", lw=2, label="GP at fiducial")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$k$ [s/km]")
    ax.set_ylabel(r"$P_F(k)$")
    ax.set_title(f"GP prediction at fiducial vs eBOSS DR14, z={z}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outpath); plt.close(fig)
    print(f"wrote {outpath}")


# ---------------------------------------------------------------------------
# Figure 2: per-parameter sensitivity
# ---------------------------------------------------------------------------


def fig_param_sensitivity(*, gp, k, z, fid, outpath):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    p_fid = gp.predict(fid, k, z)
    palette = plt.get_cmap("tab20").colors
    for i, p in enumerate(PARAMS_11D):
        h = 0.02 * p.width()
        t_p = fid.copy(); t_p[i] += h
        t_m = fid.copy(); t_m[i] -= h
        dp = (gp.predict(t_p, k, z) - gp.predict(t_m, k, z)) / (2 * h)
        sens = dp * p.width() / p_fid  # dimensionless: d log P per prior width
        ax.plot(k, sens, lw=1.4, color=palette[i % len(palette)], label=p.name)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ [s/km]")
    ax.set_ylabel(r"$d\,\ln P_F / d\hat\theta$  (per prior-width)")
    ax.set_title(f"GP per-parameter sensitivity at fiducial, z={z}")
    ax.legend(fontsize=7, ncol=3, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outpath); plt.close(fig)
    print(f"wrote {outpath}")


# ---------------------------------------------------------------------------
# Figures 3-6: 1D / 2D / 3D / 4D Fisher
# ---------------------------------------------------------------------------


def _fisher_for_subset(*, gp, fid, k, z, names: list[str]):
    sub = tuple(p for p in PARAMS_11D if p.name in names)
    sub_idx = [PARAM_NAMES.index(p.name) for p in sub]
    fid_sub = np.array([fid[i] for i in sub_idx])
    proj = _proj_model(gp, fid, sub_idx)
    lk = GaussianLikelihood(model=proj, z=z, mock_data="gp", theta_fid=fid_sub)
    res = fisher_matrix(
        likelihood=lk, theta_fid=fid_sub, params=sub,
        step_frac=0.02, rel_tol=0.05, max_halvings=2,
    )
    return res, sub


def _bar_fisher_sigma(*, results: dict[str, "FisherResult"], outpath, title):
    """One bar per param showing sigma / prior_width."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    sample_label = next(iter(results))
    sample = results[sample_label]
    names = sample.param_names
    n = len(names)
    bar_w = 0.8 / len(results)
    x = np.arange(n)
    for i, (label, fr) in enumerate(results.items()):
        widths = np.array([get_param(name).width() for name in fr.param_names])
        ratio = fr.sigma / widths
        ax.bar(x + i * bar_w, ratio, width=bar_w, label=label)
    ax.set_xticks(x + 0.4 - bar_w / 2)
    ax.set_xticklabels(names)
    ax.set_ylabel(r"$\sigma_{\rm marg}\,/\,$prior width")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(outpath); plt.close(fig)
    print(f"wrote {outpath}")


def fig_fisher_1d(*, gp, fid, k, z, outpath):
    """1D forecast for each of the 4 sensitive params, run independently."""
    results = {}
    for name in ("ns", "Ap", "hub", "omegamh2"):
        res, _ = _fisher_for_subset(gp=gp, fid=fid, k=k, z=z, names=[name])
        results[name] = res
    # Summarize as a single bar chart of sigma / width.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    names = list(results)
    sigmas = [results[n].sigma[0] / get_param(n).width() for n in names]
    ax.bar(names, sigmas, color="C0")
    ax.set_ylabel(r"$\sigma_{\rm 1D}\,/\,$prior width")
    ax.set_title(f"1D Fisher: each parameter alone (others held at fid), z={z}")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(outpath); plt.close(fig)
    print(f"wrote {outpath}")
    return results


def fig_fisher_jointND(*, gp, fid, k, z, names_list: list[str], label: str, outpath):
    """Fisher on the joint subspace `names_list`."""
    res, sub = _fisher_for_subset(gp=gp, fid=fid, k=k, z=z, names=names_list)
    _bar_fisher_sigma(
        results={f"{label} joint": res},
        outpath=outpath,
        title=f"{label} Fisher forecast on {names_list}, z={z}",
    )
    return res


def fig_fisher_corner(*, fisher_results: dict, outpath):
    """4D Gaussian corner overlay."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    sample = next(iter(fisher_results.values()))
    names = list(sample.param_names)
    n = len(names)
    fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n), dpi=120)
    fig.patch.set_facecolor("white")
    palette = plt.get_cmap("tab10").colors
    for ai in range(n):
        for aj in range(n):
            ax = axes[ai, aj]
            ax.set_facecolor("white")
            if aj > ai:
                ax.set_visible(False); continue
            for ci, (label, fr) in enumerate(fisher_results.items()):
                color = palette[ci % len(palette)]
                if ai == aj:
                    s = fr.sigma[ai]; mu = fr.theta_fid[ai]
                    xs = np.linspace(mu - 4 * s, mu + 4 * s, 200)
                    ys = np.exp(-0.5 * ((xs - mu) / s) ** 2)
                    ax.plot(xs, ys, color=color, label=label)
                else:
                    sub = np.array([
                        [fr.cov[ai, ai], fr.cov[ai, aj]],
                        [fr.cov[aj, ai], fr.cov[aj, aj]],
                    ])
                    cov2 = sub[::-1, ::-1]
                    eigvals, eigvecs = np.linalg.eigh(cov2)
                    if np.all(eigvals > 0):
                        order = eigvals.argsort()[::-1]
                        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
                        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                        w_, h_ = 2 * np.sqrt(eigvals)
                        ell = mpatches.Ellipse(
                            xy=(fr.theta_fid[aj], fr.theta_fid[ai]),
                            width=w_, height=h_, angle=angle,
                            edgecolor=color, facecolor="none", lw=1.2,
                        )
                        ax.add_patch(ell)
                        ax.scatter(fr.theta_fid[aj], fr.theta_fid[ai],
                                   marker="x", color=color, s=18)
                        pad = 1.4 * np.sqrt(max(eigvals))
                        ax.set_xlim(fr.theta_fid[aj] - pad, fr.theta_fid[aj] + pad)
                        ax.set_ylim(fr.theta_fid[ai] - pad, fr.theta_fid[ai] + pad)
            if ai == n - 1:
                ax.set_xlabel(names[aj])
            if aj == 0:
                ax.set_ylabel(names[ai])
            ax.tick_params(labelsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(outpath); plt.close(fig)
    print(f"wrote {outpath}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    gp, z, k_eboss, pf_eboss, cov, fid, p_fid = _setup()

    fig_gp_at_fiducial(
        k=k_eboss, p_fid=p_fid, pf_eboss=pf_eboss, cov=cov, z=z,
        outpath=OUT / "fig01_gp_at_fiducial.png",
    )
    fig_param_sensitivity(
        gp=gp, k=k_eboss, z=z, fid=fid,
        outpath=OUT / "fig02_gp_param_sensitivity.png",
    )

    # 1D Fisher per parameter
    fig_fisher_1d(
        gp=gp, fid=fid, k=k_eboss, z=z,
        outpath=OUT / "fig03_fisher_1d.png",
    )

    # 2D
    fr_2d = fig_fisher_jointND(
        gp=gp, fid=fid, k=k_eboss, z=z,
        names_list=["ns", "Ap"], label="2D",
        outpath=OUT / "fig04_fisher_2d.png",
    )

    # 3D
    fr_3d = fig_fisher_jointND(
        gp=gp, fid=fid, k=k_eboss, z=z,
        names_list=["ns", "Ap", "hub"], label="3D",
        outpath=OUT / "fig05_fisher_3d.png",
    )

    # 4D
    fr_4d = fig_fisher_jointND(
        gp=gp, fid=fid, k=k_eboss, z=z,
        names_list=["ns", "Ap", "hub", "omegamh2"], label="4D",
        outpath=OUT / "fig06_fisher_4d.png",
    )

    # 4D corner
    fig_fisher_corner(
        fisher_results={"GP (4D)": fr_4d},
        outpath=OUT / "fig07_fisher_corner.png",
    )


if __name__ == "__main__":
    main()
