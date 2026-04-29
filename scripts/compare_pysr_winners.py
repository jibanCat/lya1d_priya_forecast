"""Overlay the GP, val_mse-best PySR, fisher_aware-best PySR, and
sigma_targeted-best PySR equations along the ns prior, at fixed k_mid.

Run after `scripts/run_pysr_hpo.py` has produced cache directories
under `results/pysr_hypothesis/refit_ns/`,
`refit_ns_fisher/`, and `refit_ns_sigma/`.

Output: `docs/figures/pysr_hypothesis/fig_three_metric_comparison.png`
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_LYAEMU = Path("/home/mfho/student_projects/lya_emulator_full")
if _LYAEMU.exists():
    sys.path.insert(0, str(_LYAEMU))

from priya_forecast.data import load_eboss
from priya_forecast.models.gp_model import GPModel
from priya_forecast.parameters import (
    PARAM_NAMES, fiducial_vector, get_param,
)


def _best_from_cache(cache_dir: Path, sort_by_field: str):
    files = sorted(cache_dir.glob("*.pkl"))
    if not files:
        return None
    results = []
    for f in files:
        with open(f, "rb") as fp:
            results.append(pickle.load(fp))
    def _key(r):
        if sort_by_field == "val_loss":
            return r.val_loss
        return r.extra_metrics.get(sort_by_field, float("inf"))
    return sorted(results, key=_key)[0]


def _eval_eq(expr_xN: str, ns: float, k: np.ndarray, k_min: float, k_max: float) -> np.ndarray:
    """Substitute x0 → ns_norm, x1 → k_norm and evaluate."""
    p = get_param("ns")
    ns_norm = (ns - p.prior[0]) / p.width()
    k_norm = (k - k_min) / (k_max - k_min)
    # Use sympy for safe parsing.
    import sympy as sp
    expr = sp.sympify(expr_xN)
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    fn = sp.lambdify(syms, expr, modules=["numpy"])
    args = []
    for s in syms:
        if s.name == "x0":
            args.append(np.full_like(k, ns_norm))
        elif s.name == "x1":
            args.append(k_norm)
        else:
            args.append(np.zeros_like(k))
    return np.asarray(fn(*args)).ravel()


def main():
    out = Path(__file__).resolve().parent.parent / "docs" / "figures" / "pysr_hypothesis"
    out.mkdir(parents=True, exist_ok=True)

    print("Loading GP...")
    gp = GPModel()
    z = 3.6
    k_eboss, _, _ = load_eboss(z=z)
    fid = np.array(fiducial_vector(), dtype=float)
    p = get_param("ns")
    k_mid_idx = len(k_eboss) // 2
    k_mid = float(k_eboss[k_mid_idx])

    # Three winners (use whichever cache dirs exist).
    winners = {}
    base = Path(__file__).resolve().parent.parent / "results" / "pysr_hypothesis"
    if (base / "refit_ns" / "cache").exists():
        winners["val_mse-best"] = _best_from_cache(base / "refit_ns" / "cache", "val_loss")
    if (base / "refit_ns_fisher" / "cache").exists():
        winners["fisher_aware-best"] = _best_from_cache(
            base / "refit_ns_fisher" / "cache", "fisher_residual")
    if (base / "refit_ns_sigma" / "cache").exists():
        winners["sigma_targeted-best"] = _best_from_cache(
            base / "refit_ns_sigma" / "cache", "sigma_ratio")

    ns_grid = np.linspace(p.prior[0], p.prior[1], 13)
    k_min, k_max = float(k_eboss.min()), float(k_eboss.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white"); ax.grid(alpha=0.3)

    # Truth (GP)
    gp_curve = []
    for ns in ns_grid:
        theta = fid.copy(); theta[PARAM_NAMES.index("ns")] = ns
        gp_curve.append(gp.predict(theta, k_eboss, z)[k_mid_idx])
    gp_curve = np.array(gp_curve)
    ax1.plot(ns_grid, gp_curve, "k-", lw=2.5, label="GP (truth)", zorder=10)
    ax2.plot(ns_grid, gp_curve / gp_curve[len(ns_grid) // 2], "k-", lw=2.5,
             label="GP (truth)", zorder=10)

    palette = ["C3", "C2", "C0"]
    for i, (label, r) in enumerate(winners.items()):
        if r is None: continue
        eq = r.best_expression
        # Need the multiplicative-combine result: P_pysr = P_fid * f(theta) / f(fid).
        # Compute f(theta) at each ns at k_mid, divide by f(fid), multiply by GP at fid.
        f_curve = np.array([_eval_eq(eq, ns, np.array([k_mid]), k_min, k_max)[0] for ns in ns_grid])
        f_at_fid = _eval_eq(eq, p.fid, np.array([k_mid]), k_min, k_max)[0]
        gp_fid = gp_curve[len(ns_grid) // 2]
        pysr_curve = gp_fid * f_curve / f_at_fid
        ax1.plot(ns_grid, pysr_curve, "--", lw=1.6, color=palette[i % len(palette)],
                 label=label, zorder=5 - i)
        ax2.plot(ns_grid, pysr_curve / pysr_curve[len(ns_grid) // 2],
                 "--", lw=1.6, color=palette[i % len(palette)], label=label, zorder=5 - i)

    ax1.axvline(p.fid, color="grey", lw=0.5, ls=":")
    ax2.axvline(p.fid, color="grey", lw=0.5, ls=":")
    ax2.axhline(1.0, color="grey", lw=0.5)
    ax1.set_xlabel("ns"); ax1.set_ylabel(r"$P_F(k_{\rm mid})$")
    ax2.set_xlabel("ns"); ax2.set_ylabel(r"$P_F / P_F({\rm fid})$")
    ax1.set_title("Three HPO metrics — absolute P_F at k_mid")
    ax2.set_title("Three HPO metrics — relative response (slope)")
    ax1.legend(fontsize=9); ax2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_three_metric_comparison.png")
    plt.close(fig)
    print(f"wrote {out}/fig_three_metric_comparison.png")


if __name__ == "__main__":
    main()
