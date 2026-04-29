"""Multi-D PySR diagnostic: how hard is multi-parameter symbolic regression?

The student's published pipeline trains one PySR equation per parameter and
combines them multiplicatively. This factorization is exact only if
``P(theta, k)`` is separable in θ_i. Real Lyα P_F is *not* exactly
separable — the question is *how big the cost is*.

This module quantifies that cost across three regimes on the *same*
Sobol training set:

1. ``regime="1D"``        — one fit per parameter, others at fid.
                            Combine: P_pysr = P_fid · ∏_i f_i(θ_i, k) / f_i(θ_fid_i, k).
2. ``regime="2D_pairs"``  — one fit per pair (θ_i, θ_j) with the rest at
                            fid. Compare against the 1D-product on the
                            same pair to extract the *coupling residual*.
3. ``regime="full_kD"``   — one fit on the full subspace; the upper
                            bound for what symbolic regression can do.

Headline output: an 11x11 *coupling matrix*

   C_ij = (MSE_1D_product[i,j] - MSE_2D_joint[i,j]) / MSE_1D_product[i,j]

Cells near zero → 1D-factorization is fine for that pair. Cells dark →
the pair has non-trivial cross-coupling that 1D-product is missing.

Backend: defaults to a fast polynomial surrogate (numpy lstsq) so the
full 11x11 sweep finishes in seconds; pass `pysr_kwargs={...}` to swap
in real PySR for higher-fidelity equations (slow — minutes per cell).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations, combinations_with_replacement
from pathlib import Path
from typing import Callable

import numpy as np

from priya_forecast.parameters import (
    PARAM_NAMES,
    fiducial_vector,
    get_param,
)


@dataclass
class DiagnosticResult:
    """One regime × parameter-subset run."""

    regime: str
    n_params_varied: int
    param_names: list[str]
    train_mse: float
    test_mse: float
    pareto_complexities: list[int] = field(default_factory=list)
    pareto_losses: list[float] = field(default_factory=list)
    best_expression: str = ""
    wall_time_s: float = 0.0
    n_train: int = 0
    n_test: int = 0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Polynomial surrogate: fast PySR-style fits via least-squares
# ---------------------------------------------------------------------------


def _multinomial_terms(n_dim: int, max_total_degree: int):
    out = []
    for d in range(max_total_degree + 1):
        for combo in combinations_with_replacement(range(n_dim), d):
            out.append(tuple(combo.count(i) for i in range(n_dim)))
    return out


def _design(X: np.ndarray, terms) -> np.ndarray:
    return np.column_stack([np.prod(X ** np.asarray(t), axis=1) for t in terms])


def _fit_poly(X: np.ndarray, y: np.ndarray, *, order: int):
    terms = _multinomial_terms(X.shape[1], order)
    A = _design(X, terms)
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coeffs, terms


def _eval_poly(X: np.ndarray, coeffs, terms) -> np.ndarray:
    return _design(X, terms) @ coeffs


# ---------------------------------------------------------------------------
# Sobol sampling that matches train_and_forecast.py
# ---------------------------------------------------------------------------


def _sobol_thetas(*, varying_names: list[str], n: int, seed: int) -> np.ndarray:
    from scipy.stats import qmc

    fid = np.array(fiducial_vector(), dtype=float)
    sampler = qmc.Sobol(d=len(varying_names), seed=seed)
    u = sampler.random(n=n)
    out = np.tile(fid, (n, 1))
    for col, name in enumerate(varying_names):
        lo, hi = get_param(name).prior
        out[:, PARAM_NAMES.index(name)] = lo + (hi - lo) * u[:, col]
    return out


def _sobol_design_for_fit(*, gp, varying_names, n, k_grid, z, seed):
    """Return X in [0,1] (per-param normalized) of shape (n*nk, len(varying)+1)
    and y of shape (n*nk,) holding the GP prediction."""
    thetas = _sobol_thetas(varying_names=varying_names, n=n, seed=seed)
    flux = np.stack([gp.predict(t, k_grid, z) for t in thetas], axis=0)
    k_norm = (k_grid - k_grid.min()) / (k_grid.max() - k_grid.min())
    X_rows, y_rows = [], []
    p_norm = np.empty((n, len(varying_names)))
    for col, name in enumerate(varying_names):
        lo, hi = get_param(name).prior
        p_norm[:, col] = (thetas[:, PARAM_NAMES.index(name)] - lo) / (hi - lo)
    for i in range(n):
        for ki in range(k_grid.size):
            X_rows.append(list(p_norm[i]) + [k_norm[ki]])
            y_rows.append(flux[i, ki])
    return np.asarray(X_rows), np.asarray(y_rows)


# ---------------------------------------------------------------------------
# Polynomial-backend implementations of the three regimes
# ---------------------------------------------------------------------------


def _train_test_split(X, y, frac_test: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(frac_test * len(X))
    return X[idx[n_test:]], y[idx[n_test:]], X[idx[:n_test]], y[idx[:n_test]]


def _run_poly_for_subset(
    *, gp, varying: list[str], k_grid, z, n_train: int, order: int, seed: int,
) -> tuple[float, float, "_FittedPoly", float]:
    """Generic polynomial fit over `varying`. Returns train_mse, test_mse,
    fitted callable, wall-time."""
    X, y = _sobol_design_for_fit(gp=gp, varying_names=varying, n=n_train,
                                 k_grid=k_grid, z=z, seed=seed)
    Xtr, ytr, Xte, yte = _train_test_split(X, y, frac_test=0.2, seed=seed)
    t0 = time.time()
    coeffs, terms = _fit_poly(Xtr, ytr, order=order)
    elapsed = time.time() - t0
    train_mse = float(np.mean((_eval_poly(Xtr, coeffs, terms) - ytr) ** 2))
    test_mse = float(np.mean((_eval_poly(Xte, coeffs, terms) - yte) ** 2))
    return train_mse, test_mse, _FittedPoly(coeffs=coeffs, terms=terms, varying=varying), elapsed


@dataclass
class _FittedPySR:
    model: object  # the trained PySRRegressor
    pareto: object  # equations_ DataFrame
    varying: list[str]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=float).ravel()


def _run_pysr_for_subset(
    *, gp, varying: list[str], k_grid, z, n_train: int, pysr_kwargs: dict, seed: int,
) -> tuple[float, float, "_FittedPySR", float, list[int], list[float]]:
    """Like `_run_poly_for_subset` but trains a real PySRRegressor."""
    try:
        from pysr import PySRRegressor
    except ImportError as e:
        raise ImportError("pysr_kwargs= requires PySR. Install pysr or omit it.") from e
    X, y = _sobol_design_for_fit(gp=gp, varying_names=varying, n=n_train,
                                 k_grid=k_grid, z=z, seed=seed)
    Xtr, ytr, Xte, yte = _train_test_split(X, y, frac_test=0.2, seed=seed)
    defaults = dict(
        niterations=30, maxsize=20,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "exp", "square"],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        random_state=seed, deterministic=True, parallelism="serial", verbosity=0,
    )
    defaults.update(pysr_kwargs or {})
    t0 = time.time()
    model = PySRRegressor(**defaults)
    model.fit(Xtr, ytr)
    elapsed = time.time() - t0
    pareto = model.equations_
    train_mse = float(np.mean((np.asarray(model.predict(Xtr)).ravel() - ytr) ** 2))
    test_mse = float(np.mean((np.asarray(model.predict(Xte)).ravel() - yte) ** 2))
    fitted = _FittedPySR(model=model, pareto=pareto, varying=varying)
    return (
        train_mse, test_mse, fitted, elapsed,
        pareto["complexity"].astype(int).tolist(),
        pareto["loss"].astype(float).tolist(),
    )


@dataclass
class _FittedPoly:
    coeffs: np.ndarray
    terms: list
    varying: list[str]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _eval_poly(X, self.coeffs, self.terms)


def _expand_pair_to_design(*, gp, pair, k_grid, z, n_test, seed):
    """Independent test set for a (param_i, param_j) pair."""
    X, y = _sobol_design_for_fit(
        gp=gp, varying_names=list(pair), n=n_test, k_grid=k_grid, z=z, seed=seed + 9999,
    )
    return X, y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_diagnostic(
    *,
    gp_model,
    z: float,
    k_grid: np.ndarray,
    param_names: list[str],
    regime: str,
    n_train: int = 128,
    n_test: int = 256,
    poly_order: int = 4,
    pysr_kwargs: dict | None = None,
    seed: int = 0,
) -> list[DiagnosticResult]:
    """Run one regime across the chosen `param_names`.

    Backend: polynomial least-squares by default. Pass
    ``pysr_kwargs={"niterations": 100, "maxsize": 30, ...}`` to swap in
    real PySR — slow (minutes per cell) but produces interpretable
    equations.

    Returns one DiagnosticResult per "experiment":
      - 1D       : len(param_names) results, one per parameter.
      - 2D_pairs : choose(n, 2) results, one per pair.
      - full_kD  : 1 result on the full subspace.
    """
    if regime not in {"1D", "2D_pairs", "full_kD"}:
        raise ValueError(f"Unknown regime {regime!r}.")

    use_pysr = bool(pysr_kwargs)

    def _train(varying):
        if use_pysr:
            return _run_pysr_for_subset(
                gp=gp_model, varying=varying, k_grid=k_grid, z=z,
                n_train=n_train, pysr_kwargs=pysr_kwargs, seed=seed,
            )
        tr, te, fit, dt = _run_poly_for_subset(
            gp=gp_model, varying=varying, k_grid=k_grid, z=z,
            n_train=n_train, order=poly_order, seed=seed,
        )
        return tr, te, fit, dt, [], []

    results: list[DiagnosticResult] = []

    if regime == "1D":
        for name in param_names:
            tr, te, fit, dt, complexities, losses = _train([name])
            backend = "pysr" if use_pysr else f"poly(order={poly_order})"
            results.append(DiagnosticResult(
                regime=regime, n_params_varied=1, param_names=[name],
                train_mse=tr, test_mse=te, wall_time_s=dt,
                n_train=n_train, n_test=n_test,
                best_expression=f"{backend} over [{name}, k]",
                pareto_complexities=complexities, pareto_losses=losses,
            ))
        return results

    if regime == "2D_pairs":
        # Cache 1D fits per param so the 1D-product baseline is reusable.
        oned: dict[str, object] = {}
        oned_test_mse: dict[str, float] = {}
        for name in param_names:
            tr, te, fit, dt, *_ = _train([name])
            oned[name] = fit
            oned_test_mse[name] = te

        # Need a fid prediction to combine 1D fits multiplicatively.
        fid = np.array(fiducial_vector(), dtype=float)
        p_fid = gp_model.predict(fid, k_grid, z)

        for pair in combinations(param_names, 2):
            tr_j, te_j, fit_j, dt_j, complexities, losses = _train(list(pair))
            # 1D-product MSE on the *same* 2D test set
            X_te, y_te = _expand_pair_to_design(
                gp=gp_model, pair=pair, k_grid=k_grid, z=z,
                n_test=n_test, seed=seed,
            )
            # 1D-product: P = P_fid * f_i(θ_i)/f_i_fid(θ_fid_i) * f_j(θ_j)/f_j_fid(θ_fid_j).
            # All three quantities evaluated on the same (θ_i_norm, θ_j_norm, k_norm)
            # row of X_te by slicing the appropriate columns.
            i, j = pair
            # f_i is a poly over [i, k]. Build its X by columns 0 and 2 of X_te.
            X_i = np.column_stack([X_te[:, 0], X_te[:, 2]])
            X_j = np.column_stack([X_te[:, 1], X_te[:, 2]])
            # fid k_norm is the same column 2.
            k_te = X_te[:, 2]
            X_fid_i = np.column_stack([np.full_like(k_te, _norm_fid(i)), k_te])
            X_fid_j = np.column_stack([np.full_like(k_te, _norm_fid(j)), k_te])
            f_i  = oned[i].predict(X_i)
            f_if = oned[i].predict(X_fid_i)
            f_j  = oned[j].predict(X_j)
            f_jf = oned[j].predict(X_fid_j)
            # Interpolate p_fid onto the test k-grid (k_norm → physical k).
            k_phys = X_te[:, 2] * (k_grid.max() - k_grid.min()) + k_grid.min()
            p_fid_interp = np.interp(k_phys, k_grid, p_fid)
            with np.errstate(divide="ignore", invalid="ignore"):
                pred_1d = p_fid_interp * (f_i / f_if) * (f_j / f_jf)
            mse_1d_prod = float(np.mean((pred_1d - y_te) ** 2))
            # 2D joint test MSE on the same X_te.
            mse_2d_joint = float(np.mean((fit_j.predict(X_te) - y_te) ** 2))
            # Coupling metric.
            coupling = (mse_1d_prod - mse_2d_joint) / max(mse_1d_prod, 1e-30)
            backend = "pysr" if use_pysr else f"poly(order={poly_order})"
            results.append(DiagnosticResult(
                regime=regime, n_params_varied=2, param_names=list(pair),
                train_mse=tr_j, test_mse=mse_2d_joint, wall_time_s=dt_j,
                n_train=n_train, n_test=n_test,
                best_expression=f"{backend} over [{i}, {j}, k]",
                pareto_complexities=complexities, pareto_losses=losses,
                extra={
                    "mse_1D_product": mse_1d_prod,
                    "mse_2D_joint": mse_2d_joint,
                    "coupling": coupling,
                },
            ))
        return results

    # full_kD
    tr, te, fit, dt, complexities, losses = _train(param_names)
    backend = "pysr" if use_pysr else f"poly(order={poly_order})"
    results.append(DiagnosticResult(
        regime=regime, n_params_varied=len(param_names),
        param_names=list(param_names),
        train_mse=tr, test_mse=te, wall_time_s=dt,
        n_train=n_train, n_test=n_test,
        best_expression=f"{backend} over [{','.join(param_names)}, k]",
        pareto_complexities=complexities, pareto_losses=losses,
    ))
    return results


def _norm_fid(name: str) -> float:
    """Normalized fiducial of a parameter: (fid - lo) / (hi - lo)."""
    p = get_param(name)
    return (p.fid - p.prior[0]) / p.width()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_diagnostic(results_by_regime: dict[str, list[DiagnosticResult]],
                    *, outdir: str | Path) -> Path:
    """Produce the four headline plots:
    1. Scaling: test_mse vs n_params_varied per regime.
    2. Wall-time vs n_params_varied per regime.
    3. Coupling-matrix heatmap (from 2D_pairs).
    4. (deferred — Pareto-front overlay needs PySR backend)
    """
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- 1. Scaling plot ---
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    palette = plt.get_cmap("tab10").colors
    for ci, (regime, results) in enumerate(results_by_regime.items()):
        if not results:
            continue
        ns = [r.n_params_varied for r in results]
        mses = [r.test_mse for r in results]
        ax.scatter(ns, mses, color=palette[ci], label=regime, s=30)
    ax.set_yscale("log")
    ax.set_xlabel("n_params varied")
    ax.set_ylabel("Test MSE  [P_F²]")
    ax.set_title("Multi-D diagnostic: test MSE vs dimensionality")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outdir / "diag1_scaling.png"); plt.close(fig)

    # --- 2. Wall-time plot ---
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    for ci, (regime, results) in enumerate(results_by_regime.items()):
        if not results:
            continue
        ns = [r.n_params_varied for r in results]
        ts = [r.wall_time_s for r in results]
        ax.scatter(ns, ts, color=palette[ci], label=regime, s=30)
    ax.set_xlabel("n_params varied")
    ax.set_ylabel("Wall-time per fit  [s]")
    ax.set_title("Multi-D diagnostic: training cost vs dimensionality")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outdir / "diag2_walltime.png"); plt.close(fig)

    # --- 3. Coupling-matrix heatmap (the headline plot) ---
    pairs = results_by_regime.get("2D_pairs", [])
    if pairs:
        names = sorted({n for r in pairs for n in r.param_names})
        n = len(names)
        idx = {nm: i for i, nm in enumerate(names)}
        C = np.zeros((n, n))
        for r in pairs:
            i, j = idx[r.param_names[0]], idx[r.param_names[1]]
            c = float(r.extra.get("coupling", 0.0))
            C[i, j] = c
            C[j, i] = c
        np.fill_diagonal(C, np.nan)
        # Diverging colormap: positive coupling (joint beats product) → red,
        # negative (product beats joint at this budget) → blue. Clip at ±2
        # so a single huge-magnitude cell doesn't compress everything else.
        vmax = float(np.nanpercentile(np.abs(C), 95))
        vmax = max(min(vmax, 2.0), 0.3)
        fig, ax = plt.subplots(figsize=(0.85 * n + 2, 0.85 * n + 1.5), dpi=120)
        fig.patch.set_facecolor("white"); ax.set_facecolor("white")
        masked = np.ma.masked_invalid(C)
        im = ax.imshow(masked, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=8)
                    continue
                v = C[i, j]
                txt = f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"
                color = "white" if abs(v) > 0.7 * vmax else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=color)
        ax.set_title(
            "Coupling matrix:  (MSE_1D-product − MSE_2D-joint) / MSE_1D-product\n"
            "Red = joint helps (cross-coupling); Blue = product wins at this training density",
            fontsize=10,
        )
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("coupling fraction")
        fig.tight_layout(); fig.savefig(outdir / "diag3_coupling_matrix.png"); plt.close(fig)

    return outdir
