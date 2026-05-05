"""Hypothesis-testing harness for "why does PySR underperform the GP?".

Each `experiment_*` function runs one controlled comparison and returns a
result dict. They share a tiny synthetic GP-like target so they're fast
enough to run in unit tests AND in the full driver script.

The synthetic target mimics the qualitative shape of the real Lyα P_F:

  P_target(theta, k) = (A0 + amp_ns * dns + amp_Ap * dAp) * k^alpha * exp(-k * scale)

with `dns`, `dAp` ∈ [0, 1] (normalized perturbations). This has a known,
analytic structure so we can disentangle PySR-fitting effects from
GP-modeling effects.

Run:
    PYTHONPATH=src python -m priya_forecast.pysr_hypothesis
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic ground truth
# ---------------------------------------------------------------------------


def synthetic_p_f(theta_norm: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Synthetic Lyα-shaped P_F. theta_norm in [0,1]^d, k in physical (s/km).

    Form: A(theta) * k^alpha * exp(-k * scale)
    where A = 50 * (1 + 0.30*ns - 0.50*Ap)
          alpha = -0.5
          scale = 8.0
    """
    if theta_norm.ndim == 1:
        theta_norm = theta_norm[None, :]
    if theta_norm.shape[-1] >= 2:
        ns_n, Ap_n = theta_norm[..., 0], theta_norm[..., 1]
    elif theta_norm.shape[-1] == 1:
        ns_n = theta_norm[..., 0]
        Ap_n = np.zeros_like(ns_n)
    else:
        raise ValueError("theta_norm must have at least 1 column")
    A = 50.0 * (1.0 + 0.30 * (ns_n - 0.5) - 0.50 * (Ap_n - 0.5))
    out = A[..., None] * (k ** -0.5) * np.exp(-k * 8.0)
    return out.squeeze()


# ---------------------------------------------------------------------------
# Lightweight polynomial trainer (PySR proxy)
# ---------------------------------------------------------------------------


def _multinomial_terms(n_dim: int, max_degree: int):
    out = []
    for d in range(max_degree + 1):
        for c in combinations_with_replacement(range(n_dim), d):
            out.append(tuple(c.count(i) for i in range(n_dim)))
    return out


def _design(X: np.ndarray, terms) -> np.ndarray:
    return np.column_stack([np.prod(X ** np.asarray(t), axis=1) for t in terms])


def fit_polynomial(X_train: np.ndarray, y_train: np.ndarray, *, max_degree: int):
    terms = _multinomial_terms(X_train.shape[1], max_degree)
    coeffs, *_ = np.linalg.lstsq(_design(X_train, terms), y_train, rcond=None)
    return coeffs, terms


def fit_polynomial_with_parsimony(
    X_train: np.ndarray, y_train: np.ndarray, *, max_degree: int, parsimony: float,
):
    """LASSO-flavoured polynomial fit: drops terms whose coefficient magnitude
    is below `parsimony * max_coeff_magnitude`. Mimics PySR's parsimony
    pressure (where small-impact terms get pruned)."""
    coeffs, terms = fit_polynomial(X_train, y_train, max_degree=max_degree)
    threshold = parsimony * np.max(np.abs(coeffs))
    kept = np.abs(coeffs) >= threshold
    return coeffs * kept, terms


def predict_polynomial(X: np.ndarray, coeffs, terms) -> np.ndarray:
    return _design(X, terms) @ coeffs


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    name: str
    train_mse: float
    test_mse: float
    test_mse_near_fid: float
    fisher_sigma: dict[str, float]
    extra: dict = field(default_factory=dict)


def _build_dataset(*, n_train: int, n_test: int, k_grid: np.ndarray, seed: int,
                   concentrate_near_fid: bool = False):
    """Sobol-style sweep of (ns_norm, Ap_norm) ∈ [0,1]² + per-k expansion."""
    from scipy.stats import qmc

    sampler = qmc.Sobol(d=2, seed=seed)
    n_total = n_train + n_test
    u = sampler.random(n=n_total)
    if concentrate_near_fid:
        # Squeeze to a smaller box around (0.5, 0.5) to concentrate density.
        u = 0.5 + (u - 0.5) * 0.3
    rows_X, rows_y = [], []
    for i in range(n_total):
        for k in k_grid:
            row = [u[i, 0], u[i, 1], (k - k_grid.min()) / (k_grid.max() - k_grid.min())]
            rows_X.append(row)
            rows_y.append(float(synthetic_p_f(u[i], np.array([k]))))
    X = np.asarray(rows_X)
    y = np.asarray(rows_y)
    n_split = n_train * len(k_grid)
    return X[:n_split], y[:n_split], X[n_split:], y[n_split:]


def _eval_fisher_sigma(coeffs, terms, *, k_grid, h: float = 0.005):
    """Approximate Fisher σ at fid (theta_norm = 0.5) from a polynomial fit.

    For an unweighted gradient estimate:
      F_ii ~ Σ_k (df/dθ_i)² / σ_k²  (we use σ_k = constant for simplicity)
    σ_i = 1 / sqrt(F_ii). Returns ratios over the synthetic-truth Fisher.
    """
    fid_norm = np.array([0.5, 0.5])
    k_norm = (k_grid - k_grid.min()) / (k_grid.max() - k_grid.min())
    sigma_dummy = 1.0  # constant cov for the diagnostic
    sigmas = {}
    for i, name in enumerate(("ns", "Ap")):
        # Polynomial gradient at fid via finite diff.
        x_p = np.column_stack([np.full_like(k_norm, fid_norm[0]),
                               np.full_like(k_norm, fid_norm[1]), k_norm])
        x_m = x_p.copy()
        x_p[:, i] = fid_norm[i] + h
        x_m[:, i] = fid_norm[i] - h
        # Need to set the OTHER param to fid as well.
        x_p[:, 1 - i] = fid_norm[1 - i]
        x_m[:, 1 - i] = fid_norm[1 - i]
        df = (predict_polynomial(x_p, coeffs, terms) -
              predict_polynomial(x_m, coeffs, terms)) / (2 * h)
        F_ii = float(np.sum(df ** 2) / sigma_dummy ** 2)
        sigmas[name] = float(1.0 / np.sqrt(F_ii)) if F_ii > 0 else float("inf")
    return sigmas


def _truth_fisher_sigma(*, k_grid, h: float = 0.005):
    """Same Fisher σ but evaluated on the analytic truth."""
    fid = np.array([0.5, 0.5])
    sigmas = {}
    for i, name in enumerate(("ns", "Ap")):
        t_p = fid.copy(); t_p[i] += h
        t_m = fid.copy(); t_m[i] -= h
        df = (synthetic_p_f(t_p, k_grid) - synthetic_p_f(t_m, k_grid)) / (2 * h)
        F_ii = float(np.sum(df ** 2))
        sigmas[name] = float(1.0 / np.sqrt(F_ii)) if F_ii > 0 else float("inf")
    return sigmas


def _filter_near_fid(X, y, half_width=0.15):
    """Subset: rows where both ns_norm and Ap_norm are within `half_width` of 0.5."""
    mask = (np.abs(X[:, 0] - 0.5) < half_width) & (np.abs(X[:, 1] - 0.5) < half_width)
    return X[mask], y[mask]


def experiment_h1_loss_function(*, n_train: int = 64, seed: int = 0,
                                k_grid: np.ndarray | None = None) -> ExperimentResult:
    """H1: Does training only on near-fid samples improve Fisher σ at fid?"""
    if k_grid is None:
        k_grid = np.linspace(0.001, 0.02, 12)
    X_tr_full, y_tr_full, X_te, y_te = _build_dataset(
        n_train=n_train, n_test=128, k_grid=k_grid, seed=seed, concentrate_near_fid=False,
    )
    X_tr_near, y_tr_near, _, _ = _build_dataset(
        n_train=n_train, n_test=8, k_grid=k_grid, seed=seed, concentrate_near_fid=True,
    )
    coef_full, terms = fit_polynomial(X_tr_full, y_tr_full, max_degree=4)
    coef_near, _    = fit_polynomial(X_tr_near, y_tr_near, max_degree=4)
    truth = _truth_fisher_sigma(k_grid=k_grid)
    sig_full = _eval_fisher_sigma(coef_full, terms, k_grid=k_grid)
    sig_near = _eval_fisher_sigma(coef_near, terms, k_grid=k_grid)
    Xnf, ynf = _filter_near_fid(X_te, y_te)
    return ExperimentResult(
        name="h1_loss_function",
        train_mse=float(np.mean((predict_polynomial(X_tr_full, coef_full, terms) - y_tr_full) ** 2)),
        test_mse=float(np.mean((predict_polynomial(X_te, coef_full, terms) - y_te) ** 2)),
        test_mse_near_fid=float(np.mean((predict_polynomial(Xnf, coef_full, terms) - ynf) ** 2)),
        fisher_sigma=sig_full,
        extra={
            "fisher_sigma_near_fid_training": sig_near,
            "fisher_sigma_truth": truth,
            "ratio_full_to_truth_ns": sig_full["ns"] / truth["ns"],
            "ratio_near_to_truth_ns": sig_near["ns"] / truth["ns"],
        },
    )


def experiment_h2_parsimony(*, n_train: int = 64, seed: int = 0,
                            k_grid: np.ndarray | None = None) -> ExperimentResult:
    """H2: How does parsimony pruning affect which parameters appear in the equation?"""
    if k_grid is None:
        k_grid = np.linspace(0.001, 0.02, 12)
    X_tr, y_tr, X_te, y_te = _build_dataset(
        n_train=n_train, n_test=128, k_grid=k_grid, seed=seed,
    )
    sweep = {}
    for parsimony in [0.0, 1e-3, 1e-2, 1e-1]:
        coef, terms = fit_polynomial_with_parsimony(
            X_tr, y_tr, max_degree=4, parsimony=parsimony,
        )
        # Count how many "parameter-i-active" terms survived
        n_dim = X_tr.shape[1]
        kept_ns_terms = sum(1 for c, t in zip(coef, terms) if abs(c) > 1e-15 and t[0] > 0)
        kept_Ap_terms = sum(1 for c, t in zip(coef, terms) if abs(c) > 1e-15 and t[1] > 0)
        kept_total = int(np.sum(np.abs(coef) > 1e-15))
        test_mse = float(np.mean((predict_polynomial(X_te, coef, terms) - y_te) ** 2))
        sweep[f"parsimony_{parsimony:.0e}"] = {
            "kept_ns_terms": kept_ns_terms,
            "kept_Ap_terms": kept_Ap_terms,
            "kept_total": kept_total,
            "test_mse": test_mse,
        }
    return ExperimentResult(
        name="h2_parsimony", train_mse=0.0, test_mse=0.0, test_mse_near_fid=0.0,
        fisher_sigma={}, extra=sweep,
    )


def experiment_h3_normalization(*, n_train: int = 64, seed: int = 0,
                                k_grid: np.ndarray | None = None) -> ExperimentResult:
    """H3: Does training on flux_norm vs raw P_F change Fisher σ?"""
    if k_grid is None:
        k_grid = np.linspace(0.001, 0.02, 12)
    X_tr, y_tr, X_te, y_te = _build_dataset(
        n_train=n_train, n_test=128, k_grid=k_grid, seed=seed,
    )
    # Compute per-k mean/std from training set.
    n_k = len(k_grid)
    y_tr_grid = y_tr.reshape(n_train, n_k)
    mean_k = y_tr_grid.mean(axis=0)
    std_k = y_tr_grid.std(axis=0, ddof=0)
    std_k = np.where(std_k > 0, std_k, 1.0)
    # Repeat per (theta, k) for normalization.
    mean_train = np.tile(mean_k, n_train)
    std_train = np.tile(std_k, n_train)
    mean_test = np.tile(mean_k, len(y_te) // n_k)
    std_test = np.tile(std_k, len(y_te) // n_k)
    y_tr_norm = (y_tr - mean_train) / std_train
    y_te_norm = (y_te - mean_test) / std_test

    # Fit on raw P_F vs flux_norm.
    coef_raw, terms = fit_polynomial(X_tr, y_tr, max_degree=4)
    coef_norm, _ = fit_polynomial(X_tr, y_tr_norm, max_degree=4)
    # Eval test MSE in P_F units (denormalize).
    raw_pred_test = predict_polynomial(X_te, coef_raw, terms)
    norm_pred_test = predict_polynomial(X_te, coef_norm, terms) * std_test + mean_test
    sigma_truth = _truth_fisher_sigma(k_grid=k_grid)
    sigma_raw = _eval_fisher_sigma(coef_raw, terms, k_grid=k_grid)
    return ExperimentResult(
        name="h3_normalization",
        train_mse=float(np.mean((predict_polynomial(X_tr, coef_raw, terms) - y_tr) ** 2)),
        test_mse=float(np.mean((raw_pred_test - y_te) ** 2)),
        test_mse_near_fid=float(np.mean((norm_pred_test - y_te) ** 2)),
        fisher_sigma=sigma_raw,
        extra={
            "fisher_sigma_truth": sigma_truth,
            "test_mse_raw": float(np.mean((raw_pred_test - y_te) ** 2)),
            "test_mse_norm_then_denorm": float(np.mean((norm_pred_test - y_te) ** 2)),
        },
    )


def experiment_h4_operators(*, n_train: int = 64, seed: int = 0,
                            k_grid: np.ndarray | None = None) -> ExperimentResult:
    """H4: How much does having `exp` as an operator help?

    Polynomial proxy: degree-N polynomial vs degree-N polynomial-of-exp(-k).
    The synthetic truth has exp(-k * scale), so a polynomial-only basis
    needs many terms to approximate exp; adding exp(-k) as a basis
    function makes the fit dramatically easier.
    """
    if k_grid is None:
        k_grid = np.linspace(0.001, 0.02, 12)
    X_tr, y_tr, X_te, y_te = _build_dataset(
        n_train=n_train, n_test=128, k_grid=k_grid, seed=seed,
    )
    # Plain polynomial basis (3 vars).
    coef_poly, terms_poly = fit_polynomial(X_tr, y_tr, max_degree=4)
    # Augmented basis: include exp(-c * k_norm) for c=0.5, 1, 2, 4, 8.
    k_norm_train = X_tr[:, 2]
    k_norm_test = X_te[:, 2]
    cs = [0.5, 1.0, 2.0, 4.0, 8.0]
    extra_train = np.column_stack([np.exp(-c * k_norm_train) for c in cs])
    extra_test = np.column_stack([np.exp(-c * k_norm_test) for c in cs])
    X_tr_aug = np.column_stack([X_tr, extra_train])
    X_te_aug = np.column_stack([X_te, extra_test])
    coef_aug, terms_aug = fit_polynomial(X_tr_aug, y_tr, max_degree=3)
    return ExperimentResult(
        name="h4_operators",
        train_mse=float(np.mean((predict_polynomial(X_tr, coef_poly, terms_poly) - y_tr) ** 2)),
        test_mse=float(np.mean((predict_polynomial(X_te, coef_poly, terms_poly) - y_te) ** 2)),
        test_mse_near_fid=float(np.mean((predict_polynomial(X_te_aug, coef_aug, terms_aug) - y_te) ** 2)),
        fisher_sigma={},
        extra={
            "test_mse_polynomial_only": float(np.mean(
                (predict_polynomial(X_te, coef_poly, terms_poly) - y_te) ** 2)),
            "test_mse_with_exp_basis": float(np.mean(
                (predict_polynomial(X_te_aug, coef_aug, terms_aug) - y_te) ** 2)),
        },
    )


def experiment_h6_covariance_combine(*, n_train: int = 64, seed: int = 0,
                                     k_grid: np.ndarray | None = None) -> ExperimentResult:
    """H6: Does adding an explicit cross-term help when 1D-product fails?

    Synthetic non-separable truth: P = P_fid * (1 + a*dns) * (1 + b*dAp) + cross*ns*Ap*k.
    Compare:
      M0 = pure 1D-product
      M1 = 1D-product + explicit cross-term (a polynomial in ns*Ap)
    """
    if k_grid is None:
        k_grid = np.linspace(0.001, 0.02, 12)

    def truth(theta_norm, k):
        dns = theta_norm[..., 0] - 0.5
        dAp = theta_norm[..., 1] - 0.5
        sep = (1.0 + 0.5 * dns) * (1.0 + 0.3 * dAp)
        cross = 2.0 * dns * dAp * k
        return (50.0 * sep[..., None] * (k ** -0.5) * np.exp(-k * 8.0)) + cross

    from scipy.stats import qmc
    sampler = qmc.Sobol(d=2, seed=seed)
    u = sampler.random(n=n_train + 64)
    rows_X, rows_y = [], []
    for i in range(len(u)):
        for k in k_grid:
            rows_X.append([u[i, 0], u[i, 1], (k - k_grid.min()) / (k_grid.max() - k_grid.min())])
            rows_y.append(float(truth(u[i], np.array([k]))))
    X = np.asarray(rows_X)
    y = np.asarray(rows_y)
    n_split = n_train * len(k_grid)
    X_tr, y_tr = X[:n_split], y[:n_split]
    X_te, y_te = X[n_split:], y[n_split:]

    # M0: 1D-product = polynomial in (ns, k) plus polynomial in (Ap, k), then product.
    # We approximate by polynomial fit that EXCLUDES ns*Ap interaction terms.
    terms_full = _multinomial_terms(3, 4)
    terms_no_cross = [t for t in terms_full if not (t[0] > 0 and t[1] > 0)]
    A = np.column_stack([np.prod(X_tr ** np.asarray(t), axis=1) for t in terms_no_cross])
    coef_no_cross, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    pred_no_cross = np.column_stack(
        [np.prod(X_te ** np.asarray(t), axis=1) for t in terms_no_cross]) @ coef_no_cross
    mse_M0 = float(np.mean((pred_no_cross - y_te) ** 2))

    # M1: includes ns*Ap cross-terms.
    coef_full, _ = fit_polynomial(X_tr, y_tr, max_degree=4)
    pred_full = predict_polynomial(X_te, coef_full, _multinomial_terms(3, 4))
    mse_M1 = float(np.mean((pred_full - y_te) ** 2))

    return ExperimentResult(
        name="h6_covariance_combine",
        train_mse=0.0, test_mse=mse_M1, test_mse_near_fid=mse_M0,
        fisher_sigma={},
        extra={
            "mse_M0_no_cross_terms": mse_M0,
            "mse_M1_with_cross_terms": mse_M1,
            "improvement_factor": mse_M0 / max(mse_M1, 1e-30),
        },
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_all_experiments(seed: int = 0):
    return {
        "h1": experiment_h1_loss_function(seed=seed),
        "h2": experiment_h2_parsimony(seed=seed),
        "h3": experiment_h3_normalization(seed=seed),
        "h4": experiment_h4_operators(seed=seed),
        "h6": experiment_h6_covariance_combine(seed=seed),
    }
