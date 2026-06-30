"""Dimension-balanced loss for PySR fits.

Standard MSE loss treats all input dimensions equally per-row but is
**dimension-blind** at the batch level: it doesn't know that "the
residual is correlated with x0" is bad. For weakly-coupled parameters
(omegamh2, hireionz, bhfeedback in our forecast), PySR's MSE optimizer
finds equations that match the (k, z, resolution) shape of the target
WITHOUT using x0 (theta), because the per-θ residual-variance is small
relative to the (k, z, res)-driven variance — so dropping x0 gains
parsimony without much MSE cost.

This module provides a **dim-balanced loss** that adds an explicit
per-dimension correlation penalty:

    L(prediction, target, X) = MSE + α · Σ_d corr²(residual, X_d)

where `residual = prediction - target` and `corr` is the Pearson
correlation over the batch. If the residual has no structure in any
input dimension, the penalty is 0 → standard MSE recovered. If
residual is correlated with x0 (PySR didn't use x0), the penalty is
positive → forces x0 dependence into the equation.

Reference (numpy) implementation here for unit testing; the production
loss is a Julia port passed to `PySRRegressor(loss_function=...)`. The
Python ref lets us test the math independently of PySR's Julia stack.
"""

from __future__ import annotations

import numpy as np

DEFAULT_ALPHA: float = 5.0
"""Default weight on the correlation-penalty term.

Empirically: α=5 makes per-dimension residual correlations contribute
~the same scale as the MSE for our flux_norm targets (typical residual
correlations 0.1-0.3 → squared 0.01-0.09 → ×5 → 0.05-0.45, comparable
to MSE ~0.5 for partial fits). Tune in tests if the loss is too soft
to force x0 or too aggressive (over-fits the residual structure).
"""

EPS: float = 1e-12
"""Numerical floor on variances to avoid 0/0 in the correlation."""


def _validate_inputs(prediction, target, X):
    prediction = np.asarray(prediction, dtype=float).ravel()
    target = np.asarray(target, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_rows, n_features); got {X.shape}.")
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} != target shape {target.shape}."
        )
    if X.shape[0] != prediction.shape[0]:
        raise ValueError(
            f"X rows {X.shape[0]} != prediction rows {prediction.shape[0]}."
        )
    return prediction, target, X


def dim_balanced_loss_corr(
    prediction: np.ndarray,
    target: np.ndarray,
    X: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """Correlation² version: `L = MSE + α · Σ_d corr²(residual, X[:, d])`.

    Catches **linear** residual-vs-feature dependence. Cheaper to compute
    than ANOVA but less expressive — a residual that depends on x_d
    nonlinearly (with mean ~0) will have corr ≈ 0 and slip through.
    Recommended only as a sanity check / cheap fallback; `dim_balanced_loss_anova`
    is the more expressive variant. Both are **optional ablation levers** — the
    production recipe trains the Sobolev gradient loss (value baseline = plain
    MSE); ANOVA is not the production default. See feedback_anova_loss_impact.
    """
    prediction, target, X = _validate_inputs(prediction, target, X)
    residual = prediction - target
    mse = float(np.mean(residual ** 2))
    r_centered = residual - residual.mean()
    var_r = float(np.mean(r_centered ** 2)) + EPS
    n_features = X.shape[1]
    penalty = 0.0
    for d in range(n_features):
        x_d = X[:, d]
        x_centered = x_d - x_d.mean()
        var_x = float(np.mean(x_centered ** 2)) + EPS
        cov_xr = float(np.mean(x_centered * r_centered))
        corr_sq = (cov_xr ** 2) / (var_x * var_r)
        penalty += corr_sq
    return mse + alpha * penalty


# Backwards-compat alias — older tests may import `dim_balanced_loss`.
dim_balanced_loss = dim_balanced_loss_corr


def _main_effect_squared(
    residual: np.ndarray, x_d: np.ndarray, n_bins: int = 10,
    normalize: bool = True,
) -> float:
    """Functional-ANOVA estimate of the **fraction of residual variance
    explained by the main effect of x_d** (when `normalize=True`,
    default), or the absolute squared L²-norm of the main effect (when
    `normalize=False`).

    For a residual r and input dimension x_d:

        ‖r_d‖² = Var( E[r | X_d=x_d] )                            (absolute)
        η²_d   = ‖r_d‖² / Var(r)                                  (normalized; ∈ [0, 1])

    The normalized form is unitless and directly comparable to a
    correlation² metric — "what fraction of residual variance is
    explained by binned-mean dependence on x_d?". For r = c · x_d
    (perfectly correlated): η² = 1. For r ⊥ x_d (independent): η² ≈ 0.

    Estimation: bin x_d into `n_bins` quantile bins, compute per-bin
    mean residual, sum  Σ_b weight(b) · (bin_mean − overall_mean)².
    """
    n = len(residual)
    if n_bins < 1 or n == 0:
        return 0.0
    r_overall_mean = float(residual.mean())
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x_d, quantiles)
    bin_idx = np.clip(
        np.digitize(x_d, edges[1:-1], right=False),
        0, n_bins - 1,
    )
    main_eff_sq = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        bin_mean = float(residual[mask].mean())
        weight = cnt / n
        main_eff_sq += weight * (bin_mean - r_overall_mean) ** 2
    if normalize:
        var_r = float(residual.var()) + EPS
        return main_eff_sq / var_r
    return main_eff_sq


def dim_balanced_loss_anova(
    prediction: np.ndarray,
    target: np.ndarray,
    X: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    n_bins: int = 10,
) -> float:
    """Functional ANOVA main-effect version (the more expressive dim-balanced
    loss; an optional ablation lever, NOT the production default).

    `L = MSE + α · Σ_d ‖r_d‖²`, where `r_d` is the main effect of x_d
    on the residual. Catches both linear and nonlinear residual
    structure in any input dimension; punishes equations that don't
    use x_d at all.
    """
    prediction, target, X = _validate_inputs(prediction, target, X)
    residual = prediction - target
    mse = float(np.mean(residual ** 2))
    n_features = X.shape[1]
    penalty = 0.0
    for d in range(n_features):
        penalty += _main_effect_squared(residual, X[:, d], n_bins=n_bins)
    return mse + alpha * penalty


# ---------------------------------------------------------------------------
# Julia-side loss_function source (passed verbatim to PySRRegressor).
# ---------------------------------------------------------------------------

JULIA_LOSS_FUNCTION_CORR = r"""
function loss_function(tree, dataset::Dataset{T,L}, options) where {T,L}
    prediction, complete = eval_tree_array(tree, dataset.X, options)
    if !complete || any(isnan, prediction) || any(isinf, prediction)
        return L(Inf)
    end
    n = length(prediction)
    residual = prediction .- dataset.y
    mse = sum(residual .^ 2) / n
    r_mean = sum(residual) / n
    r_centered = residual .- r_mean
    var_r = sum(r_centered .^ 2) / n + L(1e-12)
    pen = zero(L)
    n_features = size(dataset.X, 1)
    for d in 1:n_features
        x_d = view(dataset.X, d, :)
        x_mean = sum(x_d) / n
        x_centered = x_d .- x_mean
        var_x = sum(x_centered .^ 2) / n + L(1e-12)
        cov_xr = sum(x_centered .* r_centered) / n
        pen += (cov_xr ^ 2) / (var_x * var_r)
    end
    return mse + L(5.0) * pen
end
"""
"""Correlation² version (legacy / cheap fallback). See
`dim_balanced_loss_corr` (Python ref) for semantics. `JULIA_LOSS_FUNCTION_ANOVA`
is the more expressive variant — both are optional ablation levers; the
production recipe uses the Sobolev gradient loss (value baseline = plain MSE)."""


JULIA_LOSS_FUNCTION_ANOVA = r"""
function loss_function(tree, dataset::Dataset{T,L}, options) where {T,L}
    prediction, complete = eval_tree_array(tree, dataset.X, options)
    if !complete || any(isnan, prediction) || any(isinf, prediction)
        return L(Inf)
    end
    n = length(prediction)
    residual = prediction .- dataset.y
    mse = sum(residual .^ 2) / n
    r_overall_mean = sum(residual) / n
    # Variance of residual for normalization (η² metric).
    var_r = sum((residual .- r_overall_mean) .^ 2) / n + L(1e-12)
    n_features = size(dataset.X, 1)
    n_bins = 10
    pen = zero(L)
    for d in 1:n_features
        x_d = collect(view(dataset.X, d, :))
        sorted_x = sort(x_d)
        edges = Vector{T}(undef, n_bins - 1)
        for b in 1:(n_bins - 1)
            q = b / n_bins
            idx = clamp(round(Int, q * n), 1, n)
            edges[b] = sorted_x[idx]
        end
        bin_sum = zeros(L, n_bins)
        bin_cnt = zeros(Int, n_bins)
        for i in 1:n
            xi = x_d[i]
            b_idx = 1
            for e in edges
                if xi <= e
                    break
                end
                b_idx += 1
            end
            bin_sum[b_idx] += residual[i]
            bin_cnt[b_idx] += 1
        end
        main_eff_sq = zero(L)
        for b in 1:n_bins
            if bin_cnt[b] == 0
                continue
            end
            bin_mean = bin_sum[b] / bin_cnt[b]
            weight = L(bin_cnt[b]) / L(n)
            main_eff_sq += weight * (bin_mean - r_overall_mean) ^ 2
        end
        # η²_d = ‖r_d‖² / Var(r) — fraction of residual variance from
        # the main effect of x_d. Unitless, comparable to corr².
        pen += main_eff_sq / var_r
    end
    return mse + L(5.0) * pen
end
"""
"""Functional ANOVA main-effect version (RECOMMENDED). See
`dim_balanced_loss_anova` (Python ref) for semantics. Catches
nonlinear residual dependence on each input dimension; punishes
equations that drop any x_d entirely."""


# Default for production: ANOVA. Use JULIA_LOSS_FUNCTION_CORR for the
# legacy corr² semantics if needed.
JULIA_LOSS_FUNCTION = JULIA_LOSS_FUNCTION_ANOVA
