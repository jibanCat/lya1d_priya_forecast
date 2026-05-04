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


def dim_balanced_loss(
    prediction: np.ndarray,
    target: np.ndarray,
    X: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """Pure numpy reference of the dim-balanced loss.

    Parameters
    ----------
    prediction : ndarray, shape (n_rows,)
        Model prediction.
    target : ndarray, shape (n_rows,)
        Training target.
    X : ndarray, shape (n_rows, n_features)
        Input features. The penalty sums correlations over all columns.
    alpha : float
        Weight on the correlation-penalty term.

    Returns
    -------
    float
        Total loss = MSE + α · Σ_d corr²(residual, X[:, d]).
    """
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

    residual = prediction - target
    mse = float(np.mean(residual ** 2))

    # Pearson correlation of residual with each feature column.
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


# ---------------------------------------------------------------------------
# Julia-side loss_function source (passed verbatim to PySRRegressor).
# ---------------------------------------------------------------------------

JULIA_LOSS_FUNCTION = r"""
function loss_function(tree, dataset::Dataset{T,L}, options) where {T,L}
    prediction, complete = eval_tree_array(tree, dataset.X, options)
    if !complete || any(isnan, prediction) || any(isinf, prediction)
        return L(Inf)
    end
    n = length(prediction)
    residual = prediction .- dataset.y
    mse = sum(residual .^ 2) / n
    # Pearson-correlation² of residual with each feature column.
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
"""Julia source for the PySR `loss_function` argument.

Mirrors `dim_balanced_loss` (Python ref) with α=5.0 hardcoded. To
override α, edit the literal `L(5.0)` in the function body. This is
passed as a string to PySRRegressor(loss_function=JULIA_LOSS_FUNCTION).
"""
