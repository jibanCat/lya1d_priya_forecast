"""Unit tests for the dimension-balanced loss.

Tests the Python reference (`dim_balanced_loss`) on synthetic data
where we know the optimal solution. The Julia version is a verbatim
port of this Python ref — if the Python tests pass, the Julia version
is mathematically correct (modulo translation bugs caught by an
end-to-end smoke test in test_pysr_dim_balanced_smoke.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from priya_forecast.dim_balanced_loss import (
    DEFAULT_ALPHA,
    EPS,
    _main_effect_squared,
    dim_balanced_loss,             # legacy alias = corr² version
    dim_balanced_loss_anova,
    dim_balanced_loss_corr,
)


def test_perfect_prediction_gives_zero_loss():
    """f == y → residual = 0 → MSE = 0 AND correlation = 0/0 → loss ≈ 0."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = X @ np.array([1.0, 2.0, 3.0])
    pred = y.copy()
    loss = dim_balanced_loss(pred, y, X, alpha=DEFAULT_ALPHA)
    # MSE = 0, residual = 0 → var_r = EPS → corr² = 0 / (var_x · EPS)
    # which is ill-conditioned but ~0 numerically. Loss ≈ 0.
    assert abs(loss) < 1e-8, f"perfect-pred loss = {loss}, expected ~0"


def test_constant_prediction_uncorrelated_target_gives_pure_mse():
    """Penalty term scales as O(D/n) for an uncorrelated residual.

    Finite-sample empirical correlation² between independent residual
    and X column is O(1/n) per column. With D columns and weight α,
    the penalty floor is α·D/n. For n=20000, D=3, α=5: floor ≈ 7.5e-4
    — small enough that loss ≈ MSE within reasonable tolerance.
    """
    rng = np.random.default_rng(1)
    n = 20000
    X = rng.normal(size=(n, 3))
    y = rng.normal(scale=0.1, size=n)
    pred = np.zeros(n)
    loss = dim_balanced_loss(pred, y, X, alpha=DEFAULT_ALPHA)
    expected_mse = float(np.mean((pred - y) ** 2))
    # Penalty floor for D=3, n=20000, α=5: ~7.5e-4. Allow 5× headroom.
    floor = DEFAULT_ALPHA * X.shape[1] / n
    assert loss - expected_mse < 5 * floor, (
        f"loss={loss}, MSE={expected_mse}, excess={loss - expected_mse}, "
        f"floor={floor}"
    )


def test_constant_prediction_on_x0_dependent_target_penalizes_correlation():
    """Constant prediction on `y = x0`: residual = -x0, fully correlated with x0.

    Standard MSE = Var(x0). Dim-balanced loss adds α · 1 (corr² = 1).
    """
    rng = np.random.default_rng(2)
    n = 500
    X = rng.normal(size=(n, 4))
    y = X[:, 0]  # only x0 matters
    pred = np.zeros(n)
    loss = dim_balanced_loss(pred, y, X, alpha=DEFAULT_ALPHA)
    mse = float(np.mean(y ** 2))   # = Var(x0) ≈ 1
    # Residual = -x0, perfectly anti-correlated with x0 → corr² = 1.
    # Other columns x1..x3 are independent of x0, so corr² ≈ 0.
    expected = mse + DEFAULT_ALPHA * 1.0
    assert abs(loss - expected) < 0.05 * expected, (
        f"loss={loss} vs expected={expected} (MSE={mse} + α·1)"
    )


def test_x0_using_prediction_gets_lower_loss_than_constant_prediction():
    """The whole point: the loss must rank `f = x0` above `f = const` on
    a target that depends only on x0 — not just match it (standard MSE
    does that already). Specifically, for a 4-input problem with
    `y = 0.1·x0` (weak coupling), a constant prediction has only
    slightly worse MSE than the optimal `f = 0.1·x0`, but the
    correlation penalty makes the dim-balanced loss MUCH worse for the
    constant prediction → PySR will prefer the x0-using equation.
    """
    rng = np.random.default_rng(3)
    n = 500
    X = rng.normal(size=(n, 4))
    coef = 0.1   # WEAK coupling to x0
    y = coef * X[:, 0]

    # f1: optimal x0-using prediction
    pred_x0 = coef * X[:, 0]
    loss_x0 = dim_balanced_loss(pred_x0, y, X, alpha=DEFAULT_ALPHA)

    # f2: constant prediction (zero, mean of y)
    pred_const = np.zeros(n)
    loss_const = dim_balanced_loss(pred_const, y, X, alpha=DEFAULT_ALPHA)

    # Standard MSE: pred_const has MSE = coef² ≈ 0.01; pred_x0 has 0.
    mse_const = float(np.mean((pred_const - y) ** 2))
    mse_x0 = float(np.mean((pred_x0 - y) ** 2))
    assert mse_x0 < mse_const, "sanity: optimal pred has lower MSE"

    # Dim-balanced loss: pred_const adds α·1 ≈ 5 because corr(residual, x0) = -1.
    # pred_x0 adds 0. The gap is HUGE compared to standard MSE.
    gap_mse = mse_const - mse_x0
    gap_balanced = loss_const - loss_x0
    assert gap_balanced > 100 * gap_mse, (
        f"dim-balanced should amplify the gap; "
        f"MSE-gap={gap_mse}, balanced-gap={gap_balanced}, ratio={gap_balanced/(gap_mse+1e-12):.1f}×"
    )


def test_penalty_zero_when_residual_uncorrelated_with_features():
    """If residual is pure noise (uncorrelated with X), penalty term = 0."""
    rng = np.random.default_rng(4)
    n = 5000  # large for tight Pearson estimate
    X = rng.normal(size=(n, 3))
    y = X @ np.array([1.0, 2.0, 3.0])
    pred = y + rng.normal(scale=0.5, size=n)  # residual = pure Gaussian noise
    loss = dim_balanced_loss(pred, y, X, alpha=DEFAULT_ALPHA)
    expected_mse = float(np.mean((pred - y) ** 2))
    assert abs(loss - expected_mse) < 0.05 * expected_mse, (
        f"loss={loss} vs expected_mse={expected_mse}; "
        f"residual is uncorrelated with X so penalty should be ≈0"
    )


def test_zero_alpha_reduces_to_mse():
    """α=0 → loss = MSE exactly."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(100, 4))
    y = X[:, 1]
    pred = np.zeros(100)
    loss = dim_balanced_loss(pred, y, X, alpha=0.0)
    expected_mse = float(np.mean(y ** 2))
    assert abs(loss - expected_mse) < 1e-8


# ---------------------------------------------------------------------------
# ANOVA-version tests
# ---------------------------------------------------------------------------


def test_anova_main_effect_zero_when_residual_is_iid_noise():
    """Residual independent of x_d → η² ≈ 0 (down to bin-level noise)."""
    rng = np.random.default_rng(0)
    n = 5000
    residual = rng.normal(size=n)
    x_d = rng.normal(size=n)
    me = _main_effect_squared(residual, x_d, n_bins=10)
    # With n=5000, η² floor ≈ n_bins / n (finite-sample bin-mean noise).
    # 10/5000 = 0.002. Allow generous tolerance.
    assert me < 0.02, f"iid residual η² should be ≈ 0; got {me}"


def test_anova_main_effect_recovers_linear_dependence():
    """Residual = c · x_d → η²_d ≈ 1 (full residual variance is main effect)."""
    rng = np.random.default_rng(1)
    n = 10000
    x_d = rng.uniform(-1, 1, size=n)
    residual = 0.3 * x_d
    me = _main_effect_squared(residual, x_d, n_bins=10)
    # Normalized: ‖r_d‖² / Var(r) ≈ 1 for r = c·x_d.
    assert 0.85 < me < 1.05, f"η² should be ≈ 1; got {me}"


def test_anova_main_effect_recovers_nonlinear_dependence():
    """Residual = (x_d - 0.5)² → η² ≈ 1 (mean-of-bin captures the parabola)."""
    rng = np.random.default_rng(2)
    n = 10000
    x_d = rng.uniform(0, 1, size=n)
    residual = (x_d - 0.5) ** 2 - np.mean((x_d - 0.5) ** 2)
    me = _main_effect_squared(residual, x_d, n_bins=10)
    # Normalized η²: most of residual variance comes from the parabolic
    # main effect of x_d on r. Should be close to 1.
    assert me > 0.85, (
        f"nonlinear residual should give η² > 0.85; got {me}. "
        "(corr² would give ~0 here — this is the ANOVA advantage.)"
    )


def test_anova_perfect_prediction_gives_zero_loss():
    """f == y → residual = 0 → MSE = 0 AND main effects = 0 → loss = 0."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([1.0, 2.0, 3.0])
    pred = y.copy()
    loss = dim_balanced_loss_anova(pred, y, X, alpha=DEFAULT_ALPHA)
    assert abs(loss) < 1e-8, f"perfect-pred ANOVA loss = {loss}"


def test_anova_constant_pred_on_x0_dependent_target_strongly_penalized():
    """Constant pred on `y = x0` → main_effect_x0 = Var(y) (entire signal).
    Loss = MSE + α · Var(y) → much larger than plain MSE."""
    rng = np.random.default_rng(4)
    n = 5000
    X = rng.uniform(0, 1, size=(n, 4))
    y = X[:, 0]
    pred = np.full(n, y.mean())   # constant prediction = best no-x0 fit
    mse = float(np.mean((pred - y) ** 2))
    loss_anova = dim_balanced_loss_anova(pred, y, X, alpha=DEFAULT_ALPHA)
    loss_corr = dim_balanced_loss_corr(pred, y, X, alpha=DEFAULT_ALPHA)
    # ANOVA: penalty ≈ α · Var(y) = α · 1/12 ≈ 0.42.
    # Both versions detect this because residual is linearly correlated with x0.
    # The ANOVA version also catches the same effect via per-bin means.
    assert loss_anova > 5 * mse, (
        f"ANOVA loss should be ≫ MSE for constant pred on x0-target; "
        f"loss={loss_anova}, mse={mse}"
    )
    # On a LINEAR target, ANOVA and corr² agree to within bin noise.
    np.testing.assert_allclose(loss_anova, loss_corr, rtol=0.2)


def test_anova_beats_corr_on_nonlinear_residual():
    """Constant pred on `y = (x0 - 0.5)²`. corr(residual, x0) ≈ 0 (centered
    quadratic) → corr² penalty ≈ 0 → loss = MSE only.
    ANOVA's binned mean-residuals DO detect the nonlinear structure →
    loss > MSE.
    """
    rng = np.random.default_rng(5)
    n = 5000
    X = rng.uniform(0, 1, size=(n, 4))
    y = (X[:, 0] - 0.5) ** 2
    pred = np.full(n, y.mean())
    mse = float(np.mean((pred - y) ** 2))
    loss_corr = dim_balanced_loss_corr(pred, y, X, alpha=DEFAULT_ALPHA)
    loss_anova = dim_balanced_loss_anova(pred, y, X, alpha=DEFAULT_ALPHA)
    # corr² should miss the symmetric quadratic — loss ≈ MSE.
    assert abs(loss_corr - mse) < 0.1 * mse + 0.01, (
        f"corr² should miss quadratic; loss_corr={loss_corr}, mse={mse}"
    )
    # ANOVA should catch it.
    assert loss_anova > 1.5 * mse, (
        f"ANOVA should catch nonlinear; loss_anova={loss_anova}, mse={mse}"
    )


def test_anova_zero_alpha_reduces_to_mse():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(200, 4))
    y = X[:, 1]
    pred = np.zeros(200)
    loss = dim_balanced_loss_anova(pred, y, X, alpha=0.0)
    expected = float(np.mean(y ** 2))
    assert abs(loss - expected) < 1e-8


def test_handles_wrong_input_shapes():
    X = np.zeros((10, 3))
    with pytest.raises(ValueError, match="2D"):
        dim_balanced_loss(np.zeros(10), np.zeros(10), np.zeros(10), alpha=1.0)
    with pytest.raises(ValueError, match="prediction shape"):
        dim_balanced_loss(np.zeros(5), np.zeros(10), X, alpha=1.0)
    with pytest.raises(ValueError, match="X rows"):
        dim_balanced_loss(np.zeros(10), np.zeros(10), np.zeros((5, 3)), alpha=1.0)
