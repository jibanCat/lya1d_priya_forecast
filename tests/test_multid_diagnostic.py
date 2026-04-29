"""Smoke + property tests for `priya_forecast.multid_diagnostic`."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from priya_forecast.models.base import P1DModel
from priya_forecast.multid_diagnostic import (
    DiagnosticResult,
    plot_diagnostic,
    run_diagnostic,
)
from priya_forecast.parameters import PARAM_NAMES, fiducial_vector


# ---------------------------------------------------------------------------
# Synthetic ground-truth model with controllable cross-coupling
# ---------------------------------------------------------------------------


class _SeparableTruth(P1DModel):
    """P_true(theta, k) = (1 + alpha_ns*ns_dev*k) * (1 + alpha_Ap*Ap_dev*k).

    Pure multiplicative-in-amplitude. 1D-product combine should give MSE
    ~ 0 (modulo polynomial fitting error) because the response is
    exactly separable.
    """

    def __init__(self, alpha_ns: float = 5.0, alpha_Ap: float = 0.5):
        self.alpha_ns = alpha_ns
        self.alpha_Ap = alpha_Ap

    def predict(self, theta, k, z):
        theta = np.asarray(theta, dtype=float)
        ns_dev = theta[PARAM_NAMES.index("ns")] - 0.983
        Ap_dev = theta[PARAM_NAMES.index("Ap")] - 1.46
        return (1.0 + self.alpha_ns * ns_dev * k) * (1.0 + self.alpha_Ap * Ap_dev * k)


class _NonSeparableTruth(P1DModel):
    """P_true(theta, k) = base + 5.0 * ns_dev * Ap_dev * k**2.

    Pure cross-coupling — the 2D-joint should easily beat 1D-product.
    """

    def predict(self, theta, k, z):
        theta = np.asarray(theta, dtype=float)
        ns_dev = theta[PARAM_NAMES.index("ns")] - 0.983
        Ap_dev = theta[PARAM_NAMES.index("Ap")] - 1.46
        return 1.0 + 5.0 * ns_dev * Ap_dev * k ** 2


# ---------------------------------------------------------------------------
# Unit tests on the regimes
# ---------------------------------------------------------------------------


def test_run_diagnostic_1d_returns_one_result_per_param():
    gp = _SeparableTruth()
    k = np.linspace(0.001, 0.02, 12)
    out = run_diagnostic(
        gp_model=gp, z=3.6, k_grid=k,
        param_names=["ns", "Ap"], regime="1D",
        n_train=64, n_test=128, poly_order=3, seed=0,
    )
    assert len(out) == 2
    for r in out:
        assert r.regime == "1D"
        assert r.n_params_varied == 1
        assert np.isfinite(r.train_mse) and np.isfinite(r.test_mse)
        assert r.test_mse >= 0


def test_run_diagnostic_2d_pairs_returns_choose_n_2_results():
    gp = _SeparableTruth()
    k = np.linspace(0.001, 0.02, 12)
    out = run_diagnostic(
        gp_model=gp, z=3.6, k_grid=k,
        param_names=["ns", "Ap"], regime="2D_pairs",
        n_train=64, n_test=128, poly_order=3, seed=0,
    )
    # C(2, 2) = 1 pair
    assert len(out) == 1
    r = out[0]
    assert r.regime == "2D_pairs"
    assert r.n_params_varied == 2
    assert "coupling" in r.extra
    assert "mse_1D_product" in r.extra
    assert "mse_2D_joint" in r.extra


def test_run_diagnostic_full_kd_returns_one_result():
    gp = _SeparableTruth()
    k = np.linspace(0.001, 0.02, 12)
    out = run_diagnostic(
        gp_model=gp, z=3.6, k_grid=k,
        param_names=["ns", "Ap"], regime="full_kD",
        n_train=64, n_test=128, poly_order=4, seed=0,
    )
    assert len(out) == 1
    assert out[0].regime == "full_kD"
    assert out[0].n_params_varied == 2


def test_run_diagnostic_rejects_unknown_regime():
    with pytest.raises(ValueError, match="Unknown regime"):
        run_diagnostic(
            gp_model=_SeparableTruth(), z=3.6, k_grid=np.linspace(0.001, 0.02, 5),
            param_names=["ns"], regime="invalid",
        )


# ---------------------------------------------------------------------------
# The headline science: coupling detection
# ---------------------------------------------------------------------------


def test_coupling_is_small_for_separable_truth():
    """For P = f(ns)*g(Ap), the 2D-joint should NOT do meaningfully
    better than the 1D-product, so coupling is small."""
    gp = _SeparableTruth(alpha_ns=2.0, alpha_Ap=0.3)
    k = np.linspace(0.001, 0.02, 12)
    out = run_diagnostic(
        gp_model=gp, z=3.6, k_grid=k,
        param_names=["ns", "Ap"], regime="2D_pairs",
        n_train=128, n_test=256, poly_order=4, seed=0,
    )
    coupling = out[0].extra["coupling"]
    # For separable truth, 1D-product is exact at infinite training data;
    # at finite data it can still beat 2D-joint due to noise. Coupling can
    # even be slightly negative. Allow up to +0.5.
    assert coupling < 0.5, f"Separable truth shouldn't have large coupling, got {coupling}"


def test_coupling_is_large_for_nonseparable_truth():
    """For P = base + alpha*ns*Ap*k², the cross-term means 1D-product
    leaves a residual that 2D-joint captures cleanly."""
    gp = _NonSeparableTruth()
    k = np.linspace(0.001, 0.02, 12)
    out = run_diagnostic(
        gp_model=gp, z=3.6, k_grid=k,
        param_names=["ns", "Ap"], regime="2D_pairs",
        n_train=128, n_test=256, poly_order=4, seed=0,
    )
    coupling = out[0].extra["coupling"]
    # 2D-joint should beat 1D-product by a large fraction. Demand > 0.5
    # (i.e. 2D-joint MSE is at most half the 1D-product MSE).
    assert coupling > 0.5, f"Non-separable truth must show coupling, got {coupling}"


def test_2d_joint_mse_beats_1d_product_on_nonseparable():
    """Direct check that mse_2D_joint < mse_1D_product on non-separable."""
    gp = _NonSeparableTruth()
    k = np.linspace(0.001, 0.02, 12)
    out = run_diagnostic(
        gp_model=gp, z=3.6, k_grid=k,
        param_names=["ns", "Ap"], regime="2D_pairs",
        n_train=128, n_test=256, poly_order=4, seed=0,
    )
    r = out[0]
    assert r.extra["mse_2D_joint"] < r.extra["mse_1D_product"]


# ---------------------------------------------------------------------------
# Plotting smoke test
# ---------------------------------------------------------------------------


def test_plot_diagnostic_writes_three_figures(tmp_path):
    gp = _NonSeparableTruth()
    k = np.linspace(0.001, 0.02, 12)
    pn = ["ns", "Ap", "hub"]
    results_by_regime = {
        "1D":       run_diagnostic(gp_model=gp, z=3.6, k_grid=k, param_names=pn, regime="1D",
                                   n_train=64, n_test=128, poly_order=3),
        "2D_pairs": run_diagnostic(gp_model=gp, z=3.6, k_grid=k, param_names=pn, regime="2D_pairs",
                                   n_train=64, n_test=128, poly_order=3),
        "full_kD":  run_diagnostic(gp_model=gp, z=3.6, k_grid=k, param_names=pn, regime="full_kD",
                                   n_train=64, n_test=128, poly_order=3),
    }
    out = plot_diagnostic(results_by_regime, outdir=tmp_path)
    assert (out / "diag1_scaling.png").exists()
    assert (out / "diag2_walltime.png").exists()
    assert (out / "diag3_coupling_matrix.png").exists()


# ---------------------------------------------------------------------------
# Property-based — hypothesis
# ---------------------------------------------------------------------------


@given(
    alpha_ns=st.floats(min_value=0.5, max_value=3.0, allow_nan=False),
    alpha_Ap=st.floats(min_value=0.1, max_value=1.0, allow_nan=False),
)
@settings(max_examples=5, deadline=None)
def test_property_coupling_bounded(alpha_ns: float, alpha_Ap: float):
    """For any reasonable separable amplitude combination, coupling is < 1
    (you can't reduce MSE by more than 100%)."""
    gp = _SeparableTruth(alpha_ns=alpha_ns, alpha_Ap=alpha_Ap)
    k = np.linspace(0.001, 0.02, 8)
    out = run_diagnostic(
        gp_model=gp, z=3.6, k_grid=k,
        param_names=["ns", "Ap"], regime="2D_pairs",
        n_train=64, n_test=128, poly_order=3, seed=0,
    )
    coupling = out[0].extra["coupling"]
    assert coupling < 1.0
