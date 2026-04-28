"""Unit + hypothesis tests for `priya_forecast.data`."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from priya_forecast.data import EBOSS_REDSHIFTS, bin_model_to_data, load_eboss


# ---------------------------------------------------------------------------
# Real DR14 data — unit tests
# ---------------------------------------------------------------------------


def test_eboss_redshift_grid_is_dr14():
    assert len(EBOSS_REDSHIFTS) == 13
    assert EBOSS_REDSHIFTS[0] == pytest.approx(2.2)
    assert EBOSS_REDSHIFTS[-1] == pytest.approx(4.6)
    diffs = np.diff(EBOSS_REDSHIFTS)
    assert np.allclose(diffs, 0.2)


def test_load_eboss_z36_shapes_and_finiteness():
    k, pf, cov = load_eboss(z=3.6)
    assert k.shape == (35,)
    assert pf.shape == (35,)
    assert cov.shape == (35, 35)
    assert np.all(np.isfinite(k))
    assert np.all(np.isfinite(pf))
    assert np.all(np.isfinite(cov))
    assert np.all(np.diff(k) > 0), "k must be strictly increasing"


def test_load_eboss_z36_covariance_is_symmetric_positive_definite():
    _, _, cov = load_eboss(z=3.6)
    assert np.allclose(cov, cov.T, atol=1e-12, rtol=0)
    eigs = np.linalg.eigvalsh(cov)
    assert np.all(eigs > 0), f"min eigenvalue {eigs.min()} not > 0"


def test_load_eboss_z36_diagonal_matches_published_variance():
    """Diagonal of the per-z covariance equals covar_diag in the bin range."""
    from priya_forecast._vendored.lyman_data import BOSSData

    boss = BOSSData()
    mask = np.abs(boss.redshifts - 3.6) < 0.01
    expected_diag = boss.covar_diag[mask]
    _, _, cov = load_eboss(z=3.6)
    # k was sorted; diagonal sorting must match
    order = np.argsort(boss.kf[mask])
    assert np.allclose(np.diag(cov), expected_diag[order], rtol=0, atol=1e-12)


def test_load_eboss_z36_pf_values_positive_finite():
    _, pf, _ = load_eboss(z=3.6)
    assert np.all(pf > 0), "P_F is positive in DR14"


def test_load_eboss_z_outside_grid_rejected():
    with pytest.raises(ValueError, match="not an eBOSS DR14 bin"):
        load_eboss(z=2.5)  # 0.1 from nearest bin


def test_load_eboss_all_13_bins_succeed():
    for z in EBOSS_REDSHIFTS:
        k, pf, cov = load_eboss(z=z)
        assert k.shape == pf.shape == (35,)
        assert cov.shape == (35, 35)
        assert np.all(np.linalg.eigvalsh(cov) > 0)


# ---------------------------------------------------------------------------
# bin_model_to_data — unit tests
# ---------------------------------------------------------------------------


def test_bin_constant_model_returns_constant():
    k_model = np.linspace(0.0005, 0.025, 1000)
    pf_model = np.full_like(k_model, 7.0)
    k_eboss, _, _ = load_eboss(z=3.6)
    out = bin_model_to_data(k_model, pf_model, k_eboss)
    assert out.shape == k_eboss.shape
    assert np.allclose(out, 7.0)


def test_bin_linear_model_recovers_centre_value():
    """For a linear model on a fine grid, top-hat avg ≈ value at bin centre."""
    k_eboss, _, _ = load_eboss(z=3.6)
    k_model = np.linspace(k_eboss[0] * 0.5, k_eboss[-1] * 1.5, 50000)
    a, b = 1.5, 30.0
    pf_model = a + b * k_model
    out = bin_model_to_data(k_model, pf_model, k_eboss)
    expected = a + b * k_eboss
    np.testing.assert_allclose(out, expected, rtol=1e-3)


def test_bin_rejects_non_monotone_model_grid():
    k_model = np.array([0.001, 0.002, 0.0015, 0.003])
    pf_model = np.ones_like(k_model)
    k_eboss = np.linspace(0.001, 0.02, 5)
    with pytest.raises(ValueError, match="strictly increasing"):
        bin_model_to_data(k_model, pf_model, k_eboss)


def test_bin_rejects_mismatched_shapes():
    k_model = np.array([0.001, 0.002, 0.003])
    pf_model = np.array([1.0, 2.0])
    k_eboss = np.linspace(0.001, 0.02, 5)
    with pytest.raises(ValueError, match="matching shape"):
        bin_model_to_data(k_model, pf_model, k_eboss)


def test_bin_falls_back_to_nearest_when_window_empty():
    """Sparse model grid with one bin having no samples in its window."""
    k_eboss = np.array([0.005, 0.010, 0.015])
    # Place two samples far away from the centre k_eboss[1] window
    k_model = np.array([0.001, 0.020])
    pf_model = np.array([10.0, 20.0])
    out = bin_model_to_data(k_model, pf_model, k_eboss)
    # Window for k_eboss[0]=0.005 spans 0.0025 .. 0.0075 → contains 0.001? no, 0.001 < 0.0025
    # so nearest fallback: nearest of {0.001, 0.020} to 0.005 is 0.001 → 10.0
    assert out[0] == 10.0
    # Centre window 0.0075 .. 0.0125 → no samples → nearest is 0.001 (0.0065 vs 0.0125) → 10.0
    assert out[1] == 10.0
    # Window for k_eboss[2]=0.015 spans 0.0125 .. 0.0175 → no samples → nearest 0.020 → 20.0
    assert out[2] == 20.0


# ---------------------------------------------------------------------------
# Property-based — hypothesis
# ---------------------------------------------------------------------------


@given(c=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
@settings(max_examples=20, deadline=None)
def test_property_constant_model_is_invariant_under_binning(c: float):
    k_model = np.linspace(1e-4, 0.05, 500)
    pf_model = np.full_like(k_model, c)
    k_eboss, _, _ = load_eboss(z=3.6)
    out = bin_model_to_data(k_model, pf_model, k_eboss)
    assert np.allclose(out, c)


@given(
    a=st.floats(min_value=-10, max_value=10, allow_nan=False),
    b=st.floats(min_value=-1000, max_value=1000, allow_nan=False),
)
@settings(max_examples=15, deadline=None)
def test_property_binning_is_linear_in_pf(a: float, b: float):
    """bin(a*pf + b) == a*bin(pf) + b. Linearity of top-hat averaging."""
    k_eboss, _, _ = load_eboss(z=3.6)
    k_model = np.linspace(1e-4, 0.05, 1000)
    pf_a = np.sin(50 * k_model) + 2.0
    out_combined = bin_model_to_data(k_model, a * pf_a + b, k_eboss)
    out_separate = a * bin_model_to_data(k_model, pf_a, k_eboss) + b
    np.testing.assert_allclose(out_combined, out_separate, rtol=1e-10, atol=1e-10)


@given(
    z=st.sampled_from(EBOSS_REDSHIFTS),
)
@settings(max_examples=13, deadline=None)
def test_property_every_z_bin_has_pos_def_cov(z: float):
    _, _, cov = load_eboss(z=z)
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    assert np.allclose(cov, cov.T)
