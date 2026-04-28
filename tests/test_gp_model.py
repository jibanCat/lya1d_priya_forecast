"""Unit + hypothesis tests for `priya_forecast.models.gp_model`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from priya_forecast.models import GPModel, MockGPModel
from priya_forecast.models.gp_model import DEFAULT_GP_BASEDIR
from priya_forecast.parameters import PARAM_NAMES, fiducial_vector, get_param


# ---------------------------------------------------------------------------
# MockGPModel
# ---------------------------------------------------------------------------


def _k_grid() -> np.ndarray:
    return np.linspace(0.001, 0.02, 35)


def test_mock_gp_returns_correct_shape_and_finite():
    gp = MockGPModel()
    out = gp.predict(np.array(fiducial_vector()), _k_grid(), z=3.6)
    assert out.shape == (35,)
    assert np.all(np.isfinite(out))
    assert np.all(out > 0)


def test_mock_gp_at_fiducial_is_smooth_decreasing():
    gp = MockGPModel()
    k = _k_grid()
    out = gp.predict(np.array(fiducial_vector()), k, z=3.6)
    # P_F is a power law × exp damping, so it should be monotone decreasing on the eBOSS k-range.
    assert np.all(np.diff(out) < 0)


def test_mock_gp_changes_with_ns():
    gp = MockGPModel()
    fid = np.array(fiducial_vector())
    bumped = fid.copy()
    bumped[PARAM_NAMES.index("ns")] = get_param("ns").prior[1]  # upper edge
    k = _k_grid()
    out_fid = gp.predict(fid, k, 3.6)
    out_bumped = gp.predict(bumped, k, 3.6)
    assert not np.allclose(out_fid, out_bumped), "ns must affect the prediction"


def test_mock_gp_z_scales_damping():
    gp = MockGPModel()
    fid = np.array(fiducial_vector())
    k = _k_grid()
    p_low = gp.predict(fid, k, z=2.6)
    p_high = gp.predict(fid, k, z=4.2)
    # Higher z → larger damping scale → more suppression at large k
    assert p_high[-1] / p_high[0] < p_low[-1] / p_low[0]


def test_mock_gp_rejects_bad_theta_shape():
    gp = MockGPModel()
    with pytest.raises(ValueError, match="theta"):
        gp.predict(np.zeros(10), _k_grid(), 3.6)


def test_mock_gp_rejects_zero_or_negative_k():
    gp = MockGPModel()
    with pytest.raises(ValueError, match="strictly positive"):
        gp.predict(np.array(fiducial_vector()), np.array([0.0, 0.005]), 3.6)


# ---------------------------------------------------------------------------
# Real GPModel — gated tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_gp_skip_if_unavailable():
    pytest.importorskip("GPy")
    pytest.importorskip("emukit")
    pytest.importorskip("lyaemu")
    if not DEFAULT_GP_BASEDIR.exists():
        pytest.skip(f"GP emulator basedir not present: {DEFAULT_GP_BASEDIR}")
    return DEFAULT_GP_BASEDIR


def test_real_gp_basedir_validation_without_loading_emulator():
    """Constructor should validate basedir + emulator_params.json without
    importing GPy."""
    bogus = Path("/nonexistent/path/does/not/exist")
    with pytest.raises(FileNotFoundError, match="basedir does not exist"):
        GPModel(basedir=bogus)


def test_real_gp_default_basedir_exists_or_skip():
    """Sanity check: the configured default exists on this machine."""
    if not DEFAULT_GP_BASEDIR.exists():
        pytest.skip(f"Default GP basedir not present: {DEFAULT_GP_BASEDIR}")
    # Should not raise.
    gp = GPModel(basedir=DEFAULT_GP_BASEDIR)
    assert gp.basedir == DEFAULT_GP_BASEDIR


def test_real_gp_a_template_validation():
    if not DEFAULT_GP_BASEDIR.exists():
        pytest.skip(f"Default GP basedir not present: {DEFAULT_GP_BASEDIR}")
    with pytest.raises(ValueError, match="a_template"):
        GPModel(basedir=DEFAULT_GP_BASEDIR, a_template=np.zeros(3))


def test_real_gp_predicts_at_fiducial(real_gp_skip_if_unavailable):
    """End-to-end smoke: real emulator returns a finite, positive P_F at fiducial."""
    gp = GPModel(basedir=real_gp_skip_if_unavailable)
    out = gp.predict(np.array(fiducial_vector()), _k_grid(), z=3.6)
    assert out.shape == (35,)
    assert np.all(np.isfinite(out))
    assert np.all(out > 0)


# ---------------------------------------------------------------------------
# Property-based — hypothesis on MockGP
# ---------------------------------------------------------------------------


@given(
    z=st.floats(min_value=2.6, max_value=4.2, allow_nan=False),
)
@settings(max_examples=15, deadline=None)
def test_property_mock_gp_positive_finite_across_z(z: float):
    gp = MockGPModel()
    out = gp.predict(np.array(fiducial_vector()), _k_grid(), z)
    assert np.all(out > 0)
    assert np.all(np.isfinite(out))


@given(
    perturbation=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False),
)
@settings(max_examples=10, deadline=None)
def test_property_mock_gp_smooth_under_param_perturbation(perturbation: float):
    """For any small Ap perturbation, P_F changes smoothly (no NaNs / infs)."""
    gp = MockGPModel()
    fid = np.array(fiducial_vector())
    fid[PARAM_NAMES.index("Ap")] *= 1.0 + perturbation
    out = gp.predict(fid, _k_grid(), 3.6)
    assert np.all(np.isfinite(out))
    assert np.all(out > 0)
