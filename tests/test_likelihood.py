"""Unit + hypothesis tests for `priya_forecast.likelihood`."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from priya_forecast.likelihood import (
    GaussianLikelihood,
    LogPosterior,
    UniformBoxPrior,
)
from priya_forecast.models import MockGPModel
from priya_forecast.parameters import (
    PARAM_NAMES,
    PARAMS_11D,
    fiducial_vector,
    get_param,
)


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------


def _likelihood(z: float = 3.6, mock_data: str = "gp", cov_scale: float = 1.0) -> GaussianLikelihood:
    return GaussianLikelihood(
        model=MockGPModel(),
        z=z,
        cov_scale=cov_scale,
        mock_data=mock_data,
    )


def test_likelihood_at_fiducial_is_max_when_mock_data_is_gp():
    """In mock_data='gp' mode, theta=theta_fid yields chi^2=0 → max log L."""
    lk = _likelihood()
    theta_fid = np.array(fiducial_vector(), dtype=float)
    log_at_fid = lk.log_likelihood(theta_fid)
    chi2_at_fid = lk.chi_squared(theta_fid)
    assert chi2_at_fid == pytest.approx(0.0, abs=1e-20)
    # Move slightly off fiducial: log L must drop.
    perturbed = theta_fid.copy()
    perturbed[PARAM_NAMES.index("ns")] += 0.01
    assert lk.log_likelihood(perturbed) < log_at_fid


def test_likelihood_chi2_matches_explicit_formula():
    """Sanity: chi2 = (d-m)^T C^-1 (d-m) recomputed directly via inv."""
    lk = _likelihood()
    theta_fid = np.array(fiducial_vector(), dtype=float)
    perturbed = theta_fid.copy()
    perturbed[PARAM_NAMES.index("Ap")] *= 1.1
    chi2 = lk.chi_squared(perturbed)
    r = lk.inputs.d - lk.model_at(perturbed)
    chi2_direct = float(r @ np.linalg.solve(lk.inputs.cov, r))
    assert chi2 == pytest.approx(chi2_direct, rel=1e-9)


def test_likelihood_log_prob_includes_normalization():
    """log L = log_norm - 0.5 * chi2; verify constant matches by construction."""
    lk = _likelihood()
    theta_fid = np.array(fiducial_vector(), dtype=float)
    log_at_fid = lk.log_likelihood(theta_fid)
    n = len(lk.inputs.d)
    log_det = np.linalg.slogdet(lk.inputs.cov)[1]
    expected_norm = -0.5 * (n * np.log(2 * np.pi) + log_det)
    assert log_at_fid == pytest.approx(expected_norm, rel=1e-9)


def test_likelihood_cov_scale_changes_chi2():
    """Doubling cov_scale must halve chi2 at the same theta."""
    lk_a = _likelihood(cov_scale=1.0)
    lk_b = _likelihood(cov_scale=2.0)
    perturbed = np.array(fiducial_vector(), dtype=float)
    perturbed[PARAM_NAMES.index("ns")] += 0.01
    assert lk_b.chi_squared(perturbed) == pytest.approx(
        0.5 * lk_a.chi_squared(perturbed), rel=1e-9
    )


def test_likelihood_rejects_nan_model():
    """If predict returns NaN we raise, never silently propagate."""
    # Build with a sound model on the eBOSS data, then swap in a broken model
    # so we test the per-call check (the ctor's fiducial check is separate).
    lk = GaussianLikelihood(model=MockGPModel(), z=3.6, mock_data="eboss")
    lk.model = type("X", (), {"predict": lambda self, t, k, z: np.full_like(k, np.nan)})()
    with pytest.raises(FloatingPointError, match="non-finite"):
        lk.log_likelihood(np.array(fiducial_vector()))


def test_likelihood_callable_alias():
    lk = _likelihood()
    theta = np.array(fiducial_vector())
    assert lk(theta) == pytest.approx(lk.log_likelihood(theta))


def test_likelihood_with_real_eboss_data():
    """mock_data='eboss' uses the real measurement; chi2 at the model-fid is non-zero."""
    lk = GaussianLikelihood(model=MockGPModel(), z=3.6, mock_data="eboss")
    chi2 = lk.chi_squared(np.array(fiducial_vector()))
    assert np.isfinite(chi2) and chi2 > 0


# ---------------------------------------------------------------------------
# UniformBoxPrior + LogPosterior
# ---------------------------------------------------------------------------


def test_uniform_box_prior_inside_returns_zero():
    pr = UniformBoxPrior()
    assert pr(np.array(fiducial_vector())) == 0.0


def test_uniform_box_prior_outside_returns_neg_inf():
    pr = UniformBoxPrior()
    theta = np.array(fiducial_vector(), dtype=float)
    theta[PARAM_NAMES.index("ns")] = get_param("ns").prior[1] + 0.1
    assert pr(theta) == -np.inf


def test_uniform_box_prior_rejects_wrong_shape():
    pr = UniformBoxPrior()
    with pytest.raises(ValueError, match="theta shape"):
        pr(np.zeros(5))


def test_log_posterior_outside_prior_short_circuits():
    lk = _likelihood()
    post = LogPosterior(likelihood=lk)
    theta = np.array(fiducial_vector(), dtype=float)
    theta[PARAM_NAMES.index("ns")] = -100.0
    assert post(theta) == -np.inf


def test_log_posterior_inside_prior_equals_likelihood():
    lk = _likelihood()
    post = LogPosterior(likelihood=lk)
    theta = np.array(fiducial_vector(), dtype=float)
    assert post(theta) == pytest.approx(lk(theta), rel=1e-12)


# ---------------------------------------------------------------------------
# Property-based — hypothesis
# ---------------------------------------------------------------------------


@given(
    delta_ns=st.floats(min_value=-0.1, max_value=0.1, allow_nan=False),
)
@settings(max_examples=15, deadline=None)
def test_property_log_likelihood_decreases_when_moving_off_fiducial(delta_ns: float):
    if abs(delta_ns) < 1e-6:
        return  # degenerate case
    lk = _likelihood()
    theta_fid = np.array(fiducial_vector(), dtype=float)
    perturbed = theta_fid.copy()
    perturbed[PARAM_NAMES.index("ns")] += delta_ns
    if (
        get_param("ns").prior[0] < perturbed[PARAM_NAMES.index("ns")] < get_param("ns").prior[1]
    ):
        assert lk(perturbed) < lk(theta_fid)


@given(
    cov_scale=st.floats(min_value=0.5, max_value=10.0, allow_nan=False),
    delta_ns=st.floats(min_value=-0.05, max_value=0.05, allow_nan=False),
)
@settings(max_examples=15, deadline=None)
def test_property_chi2_inversely_proportional_to_cov_scale(cov_scale: float, delta_ns: float):
    """For any (theta, scale, scale'), chi2(theta, scale*C) = chi2(theta, C) / scale."""
    if abs(delta_ns) < 1e-3:
        return
    lk_unit = _likelihood(cov_scale=1.0)
    lk_scaled = _likelihood(cov_scale=cov_scale)
    theta = np.array(fiducial_vector(), dtype=float)
    theta[PARAM_NAMES.index("ns")] += delta_ns
    np.testing.assert_allclose(
        lk_scaled.chi_squared(theta), lk_unit.chi_squared(theta) / cov_scale, rtol=1e-9
    )
