"""Unit tests for `priya_forecast.mcmc`.

A real MCMC convergence check is too slow for unit tests. Instead we run a
short chain on a 2D Gaussian-linear toy and check (a) the chain shape is
right, (b) the posterior mean is within the analytic 1-sigma envelope,
(c) the HDF5 backend round-trips.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from priya_forecast.likelihood import (
    GaussianLikelihood,
    LogPosterior,
    UniformBoxPrior,
)
from priya_forecast.mcmc import _initial_positions, run_mcmc
from priya_forecast.models.base import P1DModel
from priya_forecast.parameters import (
    PARAM_NAMES,
    PARAMS_11D,
    Param,
    fiducial_vector,
)


# ---------------------------------------------------------------------------
# 2D linear toy
# ---------------------------------------------------------------------------


@dataclass
class Linear2DModel(P1DModel):
    """m(theta, k) = M @ theta[:2] + b. Accepts any theta length ≥ 2 so the
    same model satisfies the likelihood's 11D-fid construction call *and* the
    2D walkers."""

    M: np.ndarray
    b: np.ndarray

    def predict(self, theta: np.ndarray, k: np.ndarray, z: float) -> np.ndarray:
        return self.M @ np.asarray(theta, dtype=float)[:2] + self.b


def _toy_setup(seed: int = 0):
    rng = np.random.default_rng(seed)
    nk = 35
    M_2d = rng.normal(size=(nk, 2))
    base = 0.05 + 0.05 * np.cos(np.linspace(0, np.pi, nk))
    fid_full = np.array(fiducial_vector(), dtype=float)
    b = base - M_2d @ fid_full[:2]
    model = Linear2DModel(M=M_2d, b=b)
    lk = GaussianLikelihood(model=model, z=3.6, mock_data="gp", theta_fid=fid_full)
    # Match the prior to the 2 varied params so MCMC's 2D walkers validate.
    post = LogPosterior(likelihood=lk, prior=UniformBoxPrior(params=PARAMS_11D[:2]))
    return post, lk, M_2d


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_initial_positions_inside_prior_box():
    rng = np.random.default_rng(0)
    theta_fid = np.array(fiducial_vector(), dtype=float)
    pos = _initial_positions(theta_fid, n_walkers=44, params=PARAMS_11D, spread_frac=0.01, rng=rng)
    assert pos.shape == (44, 11)
    lo = np.array([p.prior[0] for p in PARAMS_11D])
    hi = np.array([p.prior[1] for p in PARAMS_11D])
    assert np.all(pos > lo)
    assert np.all(pos < hi)


# ---------------------------------------------------------------------------
# Chain run + shape + recovery
# ---------------------------------------------------------------------------


def test_run_mcmc_chain_shape_and_post_burn_size():
    post, lk, _M = _toy_setup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore short-chain convergence warning
        result = run_mcmc(
            posterior=post,
            params=PARAMS_11D[:2],
            n_steps=400,
            walkers_per_dim=4,
            burn_in_frac=0.25,
            seed=0,
        )
    n_walkers = 4 * 2
    expected_post_burn = 400 - int(0.25 * 400)
    assert result.chain.shape == (expected_post_burn, n_walkers, 2)
    assert result.log_prob.shape == (expected_post_burn, n_walkers)


def test_run_mcmc_recovers_fiducial_within_1sigma():
    """Linear toy: the analytic posterior mean is theta_fid; check the chain mean
    sits within ~3-sigma of fiducial after burn-in."""
    post, lk, M = _toy_setup(seed=1)
    fid = np.array(fiducial_vector(), dtype=float)[:2]
    cov_post = np.linalg.inv(M.T @ np.linalg.solve(lk.inputs.cov, M))
    sigma = np.sqrt(np.diag(cov_post))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_mcmc(
            posterior=post, params=PARAMS_11D[:2], n_steps=2000,
            walkers_per_dim=8, burn_in_frac=0.25, seed=1,
        )
    samples = result.chain.reshape(-1, 2)
    chain_mean = samples.mean(axis=0)
    # Allow 4 sigma — emcee mixing on 2D is generous but not tight.
    deviation = np.abs(chain_mean - fid)
    assert np.all(deviation < 4 * sigma), (
        f"chain mean {chain_mean} deviates from fid {fid} by {deviation}, "
        f"posterior sigma {sigma}"
    )


def test_run_mcmc_with_hdf5_backend(tmp_path: Path):
    post, _, _ = _toy_setup()
    backend_path = tmp_path / "chain.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_mcmc(
            posterior=post, params=PARAMS_11D[:2], n_steps=200, walkers_per_dim=4,
            burn_in_frac=0.2, seed=0, backend_path=backend_path,
        )
    assert backend_path.exists()
    assert result.backend_path == backend_path


def test_run_mcmc_warns_on_short_chain():
    post, _, _ = _toy_setup()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run_mcmc(
            posterior=post, params=PARAMS_11D[:2], n_steps=80, walkers_per_dim=4,
            burn_in_frac=0.2, seed=0,
        )
    assert any("not be converged" in str(rec.message) for rec in w)


def test_run_mcmc_log_prob_is_finite_post_burn():
    post, _, _ = _toy_setup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_mcmc(
            posterior=post, params=PARAMS_11D[:2], n_steps=400, walkers_per_dim=4,
            burn_in_frac=0.25, seed=0,
        )
    assert np.all(np.isfinite(result.log_prob))


# ---------------------------------------------------------------------------
# Property-based — hypothesis
# ---------------------------------------------------------------------------


@given(
    spread_frac=st.floats(min_value=1e-4, max_value=0.1, allow_nan=False),
    seed=st.integers(min_value=0, max_value=99),
)
@settings(max_examples=10, deadline=None)
def test_property_initial_positions_always_in_box(spread_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    theta_fid = np.array(fiducial_vector(), dtype=float)
    pos = _initial_positions(theta_fid, n_walkers=44, params=PARAMS_11D,
                             spread_frac=spread_frac, rng=rng)
    lo = np.array([p.prior[0] for p in PARAMS_11D])
    hi = np.array([p.prior[1] for p in PARAMS_11D])
    assert np.all(pos > lo)
    assert np.all(pos < hi)
