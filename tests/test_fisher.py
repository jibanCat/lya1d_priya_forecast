"""Unit + hypothesis tests for `priya_forecast.fisher`.

The cleanest correctness check is a Gaussian linear model: model = M·θ for
some constant matrix M, so dm/dθ_i is independent of θ. The Fisher matrix
is then F = M^T C^-1 M (closed form) and our 5-point stencil should agree
with that to full numerical precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from priya_forecast.fisher import FisherResult, fisher_matrix
from priya_forecast.likelihood import GaussianLikelihood
from priya_forecast.models import MockGPModel
from priya_forecast.models.base import P1DModel
from priya_forecast.parameters import (
    PARAM_NAMES,
    PARAMS_11D,
    Param,
    fiducial_vector,
)


# ---------------------------------------------------------------------------
# Linear-model harness with closed-form Fisher
# ---------------------------------------------------------------------------


@dataclass
class LinearGPMock(P1DModel):
    """Synthetic model: m(theta, k) = M(k) @ theta[:n] + b(k).

    Captures dm/dtheta_i = M[:, i] independent of theta — perfect for
    closed-form Fisher comparison. Slices theta to M's column count so the
    same model accepts the full-11D fiducial used at likelihood construction
    time *and* the n-dim sub-vector that Fisher varies.
    """

    M: np.ndarray  # (Nk, n)
    b: np.ndarray  # (Nk,)

    def predict(self, theta: np.ndarray, k: np.ndarray, z: float) -> np.ndarray:
        n = self.M.shape[1]
        return self.M @ np.asarray(theta, dtype=float)[:n] + self.b


def _make_linear_likelihood(n_dim: int, nk: int = 35, seed: int = 0):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(nk, n_dim))
    b = rng.normal(size=nk)
    model = LinearGPMock(M=M, b=b)
    # Build a sub-list of the 11D parameters for an n_dim-dim test.
    params = PARAMS_11D[:n_dim]
    theta_fid = np.array([p.fid for p in params], dtype=float)
    # Inject the fiducial baseline: redefine b so that m(theta_fid) is in the
    # eBOSS k-range scale (positive, modest).
    base = 0.05 + 0.05 * np.cos(np.linspace(0, np.pi, nk))
    b_shifted = base - M @ theta_fid
    model.b = b_shifted
    lk = GaussianLikelihood(
        model=model,
        z=3.6,
        mock_data="gp",
        theta_fid=np.concatenate([theta_fid, np.array(fiducial_vector()[n_dim:])]),
    )
    # The likelihood used the full 11D theta_fid for the data, but the linear
    # model only depends on the first n_dim entries. For the Fisher we only
    # vary those n_dim — that's what `params=` selects below.
    return lk, M, params, theta_fid


def test_fisher_recovers_closed_form_on_linear_model():
    """F_stencil should equal M^T C^-1 M up to stencil + Cholesky noise.

    The 5-point stencil is exact for a linear model in exact arithmetic,
    but floating-point differences m(theta+h) - m(theta-h) lose precision
    when m is O(0.1) and h*|M| is O(0.01) → ~1e-15 absolute, which after
    division by 12h and accumulation through the eBOSS-cov Cholesky solve
    surfaces at ~few×1e-5 in rtol on the smaller off-diagonal entries.
    rtol=1e-3 is plenty for catching real bugs here.
    """
    n = 4
    lk, M, params, theta_fid = _make_linear_likelihood(n_dim=n, seed=1)
    res = fisher_matrix(
        likelihood=lk,
        theta_fid=theta_fid,
        params=params,
        step_frac=0.01,
        rel_tol=0.001,
        max_halvings=2,
    )
    cov = lk.inputs.cov
    F_expected = M.T @ np.linalg.solve(cov, M)
    np.testing.assert_allclose(res.F, F_expected, rtol=1e-3, atol=1e-9)


def test_fisher_covariance_matches_F_inverse():
    n = 4
    lk, _, params, theta_fid = _make_linear_likelihood(n_dim=n)
    res = fisher_matrix(
        likelihood=lk,
        theta_fid=theta_fid,
        params=params,
        step_frac=0.01,
        rel_tol=0.001,
        max_halvings=1,
    )
    np.testing.assert_allclose(res.cov, np.linalg.inv(res.F), rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(res.sigma, np.sqrt(np.diag(res.cov)))


def test_fisher_correlation_matrix_diagonal_is_one():
    n = 4
    lk, _, params, theta_fid = _make_linear_likelihood(n_dim=n)
    res = fisher_matrix(
        likelihood=lk, theta_fid=theta_fid, params=params, step_frac=0.01,
        rel_tol=0.001, max_halvings=1,
    )
    np.testing.assert_allclose(np.diag(res.corr), 1.0, atol=1e-12)
    assert np.all(np.abs(res.corr) <= 1.0 + 1e-9)


def test_fisher_save_npz_round_trips(tmp_path: Path):
    n = 3
    lk, _, params, theta_fid = _make_linear_likelihood(n_dim=n)
    res = fisher_matrix(likelihood=lk, theta_fid=theta_fid, params=params, max_halvings=1)
    path = tmp_path / "fisher.npz"
    res.save_npz(path)
    loaded = np.load(path)
    np.testing.assert_allclose(loaded["F"], res.F)
    np.testing.assert_allclose(loaded["cov"], res.cov)
    assert tuple(loaded["param_names"]) == tuple(p.name for p in params)


def test_fisher_markdown_table_lists_all_params():
    n = 3
    lk, _, params, theta_fid = _make_linear_likelihood(n_dim=n)
    res = fisher_matrix(likelihood=lk, theta_fid=theta_fid, params=params, max_halvings=1)
    md = res.markdown_table()
    for p in params:
        assert p.name in md
    assert "Parameter" in md and "sigma" in md


def test_fisher_with_mock_gp_full_11d_raises_on_singular_F():
    """MockGP only constrains ns/Ap/hub/omegamh2 — the other 7 parameters
    contribute zero gradient, so the full-11D Fisher is genuinely singular
    and must raise rather than return garbage."""
    lk = GaussianLikelihood(model=MockGPModel(), z=3.6, mock_data="gp")
    with pytest.raises(ValueError, match="not invertible"):
        fisher_matrix(
            likelihood=lk, params=PARAMS_11D, step_frac=0.01, rel_tol=0.05, max_halvings=2,
        )


def test_fisher_with_mock_gp_on_constrained_subset_is_pos_def():
    """Subsetting to ns/Ap/hub/omegamh2 — the params MockGP actually
    constrains — produces a finite, well-conditioned Fisher."""
    lk = GaussianLikelihood(model=MockGPModel(), z=3.6, mock_data="gp")
    sub = tuple(p for p in PARAMS_11D if p.name in {"ns", "Ap", "hub", "omegamh2"})
    sub_idx = [PARAM_NAMES.index(p.name) for p in sub]
    fid_full = np.array(fiducial_vector())
    fid_sub = np.array([fid_full[i] for i in sub_idx])

    class _Proj:
        def predict(self, theta_sub, k, z):
            full = fid_full.copy()
            for i, idx in enumerate(sub_idx):
                full[idx] = theta_sub[i]
            return MockGPModel().predict(full, k, z)

    lk_sub = GaussianLikelihood(
        model=_Proj(), z=3.6, mock_data="gp", theta_fid=fid_sub,
    )
    res = fisher_matrix(
        likelihood=lk_sub, theta_fid=fid_sub, params=sub,
        step_frac=0.01, rel_tol=0.05, max_halvings=2,
    )
    assert res.F.shape == (4, 4)
    assert np.all(np.isfinite(res.F))
    assert np.all(np.isfinite(res.sigma))
    assert np.all(res.sigma > 0)


def test_fisher_step_halving_records_converged_steps():
    n = 4
    lk, _, params, theta_fid = _make_linear_likelihood(n_dim=n)
    res = fisher_matrix(
        likelihood=lk, theta_fid=theta_fid, params=params,
        step_frac=0.05, rel_tol=0.0001, max_halvings=4,
    )
    assert res.steps.shape == (n,)
    # Linear model → first stencil already exact; final step should be small.
    assert np.all(res.steps > 0)
    assert np.all(res.steps <= 0.05 * np.array([p.width() for p in params]))


# ---------------------------------------------------------------------------
# Property-based — hypothesis
# ---------------------------------------------------------------------------


@given(
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=8, deadline=None)
def test_property_linear_fisher_matches_closed_form_for_any_seed(seed: int):
    n = 3
    lk, M, params, theta_fid = _make_linear_likelihood(n_dim=n, seed=seed)
    res = fisher_matrix(
        likelihood=lk, theta_fid=theta_fid, params=params, max_halvings=1,
    )
    cov = lk.inputs.cov
    F_expected = M.T @ np.linalg.solve(cov, M)
    np.testing.assert_allclose(res.F, F_expected, rtol=1e-3, atol=1e-9)


@given(
    cov_scale=st.floats(min_value=0.5, max_value=4.0, allow_nan=False),
)
@settings(max_examples=8, deadline=None)
def test_property_fisher_scales_inversely_with_cov(cov_scale: float):
    """F(scale*C) = F(C) / scale (for linear model)."""
    n = 3
    lk_unit, M, params, theta_fid = _make_linear_likelihood(n_dim=n, seed=2)
    lk_scaled = GaussianLikelihood(
        model=lk_unit.model, z=lk_unit.z, mock_data="gp",
        cov_scale=cov_scale, theta_fid=lk_unit.theta_fid,
    )
    res_unit = fisher_matrix(likelihood=lk_unit, theta_fid=theta_fid, params=params, max_halvings=1)
    res_scaled = fisher_matrix(likelihood=lk_scaled, theta_fid=theta_fid, params=params, max_halvings=1)
    np.testing.assert_allclose(res_scaled.F, res_unit.F / cov_scale, rtol=1e-8, atol=1e-12)
