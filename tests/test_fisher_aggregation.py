"""Tests for combine_fisher_phys_arrays — multi-z Fisher aggregator.

Validates:
  - Per-z F arrays sum element-wise to the combined F.
  - Priors are applied ONCE (not per z).
  - Singular per-z F matrices combine to a non-singular total when
    different z bins have non-zero gradients on different params.
  - Output FisherResult fields are populated correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from priya_forecast.fisher import combine_fisher_phys_arrays
from priya_forecast.parameters import PARAMS_11D


def test_combine_sums_F_phys_across_z():
    """F_total = Σ_z F_z exactly when no priors are applied."""
    n = 4
    params = PARAMS_11D[:n]
    rng = np.random.default_rng(0)
    F1 = rng.normal(size=(n, n))
    F1 = F1 @ F1.T + np.eye(n)   # SPD
    F2 = rng.normal(size=(n, n))
    F2 = F2 @ F2.T + np.eye(n)
    theta_fid = np.array([p.fid for p in params])
    fr = combine_fisher_phys_arrays(
        [F1, F2], params=params, theta_fid=theta_fid, priors_sigma=None,
    )
    np.testing.assert_allclose(fr.F, F1 + F2, rtol=1e-12, atol=1e-12)


def test_priors_applied_once_after_summation():
    """Priors must NOT be added per z — should only count once."""
    n = 4
    params = PARAMS_11D[:n]
    F = np.eye(n) * 1e-3   # weak data → priors will dominate
    theta_fid = np.array([p.fid for p in params])

    # 3 z bins of the same F. Sum = 3·F.
    fr = combine_fisher_phys_arrays(
        [F, F, F], params=params, theta_fid=theta_fid,
        priors_sigma={params[0].name: 0.1},
    )
    # F_total = 3·F + diag(0,...,1/0.01,...,0) at param[0].
    expected_F = 3 * F.copy()
    expected_F[0, 0] += 1.0 / (0.1 ** 2)
    np.testing.assert_allclose(fr.F, expected_F, rtol=1e-12, atol=1e-12)

    # Sanity: applying ONE z bin with priors as a separate call gives
    # the SAME prior contribution per z (would have been added 3× if
    # we'd called `fisher_matrix(..., priors_sigma=...)` per z).
    fr_one = combine_fisher_phys_arrays(
        [F], params=params, theta_fid=theta_fid,
        priors_sigma={params[0].name: 0.1},
    )
    expected_F_one = F.copy()
    expected_F_one[0, 0] += 1.0 / (0.1 ** 2)
    np.testing.assert_allclose(fr_one.F, expected_F_one, rtol=1e-12, atol=1e-12)


def test_singular_per_z_combines_to_non_singular():
    """Per-z Fishers can each be rank-deficient; their sum can be full rank.

    Constructed so that F_z1 has zero gradient on param[1] and F_z2
    has zero gradient on param[0] → each is rank 1 in the (0, 1) block,
    but F_total has rank 2.
    """
    n = 2
    params = PARAMS_11D[:n]
    F1 = np.array([[1.0, 0.0], [0.0, 0.0]])  # constraint on param 0 only
    F2 = np.array([[0.0, 0.0], [0.0, 1.0]])  # constraint on param 1 only
    theta_fid = np.array([p.fid for p in params])
    fr = combine_fisher_phys_arrays(
        [F1, F2], params=params, theta_fid=theta_fid, priors_sigma=None,
    )
    # F_total = identity in (0, 1) block → cov diagonal in dimensionless
    # coords; sigma_phys = width / sqrt(F_hat_diag).
    # The math:
    #   F_phys = I, W = outer(widths, widths), F_hat = W (since F_phys = I)
    #   cov_hat = W^{-1}, cov = W^{-1} ⊙ W = identity (NO — cov = cov_hat ⊙ W = identity)
    #   Hmm let me just check sigma is finite, not NaN/inf.
    assert np.all(np.isfinite(fr.sigma)), f"σ should be finite: {fr.sigma}"
    assert fr.sigma[0] > 0 and fr.sigma[1] > 0


def test_unknown_prior_param_raises():
    n = 3
    params = PARAMS_11D[:n]
    F = np.eye(n)
    theta_fid = np.array([p.fid for p in params])
    with pytest.raises(KeyError, match="prior on unknown param"):
        combine_fisher_phys_arrays(
            [F], params=params, theta_fid=theta_fid,
            priors_sigma={"not_a_param": 0.1},
        )


def test_empty_input_raises():
    n = 2
    with pytest.raises(ValueError):
        combine_fisher_phys_arrays(
            [], params=PARAMS_11D[:n],
            theta_fid=np.array([0.0, 0.0]), priors_sigma=None,
        )


def test_shape_mismatch_raises():
    n = 3
    params = PARAMS_11D[:n]
    F_good = np.eye(n)
    F_bad = np.eye(n + 1)
    theta_fid = np.array([p.fid for p in params])
    with pytest.raises(ValueError, match="F_phys shape"):
        combine_fisher_phys_arrays(
            [F_good, F_bad], params=params,
            theta_fid=theta_fid, priors_sigma=None,
        )


def test_param_names_and_theta_fid_propagate():
    n = 3
    params = PARAMS_11D[:n]
    F = np.eye(n) * 100.0
    theta_fid = np.array([p.fid for p in params])
    fr = combine_fisher_phys_arrays(
        [F], params=params, theta_fid=theta_fid, priors_sigma=None,
    )
    assert fr.param_names == tuple(p.name for p in params)
    np.testing.assert_array_equal(fr.theta_fid, theta_fid)
