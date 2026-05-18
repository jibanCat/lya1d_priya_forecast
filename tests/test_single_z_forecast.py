"""Unit tests for `priya_forecast.single_z.forecast`."""

from __future__ import annotations

import numpy as np
import pytest

from priya_forecast.single_z.forecast import per_param_local_norm


def test_per_param_local_norm_shapes_and_values():
    """Per-k mean/std of an LF flux sweep → a valid NormalizationSpec."""
    rng = np.random.default_rng(0)
    n_points, n_k = 50, 12
    k_grid = np.linspace(0.001, 0.04, n_k)
    flux_lf = rng.random((n_points, n_k)) + 1.0  # strictly positive
    norm = per_param_local_norm(
        flux_lf_z=flux_lf, k_grid=k_grid, param_min=0.8, param_max=1.05,
    )
    assert norm.mean_flux.shape == (n_k,)
    assert norm.std_flux.shape == (n_k,)
    assert np.all(norm.std_flux > 0)
    np.testing.assert_allclose(norm.mean_flux, flux_lf.mean(axis=0))
    np.testing.assert_allclose(norm.k_grid, k_grid)
    assert norm.param_min == 0.8
    assert norm.param_max == 1.05
    assert norm.k_min == pytest.approx(0.001)
    assert norm.k_max == pytest.approx(0.04)


def test_per_param_local_norm_degenerate_std_floored():
    """A k-bin with zero variance must not produce std=0 (NormalizationSpec rejects it)."""
    k_grid = np.linspace(0.001, 0.04, 5)
    flux_lf = np.ones((10, 5)) * 3.0  # zero variance everywhere
    norm = per_param_local_norm(
        flux_lf_z=flux_lf, k_grid=k_grid, param_min=0.0, param_max=1.0,
    )
    assert np.all(norm.std_flux > 0)
