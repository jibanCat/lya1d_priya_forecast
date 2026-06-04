# tests/test_multi_z_norm_io.py
import numpy as np
from priya_forecast.models.normalization import MultiZNormalizationSpec


def _spec():
    z_grid = np.array([3.4, 3.6])
    k_grid = np.linspace(0.005, 0.04, 6)
    mean = np.outer(1.0 + 0.1 * z_grid, 1.0 + k_grid)
    std = 0.2 * np.ones((2, 6))
    return MultiZNormalizationSpec(
        param_min=0.0, param_max=1.0, k_min=float(k_grid.min()),
        k_max=float(k_grid.max()), z_grid=z_grid, mean_flux=mean,
        std_flux=std, k_grid=k_grid)


def test_norm_npz_roundtrip(tmp_path):
    spec = _spec()
    path = tmp_path / "norm_ns.npz"
    spec.save_npz(path)
    back = MultiZNormalizationSpec.load_npz(path)
    np.testing.assert_allclose(back.mean_flux, spec.mean_flux)
    np.testing.assert_allclose(back.std_flux, spec.std_flux)
    np.testing.assert_allclose(back.z_grid, spec.z_grid)
    np.testing.assert_allclose(back.k_grid, spec.k_grid)
    assert (back.param_min, back.param_max) == (spec.param_min, spec.param_max)
    assert (back.k_min, back.k_max) == (spec.k_min, spec.k_max)
