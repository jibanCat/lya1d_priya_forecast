# tests/test_multi_z_combine.py
import numpy as np
import pytest
from priya_forecast.parameters import PARAM_NAMES, fiducial_vector
from priya_forecast.multi_z.combine import build_combined_model_multiz


class _StubGP:
    def predict(self, theta, k, z):
        k = np.asarray(k, float)
        return (1.0 + 0.1 * float(np.sum(theta))) * (1.0 + 0.5 * k + 0.05 * float(z))


def test_builds_and_predicts_per_z():
    k_grid = np.linspace(0.005, 0.04, 8)
    z_grid = np.array([3.4, 3.6])
    fid = np.asarray(fiducial_vector(), float)
    m = build_combined_model_multiz(
        combine_mode="additive", gp=_StubGP(), fid=fid,
        refits={n: None for n in PARAM_NAMES}, k_grid=k_grid, z_grid=z_grid,
        log_space=False)
    out = m.predict(fid, k_grid, 3.6)
    assert out.shape == k_grid.shape


def test_rejects_unimplemented_combine():
    with pytest.raises(NotImplementedError):
        build_combined_model_multiz(
            combine_mode="multiplicative", gp=_StubGP(),
            fid=np.asarray(fiducial_vector(), float),
            refits={n: None for n in PARAM_NAMES},
            k_grid=np.linspace(0.005, 0.04, 8), z_grid=np.array([3.6]))
