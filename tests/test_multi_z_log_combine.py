# tests/test_multi_z_log_combine.py
import numpy as np
import pytest
from priya_forecast.parameters import PARAM_NAMES, fiducial_vector
from priya_forecast.refit_taylor import MultiZAdditiveTaylorModel


class _StubGP:
    """Deterministic positive GP: P_F(θ,k,z) = (1 + 0.1*Σθ) * base(k,z)."""
    def __init__(self, k_grid, z_grid):
        self.k_grid = np.asarray(k_grid, float)
        self.z_grid = np.asarray(z_grid, float)
    def predict(self, theta, k, z):
        k = np.asarray(k, float)
        base = 1.0 + 0.5 * k + 0.05 * float(z)
        return (1.0 + 0.1 * float(np.sum(theta))) * base


def _model(log_space):
    k_grid = np.linspace(0.005, 0.04, 8)
    z_grid = np.array([3.4, 3.6])
    fid = np.asarray(fiducial_vector(), float)
    gp = _StubGP(k_grid, z_grid)
    refits = {n: None for n in PARAM_NAMES}   # all GP-slice fallback
    return MultiZAdditiveTaylorModel(
        gp=gp, fid=fid, refits=refits, k_grid=k_grid, z_grid=z_grid,
        log_space=log_space,
    ), k_grid, fid


def test_log_space_predict_at_fid_equals_gp_per_z():
    m, k_grid, fid = _model(log_space=True)
    for z in (3.4, 3.6):
        got = m.predict(fid, k_grid, z)
        want = m.gp.predict(fid, k_grid, z)
        np.testing.assert_allclose(got, want, rtol=1e-12)


def test_log_and_linear_agree_at_fid():
    ml, k_grid, fid = _model(log_space=True)
    mlin, _, _ = _model(log_space=False)
    for z in (3.4, 3.6):
        np.testing.assert_allclose(
            ml.predict(fid, k_grid, z), mlin.predict(fid, k_grid, z), rtol=1e-12)


def test_log_space_gp_slice_off_fid():
    m, k_grid, fid = _model(log_space=True)
    theta_off = fid.copy()
    theta_off[0] = theta_off[0] * 1.01 if theta_off[0] != 0 else 0.01
    for z in (3.4, 3.6):
        got = m.predict(theta_off, k_grid, z)
        assert np.all(got > 0)
        # Use rtol=1e-8 so that the small (1%) perturbation on fid[0]=-0.009
        # registers as genuinely different from the fid GP output.
        assert not np.allclose(got, m.gp.predict(fid, k_grid, z), rtol=1e-8)


def test_log_space_positivity_guard():
    k_grid = np.linspace(0.005, 0.04, 8)
    z_grid = np.array([3.6])
    fid = np.asarray(fiducial_vector(), float)
    class _NegGP:
        def predict(self, theta, k, z):
            return -1.0 * np.ones_like(np.asarray(k, float))
    with pytest.raises(ValueError, match="non-positive"):
        MultiZAdditiveTaylorModel(
            gp=_NegGP(), fid=fid, refits={n: None for n in PARAM_NAMES},
            k_grid=k_grid, z_grid=z_grid, log_space=True,
        )
