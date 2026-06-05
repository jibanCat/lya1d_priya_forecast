# tests/test_stage9_weights.py
import numpy as np
from priya_forecast.sobolev_loss import sobolev_target_weights


class _GP:
    """logP = s*theta[idx] + base(k); ∂logP/∂theta = s (k-independent)."""
    def __init__(self, idx, s):
        self.idx, self.s = idx, s
    def predict(self, theta, k, z):
        k = np.asarray(k, float)
        return np.exp(self.s * float(theta[self.idx]) + 0.5 * k)  # P; logP linear in theta


def test_per_fidelity_normalized_gradient():
    nk = 3
    k = np.linspace(0.01, 0.04, nk)
    params = np.zeros((2, 11)); params[1, 0] = 1.0   # two theta values for idx 0
    payload = {
        "params_lf": params, "params_hf": params,
        "kfkms_lf_z": np.tile(k, (2, 1)), "kfkms_hf_z": np.tile(k, (2, 1)),
    }
    norm_std = np.full(nk, 2.0)
    w = sobolev_target_weights(
        payload=payload, param_idx=0, gp_lf=_GP(0, 0.2), gp_hf=_GP(0, 0.4),
        z=3.6, x_param_min=0.0, x_param_max=1.0, std_flux=norm_std, norm_k_grid=k, h=1e-3)
    n_lf = 2 * nk
    np.testing.assert_allclose(w[:n_lf], 0.1, rtol=1e-3)   # LF: 0.2 * 1.0 / 2.0
    np.testing.assert_allclose(w[n_lf:], 0.2, rtol=1e-3)   # HF: 0.4 * 1.0 / 2.0
    assert w.shape == (2 * n_lf,)


def test_point_major_ordering_within_fidelity():
    # Gradient depends on BOTH the param value and k, so each (point,k) cell is
    # distinct -> a permutation of rows would fail. Verifies point-major/k-minor.
    nk = 3
    k = np.linspace(0.01, 0.04, nk)
    params = np.zeros((2, 11)); params[0, 0] = 0.3; params[1, 0] = 0.9
    payload = {
        "params_lf": params, "params_hf": params,
        "kfkms_lf_z": np.tile(k, (2, 1)), "kfkms_hf_z": np.tile(k, (2, 1)),
    }
    norm_std = np.ones(nk)

    class _GPk:
        # logP = s*theta[0]*(1+k)  -> dlogP/dtheta = s*(1+k)  (varies with k, NOT theta)
        def __init__(self, s): self.s = s
        def predict(self, theta, kk, z):
            kk = np.asarray(kk, float)
            return np.exp(self.s * float(theta[0]) * (1.0 + kk))

    w = sobolev_target_weights(
        payload=payload, param_idx=0, gp_lf=_GPk(0.2), gp_hf=_GPk(0.2),
        z=3.6, x_param_min=0.0, x_param_max=1.0, std_flux=norm_std, norm_k_grid=k, h=1e-3)
    # expected LF gradient per (point,k): dlogP/dtheta = 0.2*(1+k), independent of point,
    # but the ROWS must be ordered point0's k-bins then point1's k-bins.
    expected_block = 0.2 * (1.0 + k)          # per k for either point
    n_lf = 2 * nk
    np.testing.assert_allclose(w[0:nk], expected_block, rtol=1e-3)        # LF point 0
    np.testing.assert_allclose(w[nk:n_lf], expected_block, rtol=1e-3)     # LF point 1
    # k varies WITHIN each point's block -> confirms k-minor ordering
    assert w[0] != w[1] != w[2]               # the three k-bins differ


def test_boundary_point_does_not_exceed_range():
    nk = 3
    k = np.linspace(0.01, 0.04, nk)
    # sweep includes a point AT the upper bound (1.0) and AT the lower bound (0.0)
    params = np.zeros((2, 11)); params[0, 0] = 0.0; params[1, 0] = 1.0

    class _BoundedGP:
        # asserts like the real emulator: theta[0] must be within [0, 1]
        def predict(self, theta, kk, z):
            t = float(theta[0])
            assert -1e-9 <= t <= 1.0 + 1e-9, f"theta[0]={t} out of [0,1]"
            kk = np.asarray(kk, float); return np.exp(0.2 * t + 0.5 * kk)

    payload = {"params_lf": params, "params_hf": params,
               "kfkms_lf_z": np.tile(k, (2, 1)), "kfkms_hf_z": np.tile(k, (2, 1))}
    # must NOT raise (clamping keeps perturbations within [x_param_min, x_param_max]=[0,1])
    w = sobolev_target_weights(
        payload=payload, param_idx=0, gp_lf=_BoundedGP(), gp_hf=_BoundedGP(),
        z=3.6, x_param_min=0.0, x_param_max=1.0, std_flux=np.ones(nk), norm_k_grid=k, h=1e-3)
    assert np.all(np.isfinite(w)) and w.shape == (2 * 2 * nk,)
