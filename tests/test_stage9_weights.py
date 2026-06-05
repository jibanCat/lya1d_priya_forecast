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
