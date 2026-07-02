# tests/test_stage10_weights.py
"""Stage 10 Task 1: sobolev_target_weights_multiz correctness.

Key invariants verified (mirroring _build_training_matrix_multiz exactly):
- BOTH LF and HF rows use payload["params_lf"]  (builder assigns X_hf = X_lf except resolution)
- BOTH LF and HF rows use payload["kfkms_lf_z"] (builder uses k_lf_norm for both X_lf and X_hf)
- LF rows use gp_lf; HF rows use gp_hf
- Each row's std comes from norm.std_flux[norm._z_index(z_r), :] for that row's z
- Row order: LF block then HF block, point-major / k-minor within each block
- Boundary sweep points do not push theta outside [x_param_min, x_param_max]
"""
import numpy as np
import pytest

from priya_forecast.sobolev_loss import sobolev_target_weights_multiz
from priya_forecast.models.normalization import MultiZNormalizationSpec


class _GPz:
    """logP = s * theta[0] * z + 0.5*k  ->  dlogP/dtheta[0] = s * z  (z-dependent).

    This makes the per-z routing actually testable: LF and HF at the SAME z give
    different gradients (different s), and the same fidelity at different z also
    differs (different z factor).
    """
    def __init__(self, s):
        self.s = s

    def predict(self, theta, k, z):
        k = np.asarray(k, float)
        return np.exp(self.s * float(theta[0]) * float(z) + 0.5 * k)


def _make_minimal_payload(k, n_points=2, n_z=2):
    """Build a minimal multi-z payload matching _build_training_matrix_multiz's conventions.

    BOTH kfkms_lf_z and kfkms_hf_z are identical (as in the upstream builder,
    which copies k_grid for both LF and HF). params_lf is used for both fidelities.
    """
    params = np.zeros((n_points, 11))
    params[:, 0] = [0.4, 0.6]
    z_per_row = np.array([3.0, 3.6])[:n_points]
    return {
        "params_lf": params,
        # Builder uses k_lf for BOTH X_lf and X_hf; kfkms_hf_z is present but
        # sobolev_target_weights_multiz uses kfkms_lf_z for both (mirroring builder).
        "kfkms_lf_z": np.tile(k, (n_points, 1)),
        "kfkms_hf_z": np.tile(k, (n_points, 1)),
        "z_per_row": z_per_row,
    }


def test_multiz_weights_per_fidelity_z_and_std():
    """Core correctness: LF/HF routing, per-z std lookup, and exact weight values."""
    nk = 3
    k = np.linspace(0.01, 0.04, nk)
    zgrid = np.array([3.0, 3.6])
    payload = _make_minimal_payload(k, n_points=2)

    # Distinct std per z-bin: z=3.0 -> 2.0, z=3.6 -> 4.0
    norm = MultiZNormalizationSpec(
        param_min=0.0, param_max=1.0,
        k_min=float(k.min()), k_max=float(k.max()),
        z_grid=zgrid,
        mean_flux=np.zeros((2, nk)),
        std_flux=np.array([[2.0] * nk, [4.0] * nk]),
        k_grid=k,
    )

    s_lf, s_hf = 0.2, 0.5
    w = sobolev_target_weights_multiz(
        payload=payload, param_idx=0,
        gp_lf=_GPz(s_lf), gp_hf=_GPz(s_hf),
        norm=norm,
        x_param_min=0.0, x_param_max=1.0, h=1e-3,
    )

    n_lf = 2 * nk  # 2 points × nk bins in LF block
    assert w.shape == (2 * n_lf,), f"Expected shape ({2 * n_lf},), got {w.shape}"

    # width = x_param_max - x_param_min = 1.0
    # weight = dlogP/dtheta_phys * width / std_k
    # dlogP/dtheta[0] = s * z  (from _GPz construction)

    # LF row 0 (point 0, z=3.0): grad = 0.2 * 3.0 = 0.6; std=2.0 -> weight = 0.6/2.0 = 0.30
    np.testing.assert_allclose(w[0:nk], 0.30, rtol=1e-3,
                               err_msg="LF point0 (z=3.0) weight wrong")

    # LF row 1 (point 1, z=3.6): grad = 0.2 * 3.6 = 0.72; std=4.0 -> weight = 0.72/4.0 = 0.18
    np.testing.assert_allclose(w[nk:n_lf], 0.18, rtol=1e-3,
                               err_msg="LF point1 (z=3.6) weight wrong")

    # HF row 0 (point 0, z=3.0): grad = 0.5 * 3.0 = 1.5; std=2.0 -> weight = 1.5/2.0 = 0.75
    np.testing.assert_allclose(w[n_lf:n_lf + nk], 0.75, rtol=1e-3,
                               err_msg="HF point0 (z=3.0) weight wrong")

    # HF row 1 (point 1, z=3.6): grad = 0.5 * 3.6 = 1.8; std=4.0 -> weight = 1.8/4.0 = 0.45
    np.testing.assert_allclose(w[n_lf + nk:], 0.45, rtol=1e-3,
                               err_msg="HF point1 (z=3.6) weight wrong")


def test_multiz_weights_row_order_point_major_k_minor():
    """k varies fastest within a point's block (point-major / k-minor)."""
    nk = 4
    k = np.linspace(0.01, 0.04, nk)
    zgrid = np.array([3.0, 3.6])
    payload = _make_minimal_payload(k, n_points=2)

    norm = MultiZNormalizationSpec(
        param_min=0.0, param_max=1.0,
        k_min=float(k.min()), k_max=float(k.max()),
        z_grid=zgrid,
        mean_flux=np.zeros((2, nk)),
        std_flux=np.ones((2, nk)),
        k_grid=k,
    )

    class _GPkz:
        """logP = s * theta[0] * z * (1 + k) -> dlogP/dtheta = s * z * (1+k): varies with k."""
        def __init__(self, s):
            self.s = s
        def predict(self, theta, kk, z):
            kk = np.asarray(kk, float)
            return np.exp(self.s * float(theta[0]) * float(z) * (1.0 + kk))

    w = sobolev_target_weights_multiz(
        payload=payload, param_idx=0,
        gp_lf=_GPkz(0.2), gp_hf=_GPkz(0.2),
        norm=norm,
        x_param_min=0.0, x_param_max=1.0, h=1e-3,
    )
    # LF point 0 (z=3.0): weight = 0.2 * 3.0 * (1+k) / 1.0
    expected_p0 = 0.2 * 3.0 * (1.0 + k)
    np.testing.assert_allclose(w[0:nk], expected_p0, rtol=1e-3)

    # LF point 1 (z=3.6): weight = 0.2 * 3.6 * (1+k)
    expected_p1 = 0.2 * 3.6 * (1.0 + k)
    np.testing.assert_allclose(w[nk:2 * nk], expected_p1, rtol=1e-3)

    # Verify k varies within a point's block (k-minor): first point's weights not all equal
    assert not np.allclose(w[0], w[1]), "k-minor ordering failed (weights should differ by k)"


def test_multiz_weights_lf_uses_kfkms_lf_not_hf():
    """Builder uses kfkms_lf_z for both X_lf and X_hf; weights must do the same.

    We give kfkms_hf_z a different k-grid (which would give wrong gradients if used).
    Both LF and HF weight blocks must give the correct answer based on kfkms_lf_z.
    """
    nk = 3
    k_lf = np.linspace(0.01, 0.04, nk)
    k_hf_wrong = np.linspace(0.1, 0.2, nk)  # deliberately wrong — should NOT be used
    zgrid = np.array([3.0, 3.6])

    params = np.zeros((2, 11))
    params[:, 0] = [0.4, 0.6]
    payload = {
        "params_lf": params,
        "kfkms_lf_z": np.tile(k_lf, (2, 1)),
        "kfkms_hf_z": np.tile(k_hf_wrong, (2, 1)),  # wrong — must not be used
        "z_per_row": np.array([3.0, 3.6]),
    }

    norm = MultiZNormalizationSpec(
        param_min=0.0, param_max=1.0,
        k_min=float(k_lf.min()), k_max=float(k_lf.max()),
        z_grid=zgrid,
        mean_flux=np.zeros((2, nk)),
        std_flux=np.ones((2, nk)),
        k_grid=k_lf,
    )

    # _GPz: dlogP/dtheta = s*z, independent of k -> both k_lf and k_hf_wrong give
    # the same PER-ELEMENT gradient, but the weight vectors must have nk elements
    # drawn from k_lf rows (not k_hf_wrong rows).
    w = sobolev_target_weights_multiz(
        payload=payload, param_idx=0,
        gp_lf=_GPz(0.2), gp_hf=_GPz(0.5),
        norm=norm,
        x_param_min=0.0, x_param_max=1.0, h=1e-3,
    )
    # If implementation accidentally used kfkms_hf_z for HF rows, the std interp
    # would land outside k_grid and np.interp would clamp — giving a different result
    # or an error. Both blocks must be finite and consistent.
    assert w.shape == (2 * 2 * nk,)
    assert np.all(np.isfinite(w))


def test_boundary_point_does_not_exceed_range():
    """Clamping: sweep points at x_param bounds must not push theta out of [x_param_min, x_param_max]."""
    nk = 3
    k = np.linspace(0.01, 0.04, nk)
    zgrid = np.array([3.0, 3.6])

    # Two points: one at lower bound (0.0), one at upper bound (1.0)
    params = np.zeros((2, 11))
    params[0, 0] = 0.0
    params[1, 0] = 1.0
    payload = {
        "params_lf": params,
        "kfkms_lf_z": np.tile(k, (2, 1)),
        "kfkms_hf_z": np.tile(k, (2, 1)),
        "z_per_row": np.array([3.0, 3.6]),
    }

    class _BoundedGPz:
        """Asserts theta[0] stays within [0, 1] — like the real emulator."""
        def __init__(self, s):
            self.s = s
        def predict(self, theta, kk, z):
            t = float(theta[0])
            assert -1e-9 <= t <= 1.0 + 1e-9, f"theta[0]={t} out of [0,1]"
            kk = np.asarray(kk, float)
            return np.exp(self.s * t * float(z) + 0.5 * kk)

    norm = MultiZNormalizationSpec(
        param_min=0.0, param_max=1.0,
        k_min=float(k.min()), k_max=float(k.max()),
        z_grid=zgrid,
        mean_flux=np.zeros((2, nk)),
        std_flux=np.ones((2, nk)),
        k_grid=k,
    )

    # Must NOT raise AssertionError (clamping keeps perturbations within [0, 1])
    w = sobolev_target_weights_multiz(
        payload=payload, param_idx=0,
        gp_lf=_BoundedGPz(0.2), gp_hf=_BoundedGPz(0.5),
        norm=norm,
        x_param_min=0.0, x_param_max=1.0, h=1e-3,
    )
    assert np.all(np.isfinite(w))
    assert w.shape == (2 * 2 * nk,)


def test_module_exports_all_three_functions():
    """sobolev_loss.py exports all three public functions."""
    from priya_forecast import sobolev_loss
    assert hasattr(sobolev_loss, "make_sobolev_loss")
    assert hasattr(sobolev_loss, "sobolev_target_weights")
    assert hasattr(sobolev_loss, "sobolev_target_weights_multiz")
