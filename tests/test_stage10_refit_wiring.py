# tests/test_stage10_refit_wiring.py
import numpy as np
import pysr
from priya_forecast import refit_1d_pysr as R


def _make_multiz_payload(nt, nk, k, zpr):
    """Build a stub payload that satisfies _build_training_matrix_multiz
    and sobolev_target_weights_multiz.  All keys returned by
    _generate_1pvar_multiz_inline are present.

    `zpr` must have at least one entry per z-bin in the kodiaq grid that
    falls in [z_min, z_max], because compute_local_normalization_multiz
    iterates over that z-grid and raises if any bin has no payload rows.
    """
    params = np.zeros((nt, 11))
    params[:, 0] = np.linspace(0.8, 1.05, nt)
    flux = np.exp(0.3 * params[:, :1] + 0.5 * k[None, :])
    return {
        "params_lf": params,
        "params_hf": params,
        "flux_lf_z": flux,
        "flux_hf_z": flux,
        "kfkms_lf_z": np.tile(k, (nt, 1)),
        "kfkms_hf_z": np.tile(k, (nt, 1)),
        "z_per_row": zpr,
        "kfkms_lf_min": float(k.min()),
        "kfkms_lf_max": float(k.max()),
        "sobol_seed": 42,
    }


def test_multiz_refit_assembles_sobolev(monkeypatch):
    cap = {}

    class _FakePySR:
        def __init__(self, **kw): cap["kwargs"] = kw
        def fit(self, X, y, **kw): cap["fit"] = (np.asarray(X).shape, kw)

        @property
        def equations_(self):
            import pandas as pd
            return pd.DataFrame({"equation": ["x0"], "complexity": [1], "loss": [0.0]})

    monkeypatch.setattr(pysr, "PySRRegressor", _FakePySR)

    class _GP:
        def predict(self, theta, k, z):
            k = np.asarray(k, float)
            return np.exp(0.3 * float(theta[0]) * float(z) + 0.5 * k)

    nk = 4
    k = np.linspace(0.01, 0.04, nk)
    # z_min=3.0, z_max=4.0 → kodiaq z bins: 3.0, 3.2, 3.4, 3.6, 3.8, 4.0 (6 bins)
    # Each bin needs at least 1 payload row for compute_local_normalization_multiz.
    # Use 2 rows per bin → nt=12.
    zpr = np.array([3.0, 3.0, 3.2, 3.2, 3.4, 3.4, 3.6, 3.6, 3.8, 3.8, 4.0, 4.0])
    nt = len(zpr)
    payload = _make_multiz_payload(nt, nk, k, zpr)
    monkeypatch.setattr(R, "_generate_1pvar_multiz_inline", lambda **kw: payload, raising=False)

    R.refit_1d_multiz_for_param(
        param_name="ns", z_min=3.0, z_max=4.0, k_grid=k,
        gp_lf=_GP(), gp_hf=_GP(), n_total=nt,
        use_sobolev=True, sobolev_lambda=3.0,
        _allow_unvalidated_sobolev=True,  # deliberately exercise the disabled mechanism
    )

    assert "loss_function" in cap["kwargs"], "loss_function not injected into PySR kwargs"
    assert "3.0" in cap["kwargs"]["loss_function"], (
        f"sobolev_lambda=3.0 not in loss_function string: {cap['kwargs']['loss_function']!r}"
    )
    assert "elementwise_loss" not in cap["kwargs"], "elementwise_loss was not removed"
    w = cap["fit"][1].get("weights")
    assert w is not None, "weights not passed to model.fit"
    assert w.shape[0] == 2 * nt * nk, (
        f"Expected weights length {2 * nt * nk}, got {w.shape[0]}"
    )


def test_multiz_refit_sobolev_off_no_weights(monkeypatch):
    """Without use_sobolev=True, model.fit must not receive weights and loss_function
    should only appear if explicitly passed in pysr_kwargs."""
    cap = {}

    class _FakePySR:
        def __init__(self, **kw): cap["kwargs"] = kw
        def fit(self, X, y, **kw): cap["fit"] = (np.asarray(X).shape, kw)

        @property
        def equations_(self):
            import pandas as pd
            return pd.DataFrame({"equation": ["x0"], "complexity": [1], "loss": [0.0]})

    monkeypatch.setattr(pysr, "PySRRegressor", _FakePySR)

    class _GP:
        def predict(self, theta, k, z):
            k = np.asarray(k, float)
            return np.exp(0.3 * float(theta[0]) * float(z) + 0.5 * k)

    nk = 4
    k = np.linspace(0.01, 0.04, nk)
    zpr = np.array([3.0, 3.0, 3.2, 3.2, 3.4, 3.4, 3.6, 3.6, 3.8, 3.8, 4.0, 4.0])
    nt = len(zpr)
    payload = _make_multiz_payload(nt, nk, k, zpr)
    monkeypatch.setattr(R, "_generate_1pvar_multiz_inline", lambda **kw: payload, raising=False)

    R.refit_1d_multiz_for_param(
        param_name="ns", z_min=3.0, z_max=4.0, k_grid=k,
        gp_lf=_GP(), gp_hf=_GP(), n_total=nt,
        use_sobolev=False,
    )

    w = cap["fit"][1].get("weights")
    assert w is None, "weights should not be passed when use_sobolev=False"
    # elementwise_loss should still be present (default MSE)
    assert "elementwise_loss" in cap["kwargs"], "default elementwise_loss disappeared"


def test_multiz_refit_sobolev_requires_gps(monkeypatch):
    """use_sobolev=True without gp_lf/gp_hf must raise ValueError before
    any payload generation."""
    import pytest

    class _FakePySR:
        def __init__(self, **kw): pass
        def fit(self, X, y, **kw): pass

        @property
        def equations_(self):
            import pandas as pd
            return pd.DataFrame({"equation": ["x0"], "complexity": [1], "loss": [0.0]})

    monkeypatch.setattr(pysr, "PySRRegressor", _FakePySR)

    nk = 4
    k = np.linspace(0.01, 0.04, nk)
    zpr = np.array([3.0, 3.0, 3.2, 3.2, 3.4, 3.4, 3.6, 3.6, 3.8, 3.8, 4.0, 4.0])
    nt = len(zpr)
    payload = _make_multiz_payload(nt, nk, k, zpr)
    monkeypatch.setattr(R, "_generate_1pvar_multiz_inline", lambda **kw: payload, raising=False)

    with pytest.raises(ValueError, match="requires gp_lf and gp_hf"):
        R.refit_1d_multiz_for_param(
            param_name="ns", z_min=3.0, z_max=4.0, k_grid=k,
            gp_lf=None, gp_hf=None, n_total=nt,
            use_sobolev=True,
        )


def test_multiz_refit_low_level_sobolev_guard():
    """Defense-in-depth (M2): the low-level refit_1d_multiz_for_param must itself
    refuse a multi-z Sobolev fit BY DEFAULT. The multi-z training target is built in
    linear P_F (_build_training_matrix_multiz) while the Sobolev target gradient is
    log-P (sobolev_loss._fidelity_grad_weights_multiz); that log/linear mismatch would
    silently corrupt the fit. The driver refit_one_param_multi_z already guards this;
    this closes the gap for a direct library caller. The tested wiring path opts in
    explicitly via _allow_unvalidated_sobolev=True (see test_multiz_refit_assembles_sobolev)."""
    import pytest
    with pytest.raises(NotImplementedError, match="Multi-z Sobolev"):
        R.refit_1d_multiz_for_param(
            param_name="ns", z_min=3.0, z_max=4.0,
            k_grid=np.linspace(0.01, 0.04, 4),
            gp_lf=object(), gp_hf=object(), n_total=4,
            use_sobolev=True,
        )
