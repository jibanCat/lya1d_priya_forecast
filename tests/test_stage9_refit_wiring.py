# tests/test_stage9_refit_wiring.py
import numpy as np
from priya_forecast import refit_1d_pysr as R


def test_refit_assembles_sobolev_loss_and_weights(monkeypatch):
    captured = {}

    class _FakePySR:
        def __init__(self, **kw): captured["kwargs"] = kw
        def fit(self, X, y, **kw): captured["fit"] = (np.asarray(X).shape, kw)
        @property
        def equations_(self):
            import pandas as pd
            return pd.DataFrame({"equation": ["x0"], "complexity": [1], "loss": [0.0]})

    # PySRRegressor is imported locally inside refit_1d_for_param via
    # `from pysr import PySRRegressor`, so we must patch pysr.PySRRegressor
    # directly (not as a module-level attribute of refit_1d_pysr).
    import pysr
    monkeypatch.setattr(pysr, "PySRRegressor", _FakePySR)

    class _GP:
        def predict(self, theta, k, z):
            k = np.asarray(k, float); return np.exp(0.3 * float(theta[0]) + 0.5 * k)

    k = np.linspace(0.01, 0.04, 4)
    params = np.zeros((5, 11)); params[:, 0] = np.linspace(0.8, 1.05, 5)
    flux = np.exp(0.3 * params[:, :1] + 0.5 * k[None, :])
    # Include all keys that _generate_1pvar_inline returns (read by
    # _build_training_matrix and compute_local_normalization)
    payload = {
        "params_lf": params, "params_hf": params,
        "flux_lf_z": flux, "flux_hf_z": flux,
        "kfkms_lf_z": np.tile(k, (5, 1)), "kfkms_hf_z": np.tile(k, (5, 1)),
        "zindex_lf": -1, "zindex_hf": -1,
        "kfkms_lf_min": float(k.min()), "kfkms_lf_max": float(k.max()),
    }
    monkeypatch.setattr(R, "_generate_1pvar_inline", lambda **kw: payload, raising=False)

    R.refit_1d_for_param(
        param_name="ns", z=3.6, k_grid=k, gp_lf=_GP(), gp_hf=_GP(),
        log_space=True, use_sobolev=True, sobolev_lambda=3.0, sobolev_h=1e-4)

    assert "loss_function" in captured["kwargs"]
    assert "3.0" in captured["kwargs"]["loss_function"]
    assert "elementwise_loss" not in captured["kwargs"]
    w = captured["fit"][1].get("weights")
    assert w is not None and w.shape[0] == 2 * (5 * 4)


def test_sobolev_requires_gps(monkeypatch):
    import pytest
    from priya_forecast import refit_1d_pysr as R
    import numpy as np

    class _FakePySR:
        def __init__(self, **kw): pass
        def fit(self, X, y, **kw): pass
        @property
        def equations_(self):
            import pandas as pd
            return pd.DataFrame({"equation": ["x0"], "complexity": [1], "loss": [0.0]})

    import pysr
    monkeypatch.setattr(pysr, "PySRRegressor", _FakePySR)

    k = np.linspace(0.01, 0.04, 4)
    params = np.zeros((5, 11))
    flux = np.ones((5, 4))
    payload = {
        "params_lf": params, "params_hf": params,
        "flux_lf_z": flux, "flux_hf_z": flux,
        "kfkms_lf_z": np.tile(k, (5, 1)), "kfkms_hf_z": np.tile(k, (5, 1)),
        "zindex_lf": -1, "zindex_hf": -1,
        "kfkms_lf_min": float(k.min()), "kfkms_lf_max": float(k.max()),
    }
    monkeypatch.setattr(R, "_load_1pvar", lambda **kw: payload, raising=False)

    with pytest.raises(ValueError, match="requires gp_lf and gp_hf"):
        R.refit_1d_for_param(param_name="ns", z=3.6, k_grid=k,
            gp_lf=None, gp_hf=None, use_sobolev=True, data_dir="/tmp")
