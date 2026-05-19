"""Tests for the pareto_csv_out feature added to refit_1d_for_param."""
from __future__ import annotations


def test_pareto_csv_out_is_load_pareto_csv_compatible(tmp_path):
    """A frame written the way pareto_csv_out writes it round-trips through load_pareto_csv."""
    import pandas as pd
    from priya_forecast.models.pysr_model import load_pareto_csv

    eqs = pd.DataFrame({"complexity": [1, 3], "loss": [0.5, 0.1],
                        "equation": ["x0", "x0 + x1"]})
    out = tmp_path / "pareto_ns.csv"
    eqs.rename(columns={"complexity": "Complexity", "loss": "Loss",
                        "equation": "Equation"}).to_csv(out, index=False)
    df = load_pareto_csv(out)
    assert list(df.columns[:3]) == ["Complexity", "Loss", "Equation"]
    assert len(df) == 2


def test_build_training_matrix_log_space():
    """_build_training_matrix log_space normalizes log(flux)."""
    import numpy as np
    from priya_forecast.refit_1d_pysr import _build_training_matrix
    from priya_forecast.models.normalization import NormalizationSpec

    n_pts, n_k = 50, 6
    k = np.linspace(0.001, 0.04, n_k)
    flux = np.geomspace(10.0, 80.0, n_pts * n_k).reshape(n_pts, n_k)
    params = np.tile(np.linspace(0.8, 1.05, n_pts)[:, None], (1, 11))
    payload = dict(
        flux_lf_z=flux, flux_hf_z=flux,
        kfkms_lf_z=np.tile(k, (n_pts, 1)), kfkms_hf_z=np.tile(k, (n_pts, 1)),
        params_lf=params, params_hf=params,
    )
    log_flux = np.log(flux)
    norm = NormalizationSpec(
        param_min=0.0, param_max=1.0, k_min=float(k.min()), k_max=float(k.max()),
        mean_flux=log_flux.mean(axis=0),
        std_flux=np.where(log_flux.std(axis=0) > 0, log_flux.std(axis=0), 1.0),
        k_grid=k,
    )
    X, Y, ranges, farr = _build_training_matrix(
        payload=payload, param_idx=2, global_norm=norm, log_space=True,
    )
    # Y is normalized log-flux → mean ≈ 0
    assert abs(float(Y.mean())) < 1e-6
    # fidelity_arrays still expose raw flux for diagnostics
    np.testing.assert_allclose(farr["flux_lf"], flux)


def _hand_refit_log(equation_str, k, *, log_space):
    """A Refit1DResult with a hand-written equation, in linear or log space."""
    import numpy as np
    from priya_forecast.models.normalization import NormalizationSpec
    from priya_forecast.refit_1d_pysr import Refit1DResult, HF_RESOLUTION, LF_RESOLUTION

    nk = len(k)
    norm = NormalizationSpec(
        param_min=0.8, param_max=1.05, k_min=float(k.min()), k_max=float(k.max()),
        mean_flux=np.full(nk, 2.0 if log_space else 30.0),
        std_flux=np.full(nk, 0.5 if log_space else 5.0),
        k_grid=np.asarray(k, dtype=float),
    )
    return Refit1DResult(
        param_name="ns", z=3.6, equation_str=equation_str,
        pareto_complexity=3, pareto_loss=0.0,
        pareto_complexities=[3], pareto_losses=[0.0],
        x_param_min=0.8, x_param_max=1.05,
        k_min=float(k.min()), k_max=float(k.max()),
        lf_resolution=LF_RESOLUTION, hf_resolution=HF_RESOLUTION,
        fid_value=0.983, norm=norm, k_grid=np.asarray(k, dtype=float),
        wall_time_s=0.0, lf_train_mean_rel_err=0.0, hf_train_mean_rel_err=0.0,
        lf_train_max_rel_err=0.0, hf_train_max_rel_err=0.0,
        log_space=log_space,
    )


def test_refit1dresult_predict_log_consistency():
    """predict and predict_log are exp/log consistent in both spaces."""
    import numpy as np
    k = np.linspace(0.001, 0.04, 10)
    for log_space in (False, True):
        r = _hand_refit_log("x0 + x1", k, log_space=log_space)
        p = r.predict(theta_phys=0.98, k=k)
        plog = r.predict_log(theta_phys=0.98, k=k)
        assert np.all(p > 0)                       # raw P_F positive
        np.testing.assert_allclose(plog, np.log(p), rtol=1e-9)
        np.testing.assert_allclose(p, np.exp(plog), rtol=1e-9)
