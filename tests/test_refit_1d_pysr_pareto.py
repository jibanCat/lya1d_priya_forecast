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
