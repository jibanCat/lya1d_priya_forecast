"""Unit tests for `priya_forecast.single_z.forecast`."""

from __future__ import annotations

import numpy as np
import pytest

from priya_forecast.single_z.forecast import per_param_local_norm


def test_per_param_local_norm_shapes_and_values():
    """Per-k mean/std of an LF flux sweep → a valid NormalizationSpec."""
    rng = np.random.default_rng(0)
    n_points, n_k = 50, 12
    k_grid = np.linspace(0.001, 0.04, n_k)
    flux_lf = rng.random((n_points, n_k)) + 1.0  # strictly positive
    norm = per_param_local_norm(
        flux_lf_z=flux_lf, k_grid=k_grid, param_min=0.8, param_max=1.05,
    )
    assert norm.mean_flux.shape == (n_k,)
    assert norm.std_flux.shape == (n_k,)
    assert np.all(norm.std_flux > 0)
    np.testing.assert_allclose(norm.mean_flux, flux_lf.mean(axis=0))
    np.testing.assert_allclose(norm.k_grid, k_grid)
    assert norm.param_min == 0.8
    assert norm.param_max == 1.05
    assert norm.k_min == pytest.approx(0.001)
    assert norm.k_max == pytest.approx(0.04)


def test_per_param_local_norm_degenerate_std_floored():
    """A k-bin with zero variance must not produce std=0 (NormalizationSpec rejects it)."""
    k_grid = np.linspace(0.001, 0.04, 5)
    flux_lf = np.ones((10, 5)) * 3.0  # zero variance everywhere
    norm = per_param_local_norm(
        flux_lf_z=flux_lf, k_grid=k_grid, param_min=0.0, param_max=1.0,
    )
    assert np.all(norm.std_flux > 0)


def test_build_refit_from_pareto(tmp_path):
    """A Pareto CSV + regenerated 1pvar data → a usable Refit1DResult."""
    import pandas as pd

    from priya_forecast.single_z.training_data import write_1pvar_hdf5
    from priya_forecast.single_z.forecast import build_refit_from_pareto

    # synthetic 1pvar data for param 'ns' across 3 z-bins
    n_points, n_z, n_k = 50, 3, 8
    k_grid = np.linspace(0.001, 0.04, n_k)
    kfkms = np.broadcast_to(k_grid, (n_points, n_z, n_k)).copy()
    rng = np.random.default_rng(1)
    flux = rng.random((n_points, n_z, n_k)) + 1.0
    params = np.tile(np.array([p.fid for p in __import__(
        "priya_forecast.parameters", fromlist=["PARAMS_11D"]).PARAMS_11D]),
        (n_points, 1))
    zout = np.array([3.2, 3.4, 3.6])
    for fid in ("lf", "hf"):
        write_1pvar_hdf5(tmp_path / f"{fid}_ns_npoints50.hdf5",
                         params=params, kfkms=kfkms, flux_vectors=flux, zout=zout)

    # a minimal Pareto CSV: safe equations in x0 (θ_norm)
    csv = tmp_path / "pareto_ns.csv"
    pd.DataFrame({
        "Complexity": [1, 3, 5],
        "Loss": [1.0, 0.1, 0.05],
        "Equation": ["x0", "x0 + x1", "x0 + x1 + 0.1*x2"],
    }).to_csv(csv, index=False)

    refit = build_refit_from_pareto(
        param_name="ns", z=3.6, pareto_csv=csv, pick_rule="best_loss",
        data_1pvar_dir=tmp_path,
    )
    assert refit.param_name == "ns"
    assert refit.z == 3.6
    # best_loss picks the min-Loss row that survives the safety filter
    assert refit.equation_str in {"x0 + x1", "x0 + x1 + 0.1*x2"}
    # the reconstructed result evaluates without error
    pred = refit.predict(theta_phys=0.98, k=k_grid)
    assert pred.shape == k_grid.shape
    assert np.all(np.isfinite(pred))


def test_build_refit_from_pareto_all_filtered_raises(tmp_path):
    """If every Pareto row is Fisher-pathological, fail loud naming the param."""
    import pandas as pd

    from priya_forecast.single_z.training_data import write_1pvar_hdf5
    from priya_forecast.single_z.forecast import build_refit_from_pareto
    from priya_forecast.parameters import PARAMS_11D

    n_points, n_z, n_k = 50, 1, 6
    k_grid = np.linspace(0.001, 0.04, n_k)
    kfkms = np.broadcast_to(k_grid, (n_points, n_z, n_k)).copy()
    flux = np.ones((n_points, n_z, n_k)) + 1.0
    params = np.tile(np.array([p.fid for p in PARAMS_11D]), (n_points, 1))
    for fid in ("lf", "hf"):
        write_1pvar_hdf5(tmp_path / f"{fid}_ns_npoints50.hdf5",
                         params=params, kfkms=kfkms, flux_vectors=flux,
                         zout=np.array([3.6]))
    csv = tmp_path / "pareto_ns.csv"
    # equation with a huge pathological constant — filtered out
    pd.DataFrame({
        "Complexity": [3], "Loss": [0.01],
        "Equation": ["x0 + 1e9"],
    }).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="ns"):
        build_refit_from_pareto(
            param_name="ns", z=3.6, pareto_csv=csv, pick_rule="best_loss",
            data_1pvar_dir=tmp_path,
        )


def test_resolve_pareto_csvs_per_parameter(tmp_path):
    from priya_forecast.single_z.config import (
        PipelineConfig, ParetoCSVsConfig, ParetoEntry,
    )
    from priya_forecast.single_z.forecast import resolve_pareto_csvs

    csv = tmp_path / "ns.csv"
    csv.write_text("Complexity,Loss,Equation\n1,0.1,x0\n")
    cfg = PipelineConfig(
        mode="forecast_only", parameters=["ns"],
        pareto_csvs=ParetoCSVsConfig(
            source="per_parameter",
            per_parameter={"ns": ParetoEntry(pareto_csv=str(csv))},
        ),
    )
    paths = resolve_pareto_csvs(cfg)
    assert paths["ns"] == csv


def test_resolve_pareto_csvs_from_refit(tmp_path):
    from priya_forecast.single_z.config import PipelineConfig, ParetoCSVsConfig
    from priya_forecast.single_z.forecast import resolve_pareto_csvs

    refit_dir = tmp_path / "out" / "refit" / "z3.6"
    refit_dir.mkdir(parents=True)
    (refit_dir / "pareto_ns.csv").write_text("Complexity,Loss,Equation\n1,0.1,x0\n")
    cfg = PipelineConfig(
        mode="forecast_only", redshift=3.6, parameters=["ns"],
        output_dir=str(tmp_path / "out"),
        pareto_csvs=ParetoCSVsConfig(source="from_refit"),
    )
    paths = resolve_pareto_csvs(cfg)
    assert paths["ns"] == refit_dir / "pareto_ns.csv"


def test_resolve_pareto_csvs_missing_raises(tmp_path):
    from priya_forecast.single_z.config import PipelineConfig, ParetoCSVsConfig
    from priya_forecast.single_z.forecast import resolve_pareto_csvs

    cfg = PipelineConfig(
        mode="forecast_only", redshift=3.6, parameters=["ns"],
        output_dir=str(tmp_path / "out"),
        pareto_csvs=ParetoCSVsConfig(source="from_refit"),
    )
    with pytest.raises(FileNotFoundError, match="ns"):
        resolve_pareto_csvs(cfg)


def test_equation_uses_param():
    from priya_forecast.single_z.forecast import equation_uses_param
    assert equation_uses_param("x0 + x1")
    assert equation_uses_param("square(x0) * x2")
    assert not equation_uses_param("x2 * -1.77")
    assert not equation_uses_param("log(x1 - log(x2))")


def test_filter_fisher_safe_drops_x0_free_rows():
    """_filter_fisher_safe keeps only equations that reference x0."""
    import pandas as pd
    from priya_forecast.single_z.forecast import _filter_fisher_safe

    df = pd.DataFrame({
        "Equation": ["x0 + x1", "x2 * -1.77"],
        "Complexity": [3, 3],
        "Loss": [0.1, 0.05],
    })
    result = _filter_fisher_safe(df, n_features=3)
    assert len(result) == 1
    assert result["Equation"].iloc[0] == "x0 + x1"


def test_per_param_local_norm_log_space():
    """per_param_local_norm log_space normalizes log(flux)."""
    rng = np.random.default_rng(0)
    k = np.linspace(0.001, 0.04, 10)
    flux = rng.random((50, 10)) + 1.0
    norm = per_param_local_norm(
        flux_lf_z=flux, k_grid=k, param_min=0.8, param_max=1.05,
        log_space=True,
    )
    np.testing.assert_allclose(norm.mean_flux, np.log(flux).mean(axis=0))
    assert np.all(norm.std_flux > 0)


def test_run_three_fisher_with_mock_gp():
    """run_three_fisher returns 3 comparable FisherResults (eBOSS path, offline)."""
    from priya_forecast.models.gp_model import MockGPModel
    from priya_forecast.parameters import PARAM_NAMES, fiducial_vector
    from priya_forecast.fisher import FisherResult
    from priya_forecast.single_z.config import PipelineConfig, DataConfig, FisherConfig
    from priya_forecast.single_z.forecast import run_three_fisher

    gp = MockGPModel()
    fid = np.asarray(fiducial_vector(), dtype=float)
    cfg = PipelineConfig(
        mode="forecast_only", redshift=3.6, parameters=["ns", "Ap"],
        combine="additive",
        data=DataConfig(source="eboss_dr14"),
        fisher=FisherConfig(step_frac=0.05, rel_tol=0.05),
    )
    results = run_three_fisher(
        cfg=cfg, gp=gp, fid=fid, refits={n: None for n in PARAM_NAMES},
    )
    assert set(results) == {"GP", "perfect_1D", "PySR"}
    for label, fr in results.items():
        assert isinstance(fr, FisherResult)
        assert fr.sigma.shape == (2,)
        assert np.all(np.isfinite(fr.sigma))
        assert np.all(fr.sigma > 0)
    # All three share one covariance → perfect_1D σ == GP σ at fiducial.
    np.testing.assert_allclose(
        results["perfect_1D"].sigma, results["GP"].sigma, rtol=1e-6,
    )
