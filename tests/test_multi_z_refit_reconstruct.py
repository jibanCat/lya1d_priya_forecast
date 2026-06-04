# tests/test_multi_z_refit_reconstruct.py
import numpy as np
import pandas as pd
from priya_forecast.models.normalization import MultiZNormalizationSpec
from priya_forecast.multi_z.refit import build_refit_from_pareto_multiz


def _write_artifacts(tmp_path, param="ns"):
    df = pd.DataFrame({
        "Complexity": [1, 3],
        "Loss": [1.0, 0.01],
        "Equation": ["0.5", "x0 + x1 + 0.1 * x3"],
    })
    csv = tmp_path / "pareto_ns.csv"
    df.to_csv(csv, index=False)
    z_grid = np.array([3.4, 3.6])
    k_grid = np.linspace(0.005, 0.04, 6)
    spec = MultiZNormalizationSpec(
        param_min=0.0, param_max=2.0, k_min=float(k_grid.min()),
        k_max=float(k_grid.max()), z_grid=z_grid,
        mean_flux=np.outer(1.0 + 0.1 * z_grid, 1.0 + k_grid),
        std_flux=0.2 * np.ones((2, 6)), k_grid=k_grid)
    np.savez(
        tmp_path / "norm_ns.npz",
        param_min=spec.param_min, param_max=spec.param_max,
        k_min=spec.k_min, k_max=spec.k_max,
        z_grid=spec.z_grid, mean_flux=spec.mean_flux,
        std_flux=spec.std_flux, k_grid=spec.k_grid,
        x_param_min=0.1, x_param_max=1.9,        # empirical Sobol range (≠ prior 0..2)
        result_k_min=spec.k_min, result_k_max=spec.k_max,
    )
    return csv, tmp_path / "norm_ns.npz"


def test_reconstruct_predicts_per_z(tmp_path):
    csv, norm = _write_artifacts(tmp_path)
    r = build_refit_from_pareto_multiz(
        param_name="ns", z_min=3.4, z_max=3.6, pareto_csv=csv,
        norm_npz=norm, pick_rule="best_loss")
    assert r.is_multiz
    out = r.predict(theta_phys=r.fid_value, k=np.linspace(0.005, 0.04, 6),
                    resolution=0.8, z=3.6)
    assert out.shape == (6,)


def test_reconstruct_uses_empirical_param_range_not_prior(tmp_path):
    csv, norm = _write_artifacts(tmp_path)   # prior 0..2, empirical 0.1..1.9
    r = build_refit_from_pareto_multiz(
        param_name="ns", z_min=3.4, z_max=3.6, pareto_csv=csv,
        norm_npz=norm, pick_rule="best_loss")
    assert r.x_param_min == 0.1 and r.x_param_max == 1.9
