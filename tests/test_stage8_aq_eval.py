# tests/test_stage8_aq_eval.py
import numpy as np
from priya_forecast.refit_1d_pysr import Refit1DResult
from priya_forecast.models.normalization import NormalizationSpec
from priya_forecast.pareto_filters import is_fisher_stencil_safe


def _aq_refit():
    k_grid = np.linspace(0.005, 0.04, 6)
    norm = NormalizationSpec(
        param_min=0.0, param_max=2.0, k_min=float(k_grid.min()),
        k_max=float(k_grid.max()), mean_flux=np.zeros(6), std_flux=np.ones(6),
        k_grid=k_grid)
    return Refit1DResult(
        param_name="ns", z=3.6, equation_str="aq(x0, x1)",
        pareto_complexity=3, pareto_loss=0.01, pareto_complexities=[3],
        pareto_losses=[0.01], x_param_min=0.0, x_param_max=2.0,
        k_min=float(k_grid.min()), k_max=float(k_grid.max()),
        lf_resolution=0.4, hf_resolution=0.8, fid_value=1.0, norm=norm,
        k_grid=k_grid, wall_time_s=0.0, lf_train_mean_rel_err=0.0,
        hf_train_mean_rel_err=0.0, lf_train_max_rel_err=0.0,
        hf_train_max_rel_err=0.0)


def test_predict_normalized_handles_aq():
    r = _aq_refit()
    k_grid = np.linspace(0.005, 0.04, 6)
    out = r.predict_normalized(theta_phys=1.0, k=k_grid, resolution=0.8)
    k_norm = (k_grid - k_grid.min()) / (k_grid.max() - k_grid.min())
    np.testing.assert_allclose(out, 0.5 / np.sqrt(1 + k_norm**2), rtol=1e-10)


def test_is_fisher_stencil_safe_handles_aq():
    assert is_fisher_stencil_safe("aq(x0, x1)", n_features=3) is True
