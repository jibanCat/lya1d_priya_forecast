# tests/test_stage8_gradients.py
import numpy as np
from priya_forecast.derivative_gate import gp_param_gradient, equation_param_gradient
from priya_forecast.refit_1d_pysr import Refit1DResult
from priya_forecast.models.normalization import NormalizationSpec


class _LinGP:
    # P_F(theta,k,z) = base(k) * (1 + s*theta[i]); dP/dtheta_i = base(k)*s
    def __init__(self, i, s=0.3):
        self.i, self.s = i, s
    def predict(self, theta, k, z):
        k = np.asarray(k, float)
        return (1.0 + 0.5 * k) * (1.0 + self.s * float(theta[self.i]))


def test_gp_param_gradient_matches_analytic():
    k = np.linspace(0.005, 0.04, 6)
    fid = np.zeros(11)
    g = gp_param_gradient(gp=_LinGP(2), fid=fid, k_grid=k, z=3.6, param_idx=2, h=1e-3)
    np.testing.assert_allclose(g, (1.0 + 0.5 * k) * 0.3, rtol=1e-4)


def test_equation_param_gradient_matches_predict_fd():
    k = np.linspace(0.005, 0.04, 6)
    norm = NormalizationSpec(param_min=0.0, param_max=2.0, k_min=float(k.min()),
        k_max=float(k.max()), mean_flux=np.zeros(6), std_flux=np.ones(6), k_grid=k)
    r = Refit1DResult(param_name="ns", z=3.6, equation_str="x0 + x1",
        pareto_complexity=3, pareto_loss=0.01, pareto_complexities=[3],
        pareto_losses=[0.01], x_param_min=0.0, x_param_max=2.0, k_min=float(k.min()),
        k_max=float(k.max()), lf_resolution=0.4, hf_resolution=0.8, fid_value=1.0,
        norm=norm, k_grid=k, wall_time_s=0.0, lf_train_mean_rel_err=0.0,
        hf_train_mean_rel_err=0.0, lf_train_max_rel_err=0.0, hf_train_max_rel_err=0.0)
    g = equation_param_gradient(refit=r, fid_value=1.0, k_grid=k, z=3.6, h=1e-3)
    assert g.shape == (6,) and np.all(np.isfinite(g))
