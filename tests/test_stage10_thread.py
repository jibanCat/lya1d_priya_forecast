# tests/test_stage10_thread.py
import numpy as np, tempfile
import pytest
from priya_forecast.multi_z import refit as _mr


def test_multiz_refit_one_param_sobolev_is_disabled():
    """M2 guard: multi-z Sobolev is disabled because the multi-z training target is
    built in linear P_F while the Sobolev target gradient is log-P; that log/linear
    mismatch would silently corrupt the forecast. The driver must raise rather than
    thread the request through to a corrupt run (single-z Sobolev is the supported
    path; the multi-z money plot is dropped)."""
    from priya_forecast.multi_z.config import MultiZPipelineConfig

    cfg = MultiZPipelineConfig(mode="refit_and_forecast", z_min=2.6, z_max=4.2,
                               target_space="log")
    cfg.pysr.use_sobolev = True
    cfg.pysr.sobolev_lambda = 4.0

    with pytest.raises(NotImplementedError, match="Multi-z Sobolev"):
        _mr.refit_one_param_multi_z(
            param_name="ns", z_min=2.6, z_max=4.2, cfg=cfg,
            gp_lf=object(), gp_hf=object(),
            k_grid=np.linspace(0.01, 0.04, 4), out_dir=tempfile.mkdtemp(),
        )
