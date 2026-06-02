# tests/test_multi_z_forecast_joint.py
import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_FORECAST_ONLY"),
    reason="needs lyaemu + data/kodiaq_gp; set RUN_SLOW_FORECAST_ONLY=1",
)


def test_perfect_1d_equals_gp_joint_linear_and_log():
    from priya_forecast.models.gp_model import GPModel
    from priya_forecast.parameters import fiducial_vector, PARAM_NAMES
    from priya_forecast.multi_z.config import MultiZPipelineConfig
    from priya_forecast.multi_z.forecast import run_three_fisher_multiz

    fid = np.asarray(fiducial_vector(), float)
    for space in ("linear", "log"):
        cfg = MultiZPipelineConfig(
            mode="forecast_only", z_min=3.4, z_max=3.6,
            parameters=["ns", "Ap", "tau0"], target_space=space,
        )
        cfg.gp.basedir = "data/kodiaq_gp"
        cfg.validate()
        gp = GPModel(basedir=cfg.gp.basedir)
        refits = {n: None for n in PARAM_NAMES}   # perfect_1D == GP
        res = run_three_fisher_multiz(cfg=cfg, gp=gp, fid=fid, refits=refits)
        np.testing.assert_allclose(
            res["perfect_1D"].sigma, res["GP"].sigma, rtol=1e-3,
            err_msg=f"perfect_1D != GP in {space} space",
        )
