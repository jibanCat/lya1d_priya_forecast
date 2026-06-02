# tests/test_multi_z_pipeline.py
import pytest
from priya_forecast.multi_z.pipeline import DISPATCH, run
from priya_forecast.multi_z.config import MultiZPipelineConfig


def test_dispatch_has_three_modes():
    assert set(DISPATCH) == {"gp_only", "forecast_only", "refit_and_forecast"}


def test_run_validates_before_dispatch():
    cfg = MultiZPipelineConfig(mode="gp_only", z_min=5.0, z_max=6.0)
    cfg.gp.basedir = "."
    with pytest.raises(ValueError, match="z_min/z_max"):
        run(cfg)
