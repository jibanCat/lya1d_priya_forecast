# tests/test_multi_z_config.py
import pytest
from priya_forecast.multi_z.config import MultiZPipelineConfig


def test_defaults_valid():
    cfg = MultiZPipelineConfig(mode="gp_only")
    cfg.gp.basedir = "."          # point at an existing dir to pass GPConfig.validate
    cfg.validate()
    assert cfg.z_min == 2.6 and cfg.z_max == 4.2


def test_rejects_inverted_z_range():
    cfg = MultiZPipelineConfig(z_min=4.2, z_max=2.6)
    cfg.gp.basedir = "."
    with pytest.raises(ValueError, match="z_min"):
        cfg.validate()


def test_rejects_multi_d_combine_on_log():
    cfg = MultiZPipelineConfig(combine="multiplicative", target_space="log")
    cfg.gp.basedir = "."
    with pytest.raises(ValueError, match="log"):
        cfg.validate()
