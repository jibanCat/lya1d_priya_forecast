# tests/test_stage9_config.py
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from priya_forecast.single_z.config import PipelineConfig, PySRConfig


def test_sobolev_defaults_off():
    c = PySRConfig()
    assert c.use_sobolev is False and c.sobolev_lambda == 1.0
    c.validate()


def test_sobolev_lambda_validated():
    c = PySRConfig(use_sobolev=True, sobolev_lambda=-1.0)
    with pytest.raises(ValueError, match="sobolev_lambda"):
        c.validate()


def test_sobolev_requires_log_target():
    # The Sobolev loss matches d(logP)/dtheta; a linear target silently
    # mismatches the gradient. The pipeline must reject the combination.
    cfg = PipelineConfig(
        target_space="linear",
        pysr=PySRConfig(use_sobolev=True, sobolev_lambda=5.0),
    )
    with pytest.raises(ValueError, match="target_space"):
        cfg.validate()


def test_sobolev_with_log_target_passes_guard(tmp_path):
    # tmp_path gives gp.validate() an existing basedir so we isolate the guard.
    from priya_forecast.single_z.config import GPConfig
    cfg = PipelineConfig(
        target_space="log",
        gp=GPConfig(basedir=str(tmp_path)),
        pysr=PySRConfig(use_sobolev=True, sobolev_lambda=5.0),
    )
    cfg.validate()  # must not raise


@settings(max_examples=20, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(use_sobolev=st.booleans(), target_space=st.sampled_from(["linear", "log"]))
def test_property_sobolev_guard_iff_linear(tmp_path, use_sobolev, target_space):
    from priya_forecast.single_z.config import GPConfig
    cfg = PipelineConfig(
        target_space=target_space,
        gp=GPConfig(basedir=str(tmp_path)),
        pysr=PySRConfig(use_sobolev=use_sobolev, sobolev_lambda=5.0),
    )
    should_raise = use_sobolev and target_space == "linear"
    if should_raise:
        with pytest.raises(ValueError, match="target_space"):
            cfg.validate()
    else:
        cfg.validate()  # the only failure mode here is the guard; must pass
