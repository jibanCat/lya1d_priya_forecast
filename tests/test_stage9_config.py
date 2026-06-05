# tests/test_stage9_config.py
from priya_forecast.single_z.config import PySRConfig


def test_sobolev_defaults_off():
    c = PySRConfig()
    assert c.use_sobolev is False and c.sobolev_lambda == 1.0
    c.validate()


def test_sobolev_lambda_validated():
    import pytest
    c = PySRConfig(use_sobolev=True, sobolev_lambda=-1.0)
    with pytest.raises(ValueError, match="sobolev_lambda"):
        c.validate()
