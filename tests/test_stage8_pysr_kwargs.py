# tests/test_stage8_pysr_kwargs.py
from priya_forecast.refit_1d_pysr import DEFAULT_PYSR_KWARGS, SMART_REFIT_PYSR_KWARGS
from priya_forecast.custom_operators import AQ_JULIA


def test_raw_division_dropped_aq_added():
    for kw in (DEFAULT_PYSR_KWARGS, SMART_REFIT_PYSR_KWARGS):
        ops = kw["binary_operators"]
        assert "/" not in ops, f"raw / must be dropped, got {ops}"
        assert AQ_JULIA in ops, f"aq must be present, got {ops}"


def test_extra_sympy_mappings_have_aq():
    for kw in (DEFAULT_PYSR_KWARGS, SMART_REFIT_PYSR_KWARGS):
        assert "aq" in kw["extra_sympy_mappings"]
