"""Unit tests for `priya_forecast.single_z.refit`."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from priya_forecast.single_z.refit import kodiaq_k_grid, pysr_kwargs_for_cfg


def test_kodiaq_k_grid_is_log_spaced_in_range():
    k = kodiaq_k_grid(0.001, 0.04, 48)
    assert k.shape == (48,)
    assert k[0] == pytest.approx(0.001)
    assert k[-1] == pytest.approx(0.04)
    ratios = k[1:] / k[:-1]
    assert np.allclose(ratios, ratios[0])


def test_pysr_kwargs_for_cfg_smart_and_default():
    from priya_forecast.single_z.config import PipelineConfig, PySRConfig

    smart = pysr_kwargs_for_cfg(PipelineConfig(
        pysr=PySRConfig(smart_kwargs=True, niterations=7, maxsize=15)))
    assert smart["niterations"] == 7
    assert smart["maxsize"] == 15
    assert set(smart["unary_operators"]) == {"exp", "log", "square"}

    plain = pysr_kwargs_for_cfg(PipelineConfig(
        pysr=PySRConfig(smart_kwargs=False, niterations=9)))
    assert plain["niterations"] == 9
    assert "sqrt" in plain["unary_operators"]
