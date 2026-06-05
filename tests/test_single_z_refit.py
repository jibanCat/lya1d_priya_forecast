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


def test_refit_one_param_single_z_retries_until_usable(tmp_path, monkeypatch):
    import pandas as pd
    import priya_forecast.single_z.refit as refit_mod
    from priya_forecast.single_z.config import PipelineConfig, PySRConfig

    calls = []

    class _FakeResult:
        def __init__(self, seed):
            self.seed = seed
            self.equation_str = "x0 + x1" if seed >= 1 else "x2 * 2.0"

    def fake_refit(*, param_name, z, k_grid, gp_lf, gp_hf,
                   pysr_kwargs, seed, pareto_csv_out, log_space=False, **kwargs):
        calls.append(seed)
        # seed 0 → x0-free front; seed >= 1 → has an x0 equation
        eq = "x0 + x1" if seed >= 1 else "x2 * 2.0"
        pd.DataFrame({"Complexity": [3], "Loss": [0.1],
                      "Equation": [eq]}).to_csv(pareto_csv_out, index=False)
        return _FakeResult(seed)

    monkeypatch.setattr(refit_mod, "refit_1d_for_param", fake_refit)
    from priya_forecast.single_z.config import GPConfig
    cfg = PipelineConfig(
        mode="refit_and_forecast",
        gp=GPConfig(basedir="data/kodiaq_gp"),
        pysr=PySRConfig(seed=0),
    )
    result = refit_mod.refit_one_param_single_z(
        param_name="ns", z=3.6, cfg=cfg, gp_lf=None, gp_hf=None,
        k_grid=np.linspace(0.001, 0.04, 8), out_dir=tmp_path,
    )
    # attempt 0 (seed 0) had no x0 equation → retried; attempt 1 (seed 1) usable
    assert calls == [0, 1]
    assert result.seed == 1
