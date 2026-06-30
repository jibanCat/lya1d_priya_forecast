"""Unit tests for `priya_forecast.single_z.refit`."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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


def test_pysr_kwargs_loss_decoupled_from_operators():
    """Operator set (smart_kwargs) and training loss (use_anova_loss) are
    independent knobs. The production value baseline = no-trig operators +
    plain MSE (no ANOVA)."""
    from priya_forecast.single_z.config import PipelineConfig, PySRConfig

    # Default production value baseline: smart operators, MSE, no ANOVA.
    value = pysr_kwargs_for_cfg(PipelineConfig(
        pysr=PySRConfig(smart_kwargs=True, use_anova_loss=False)))
    assert "loss_function" not in value
    assert "elementwise_loss" in value
    # No trig — oscillatory derivatives wreck Fisher conditioning.
    assert set(value["unary_operators"]) == {"exp", "log", "square"}

    # Ablation: same operators, ANOVA loss instead of MSE.
    anova = pysr_kwargs_for_cfg(PipelineConfig(
        pysr=PySRConfig(smart_kwargs=True, use_anova_loss=True)))
    assert "loss_function" in anova
    assert "elementwise_loss" not in anova
    assert set(anova["unary_operators"]) == {"exp", "log", "square"}


@settings(max_examples=24, deadline=None)
@given(
    smart=st.booleans(),
    anova=st.booleans(),
    niter=st.integers(min_value=1, max_value=500),
)
def test_property_exactly_one_loss_and_no_trig_when_smart(smart, anova, niter):
    """Invariant: PySR forbids both elementwise_loss and loss_function, so
    exactly one must be present; and the smart operator set never contains
    trig functions."""
    from priya_forecast.single_z.config import PipelineConfig, PySRConfig

    kw = pysr_kwargs_for_cfg(PipelineConfig(
        pysr=PySRConfig(smart_kwargs=smart, use_anova_loss=anova,
                        niterations=niter)))
    has_ew = "elementwise_loss" in kw
    has_lf = "loss_function" in kw
    assert has_ew != has_lf  # exactly one
    assert has_lf == anova   # loss_function present iff ANOVA requested
    assert kw["niterations"] == niter
    if smart:
        assert not ({"sin", "cos", "tan"} & set(kw["unary_operators"]))


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


def test_save_artifacts_propagates_to_engine(tmp_path, monkeypatch):
    """The CLI/wrapper --save-artifacts flag must reach refit_1d_for_param."""
    import pandas as pd
    import priya_forecast.single_z.refit as refit_mod
    from priya_forecast.single_z.config import GPConfig, PipelineConfig, PySRConfig

    seen = {}

    class _R:
        equation_str = "x0 + x1"

    def fake_refit(*, pareto_csv_out, save_artifacts=False, **kwargs):
        seen["save_artifacts"] = save_artifacts
        pd.DataFrame({"Complexity": [3], "Loss": [0.1],
                      "Equation": ["x0 + x1"]}).to_csv(pareto_csv_out, index=False)
        return _R()

    monkeypatch.setattr(refit_mod, "refit_1d_for_param", fake_refit)
    cfg = PipelineConfig(mode="refit_and_forecast",
                         gp=GPConfig(basedir="data/kodiaq_gp"),
                         pysr=PySRConfig(seed=0))
    refit_mod.refit_one_param_single_z(
        param_name="ns", z=3.6, cfg=cfg, gp_lf=None, gp_hf=None,
        k_grid=np.linspace(0.001, 0.04, 8), out_dir=tmp_path,
        save_artifacts=True,
    )
    assert seen["save_artifacts"] is True


def test_cli_rejects_sobolev_without_log_target():
    """The CLI guard fires before any GP import (fast, no heavy deps)."""
    import subprocess
    import sys
    repo = Path(__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONPATH": f"{repo}/src"}
    out = subprocess.run(
        [sys.executable, str(repo / "scripts" / "refit_one_param_single_z.py"),
         "--param", "ns", "--z", "3.6", "--output-dir", "/tmp/_x",
         "--use-sobolev", "--target-space", "linear"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert out.returncode != 0
    assert "target-space log" in out.stderr
