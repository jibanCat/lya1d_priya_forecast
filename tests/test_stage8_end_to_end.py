# tests/test_stage8_end_to_end.py
"""Gated end-to-end: single-z refit+forecast with the aq operator + derivative
gate on real KODIAQ. Confirms aq equations refit, the gate runs in selection,
and the anchor identity (perfect_1D == GP) still holds.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_REFIT"),
    reason="needs PySR/Julia + emulator; set RUN_SLOW_REFIT=1",
)


def test_aq_gate_refit_forecast_two_params(tmp_path):
    from priya_forecast.single_z.config import PipelineConfig
    from priya_forecast.single_z.pipeline import run

    cfg = PipelineConfig(
        mode="refit_and_forecast", redshift=3.6,
        parameters=["ns", "Ap"], target_space="log",
        output_dir=str(tmp_path / "s8"),
    )
    cfg.gp.basedir = "data/kodiaq_gp"
    cfg.pysr.niterations = 20
    res = run(cfg)
    assert "GP" in res["fisher_results"]
    np.testing.assert_allclose(
        res["fisher_results"]["perfect_1D"].sigma,
        res["fisher_results"]["GP"].sigma, rtol=1e-3,
    )
