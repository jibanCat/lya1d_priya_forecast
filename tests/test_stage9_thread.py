# tests/test_stage9_thread.py
import numpy as np, tempfile
from priya_forecast.single_z import refit as _r


def test_refit_one_param_passes_sobolev(monkeypatch):
    seen = {}
    def _fake_refit_1d_for_param(**kw):
        seen.update(kw)
        class _R:
            equation_str = "x0"; pareto_complexity = 1; pareto_loss = 0.0
        return _R()
    monkeypatch.setattr(_r, "refit_1d_for_param", _fake_refit_1d_for_param, raising=True)
    # make the retry-loop's "is it fisher-safe?" check pass on the first attempt.
    # load_pareto_csv is imported inside the function from priya_forecast.models.pysr_model,
    # so patch it there.
    import pandas as pd
    import priya_forecast.models.pysr_model as _pysr_model
    monkeypatch.setattr(
        _pysr_model, "load_pareto_csv",
        lambda p: pd.DataFrame({"Equation": ["x0"], "Complexity": [1], "Loss": [0.0]}),
        raising=True,
    )
    from priya_forecast.single_z.forecast import _filter_fisher_safe  # noqa: F401
    from priya_forecast.single_z.config import PipelineConfig
    cfg = PipelineConfig(mode="refit_and_forecast", redshift=3.6, target_space="log")
    cfg.pysr.use_sobolev = True
    cfg.pysr.sobolev_lambda = 4.0
    _r.refit_one_param_single_z(param_name="ns", z=3.6, cfg=cfg, gp_lf=object(),
        gp_hf=object(), k_grid=np.linspace(0.01, 0.04, 4), out_dir=tempfile.mkdtemp())
    assert seen.get("use_sobolev") is True
    assert seen.get("sobolev_lambda") == 4.0
