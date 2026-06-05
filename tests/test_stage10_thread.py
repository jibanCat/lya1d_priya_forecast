# tests/test_stage10_thread.py
import numpy as np, tempfile
from priya_forecast.multi_z import refit as _mr


def test_multiz_refit_one_param_passes_sobolev(monkeypatch):
    seen = {}

    def _fake(**kw):
        seen.update(kw)

        class _Norm:
            def save_npz(self, path):
                pass

        class _R:
            equation_str = "x0"
            pareto_complexity = 1
            pareto_loss = 0.0
            norm = _Norm()
            pareto_complexities = [1]
            pareto_losses = [0.0]

        return _R()

    monkeypatch.setattr(_mr, "refit_1d_multiz_for_param", _fake, raising=True)

    import pandas as pd

    monkeypatch.setattr(
        _mr,
        "load_pareto_csv",
        lambda p: pd.DataFrame(
            {"Equation": ["x0"], "Complexity": [1], "Loss": [0.0]}
        ),
        raising=True,
    )

    # Stub _write_pareto_csv so it doesn't touch the filesystem
    monkeypatch.setattr(_mr, "_write_pareto_csv", lambda result, path: None, raising=True)

    # Stub _save_sidecar so it doesn't need a real norm object or filesystem
    monkeypatch.setattr(_mr, "_save_sidecar", lambda result, path: None, raising=True)

    from priya_forecast.multi_z.config import MultiZPipelineConfig

    cfg = MultiZPipelineConfig(mode="refit_and_forecast", z_min=2.6, z_max=4.2, target_space="log")
    cfg.pysr.use_sobolev = True
    cfg.pysr.sobolev_lambda = 4.0

    _mr.refit_one_param_multi_z(
        param_name="ns",
        z_min=2.6,
        z_max=4.2,
        cfg=cfg,
        gp_lf=object(),
        gp_hf=object(),
        k_grid=np.linspace(0.01, 0.04, 4),
        out_dir=tempfile.mkdtemp(),
    )

    assert seen.get("use_sobolev") is True and seen.get("sobolev_lambda") == 4.0
