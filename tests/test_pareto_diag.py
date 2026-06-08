import numpy as np
import pandas as pd

from priya_forecast.grad_faith_io import write_grad_faith_sidecar
from priya_forecast.pareto_diag import load_front, render_grid


def _write_pareto(path):
    pd.DataFrame({
        "Complexity": [1, 3, 6],
        "Loss": [24.6, 10.0, 4.2],
        "Equation": ["x0", "x0 * 2.66", "log(x0 + 0.16)"],
    }).to_csv(path, index=False)


def test_load_front_without_sidecar_is_all_nan(tmp_path):
    pareto = tmp_path / "pareto_ns.csv"
    _write_pareto(pareto)
    front = load_front(pareto, None)
    assert list(front["Complexity"]) == [1, 3, 6]
    assert front["grad_err"].isna().all()


def test_load_front_with_sidecar_joins_grad_err(tmp_path):
    pareto = tmp_path / "pareto_ns.csv"
    _write_pareto(pareto)
    side = write_grad_faith_sidecar(
        tmp_path / "grad_faith_ns.csv",
        [
            {"Complexity": 1, "Loss": 24.6, "grad_err": 0.90,
             "n_keep": 40, "gate_pass": False, "x0_enters": True},
            {"Complexity": 3, "Loss": 10.0, "grad_err": 0.13,
             "n_keep": 40, "gate_pass": True, "x0_enters": True},
        ],
        param="ns", z=3.6, tol=0.25, log_space=True, source_pareto=str(pareto),
    )
    front = load_front(pareto, side)
    # complexity 6 has no sidecar row -> grad_err NaN (left join)
    assert np.isnan(front.loc[front.Complexity == 6, "grad_err"].item())
    assert front.loc[front.Complexity == 1, "grad_err"].item() == 0.90


def test_render_grid_writes_nonempty_png(tmp_path):
    pareto = tmp_path / "pareto_ns.csv"
    _write_pareto(pareto)
    front = load_front(pareto, None)  # gray-fallback path
    out = tmp_path / "fig.png"
    render_grid(
        {"ns": [{"front": front, "label": "value@20", "marker": "o"}]},
        out, param_order=["ns"],
    )
    assert out.exists() and out.stat().st_size > 0
