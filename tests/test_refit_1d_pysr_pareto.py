"""Tests for the pareto_csv_out feature added to refit_1d_for_param."""
from __future__ import annotations


def test_pareto_csv_out_is_load_pareto_csv_compatible(tmp_path):
    """A frame written the way pareto_csv_out writes it round-trips through load_pareto_csv."""
    import pandas as pd
    from priya_forecast.models.pysr_model import load_pareto_csv

    eqs = pd.DataFrame({"complexity": [1, 3], "loss": [0.5, 0.1],
                        "equation": ["x0", "x0 + x1"]})
    out = tmp_path / "pareto_ns.csv"
    eqs.rename(columns={"complexity": "Complexity", "loss": "Loss",
                        "equation": "Equation"}).to_csv(out, index=False)
    df = load_pareto_csv(out)
    assert list(df.columns[:3]) == ["Complexity", "Loss", "Equation"]
    assert len(df) == 2
