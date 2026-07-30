from pathlib import Path
import pandas as pd
from priya_forecast.rerun import compare_to_production


def _sidecar(path, grad_err, value_mse):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("# fixture\n")
    pd.DataFrame({"Complexity": [8], "Loss": [0.1], "grad_err": [grad_err],
                  "value_mse": [value_mse], "n_keep": [40], "gate_pass": [grad_err <= 0.25],
                  "x0_enters": [True]}).to_csv(path, mode="a", index=False)


def test_compare_reports_deltas_and_flags(tmp_path):
    run = tmp_path / "rerun"; prod = tmp_path / "prod"
    _sidecar(run / "sobolev/refit/z3.6/grad_faith_ns.csv", 0.40, 2e-4)   # worse
    _sidecar(prod / "sobolev/refit/z3.6/grad_faith_ns.csv", 0.19, 1e-4)
    df = compare_to_production(run, prod, zs=[3.6], arms=["sobolev"])
    row = df[df.param == "ns"].iloc[0]
    assert round(row.d_grad_err, 3) == round(0.40 - 0.19, 3)
    assert row.flag == "worse"                    # higher grad_err
    assert bool(row.flipped) is True              # prod faithful(<=.25), rerun not


def test_compare_missing_sidecar_is_na_not_error(tmp_path):
    run = tmp_path / "rerun"; prod = tmp_path / "prod"
    _sidecar(prod / "sobolev/refit/z3.6/grad_faith_ns.csv", 0.19, 1e-4)  # only prod
    df = compare_to_production(run, prod, zs=[3.6], arms=["sobolev"], )
    row = df[df.param == "ns"].iloc[0]
    assert row.flag == "n/a"
