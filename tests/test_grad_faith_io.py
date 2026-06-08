from priya_forecast.grad_faith_io import (
    equation_has_x0, write_grad_faith_sidecar, read_grad_faith_sidecar,
    SIDECAR_COLUMNS,
)


def test_equation_has_x0_word_boundary():
    assert equation_has_x0("x0 * 2.6589415")
    assert equation_has_x0("log(x0 + 0.1623519)")
    assert not equation_has_x0("x1 * 3.0")
    assert not equation_has_x0("2.5")
    # must not match a longer feature name that merely starts with x0
    assert not equation_has_x0("x01 + 1.0")


def test_sidecar_roundtrip_preserves_columns_and_bool(tmp_path):
    rows = [
        {"Complexity": 1, "Loss": 24.636, "grad_err": 0.90,
         "n_keep": 40, "gate_pass": False, "x0_enters": True},
        {"Complexity": 3, "Loss": 10.020, "grad_err": 0.134,
         "n_keep": 40, "gate_pass": True, "x0_enters": True},
    ]
    out = write_grad_faith_sidecar(
        tmp_path / "grad_faith_ns.csv", rows,
        param="ns", z=3.6, tol=0.25, log_space=True,
        source_pareto="results/x/pareto_ns.csv",
    )
    df = read_grad_faith_sidecar(out)
    assert list(df.columns) == SIDECAR_COLUMNS
    # the leading "# param=..." comment line must be skipped, not parsed as data
    assert len(df) == 2
    # gate_pass must round-trip as a real boolean column
    assert df["gate_pass"].dtype == bool
    assert bool(df.loc[df.Complexity == 3, "gate_pass"].item()) is True
