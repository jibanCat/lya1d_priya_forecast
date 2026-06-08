"""Read/write per-candidate gradient-faithfulness sidecars.

A sidecar pairs 1:1 with a PySR Pareto CSV and records, for every
Fisher-safe candidate, two emulator-grounded metrics evaluated against the
GP: the derivative-faithfulness metric the production gate uses (`grad_err`
= median_k |d_eq/d_theta / d_P_GP/d_theta - 1| at fid) and a common value
loss (`value_mse` = mean (logP_eq - logP_GP)^2 over a theta x k grid). The
common value loss makes the value-Pareto y-axis comparable across runs that
were trained with different objectives (value MSE vs the Sobolev loss).
Kept emulator-free so the plotter can consume it without GPy/lyaemu.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SIDECAR_COLUMNS = [
    "Complexity", "Loss", "grad_err", "value_mse", "n_keep", "gate_pass",
    "x0_enters",
]

_X0 = re.compile(r"\bx0\b")


def equation_has_x0(equation_str: str) -> bool:
    """True if the PySR equation references the parameter feature x0.

    Word-boundary match so a different feature like x01 is not counted.
    """
    return _X0.search(str(equation_str)) is not None


def write_grad_faith_sidecar(out_path, rows, *, param, z, tol,
                             log_space, source_pareto):
    """Write a sidecar CSV (one row per candidate) with a provenance header.

    rows: iterable of dicts keyed by SIDECAR_COLUMNS.
    Returns the written Path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows), columns=SIDECAR_COLUMNS)
    header = (f"# param={param} z={z} tol={tol} log_space={log_space} "
              f"source={source_pareto}\n")
    with open(out_path, "w") as fh:
        fh.write(header)
        df.to_csv(fh, index=False)
    return out_path


def read_grad_faith_sidecar(path) -> pd.DataFrame:
    """Read a sidecar CSV, skipping the leading '#' provenance comment."""
    return pd.read_csv(path, comment="#")
