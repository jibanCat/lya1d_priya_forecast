"""Unit tests for scripts/aggregate_z.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "aggregate_z",
    Path(__file__).parent.parent / "scripts" / "aggregate_z.py",
)


def _load_module():
    mod = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(mod)
    return mod


def _write_fisher_npz(path, sigma, names=("ns", "Ap")):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, F=np.eye(len(names)), cov=np.diag(np.array(sigma) ** 2),
        sigma=np.array(sigma, dtype=float), corr=np.eye(len(names)),
        steps=np.full(len(names), 0.01), param_names=np.array(names),
        theta_fid=np.array([0.98, 1.46]),
    )


def test_collect_sigma_z(tmp_path):
    mod = _load_module()
    for z in (3.4, 3.6):
        _write_fisher_npz(tmp_path / f"z{z}" / "fisher_GP.npz", [0.1 * z, 0.2 * z])
    table = mod.collect_sigma_z(base_dir=tmp_path, label="GP",
                                z_bins=[3.4, 3.6])
    assert table["ns"][3.6] == pytest.approx(0.1 * 3.6)
    assert table["Ap"][3.4] == pytest.approx(0.2 * 3.4)


def test_aggregate_writes_outputs(tmp_path):
    mod = _load_module()
    for z in (3.4, 3.6):
        for lab in ("GP", "perfect_1D", "PySR"):
            _write_fisher_npz(tmp_path / f"z{z}" / f"fisher_{lab}.npz",
                              [0.1, 0.2])
    out = mod.aggregate(base_dir=tmp_path, z_bins=[3.4, 3.6])
    assert (tmp_path / "aggregate" / "sigma_vs_z.png").exists()
    assert (tmp_path / "aggregate" / "sigma_table.md").exists()
    assert out == tmp_path / "aggregate"
