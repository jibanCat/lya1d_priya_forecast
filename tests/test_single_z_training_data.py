"""Unit tests for `priya_forecast.single_z.training_data`.

Pure tests (no lyaemu / no emulator). The end-to-end regen smoke at the
bottom is gated on `RUN_SLOW_REGEN_1PVAR=1`.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from priya_forecast.single_z.training_data import (
    load_1pvar,
    regenerate_param,
    write_1pvar_hdf5,
)


def test_write_load_1pvar_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    n_points, n_z, n_k = 5, 3, 7
    params = rng.random((n_points, 11))
    kfkms = np.broadcast_to(
        np.linspace(0.001, 0.04, n_k), (n_points, n_z, n_k)
    ).copy()
    flux_lf = rng.random((n_points, n_z, n_k))
    flux_hf = rng.random((n_points, n_z, n_k))
    zout = np.array([3.2, 3.4, 3.6])
    write_1pvar_hdf5(
        tmp_path / "lf_ns_npoints50.hdf5",
        params=params, kfkms=kfkms, flux_vectors=flux_lf, zout=zout,
    )
    write_1pvar_hdf5(
        tmp_path / "hf_ns_npoints50.hdf5",
        params=params, kfkms=kfkms, flux_vectors=flux_hf, zout=zout,
    )
    got = load_1pvar(param_name="ns", z=3.4, data_dir=tmp_path)
    # z=3.4 is z-index 1; lf and hf are distinct arrays
    np.testing.assert_allclose(got["flux_lf_z"], flux_lf[:, 1, :])
    np.testing.assert_allclose(got["flux_hf_z"], flux_hf[:, 1, :])
    np.testing.assert_allclose(got["kfkms_lf_z"], kfkms[:, 1, :])
    assert got["params_lf"].shape == (n_points, 11)
    assert got["kfkms_lf_min"] == pytest.approx(0.001)
    assert got["kfkms_lf_max"] == pytest.approx(0.04)
