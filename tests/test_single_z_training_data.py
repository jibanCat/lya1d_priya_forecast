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


def test_load_1pvar_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="regen_1pvar.py"):
        load_1pvar(param_name="ns", z=3.6, data_dir=tmp_path)


def test_load_1pvar_z_not_in_grid(tmp_path):
    params = np.zeros((2, 11))
    kfkms = np.ones((2, 1, 3))
    flux = np.ones((2, 1, 3))
    zout = np.array([3.6])
    for fidelity in ("lf", "hf"):
        write_1pvar_hdf5(
            tmp_path / f"{fidelity}_ns_npoints50.hdf5",
            params=params, kfkms=kfkms, flux_vectors=flux, zout=zout,
        )
    with pytest.raises(ValueError, match="not in"):
        load_1pvar(param_name="ns", z=2.4, data_dir=tmp_path)


def test_write_1pvar_rejects_bad_params_shape(tmp_path):
    with pytest.raises(ValueError, match=r"params must be \(n_points, 11\)"):
        write_1pvar_hdf5(
            tmp_path / "lf_ns_npoints50.hdf5",
            params=np.zeros((2, 5)), kfkms=np.ones((2, 1, 3)),
            flux_vectors=np.ones((2, 1, 3)), zout=np.array([3.6]),
        )


def test_regenerate_param_stacks_z(monkeypatch):
    """regenerate_param loops _generate_1pvar_inline over z, stacks on axis 1."""
    n_points, n_k = 4, 6

    def fake_inline(*, gp_lf, gp_hf, param_name, z, k_grid, n_points):
        nk = len(k_grid)
        return {
            "params_lf": np.full((n_points, 11), 1.0),
            "params_hf": np.full((n_points, 11), 1.0),
            "kfkms_lf_z": np.broadcast_to(k_grid, (n_points, nk)).copy(),
            "kfkms_hf_z": np.broadcast_to(k_grid, (n_points, nk)).copy(),
            "flux_lf_z": np.full((n_points, nk), z),
            "flux_hf_z": np.full((n_points, nk), 2.0 * z),
        }

    import priya_forecast.refit_1d_pysr as r1d
    monkeypatch.setattr(r1d, "_generate_1pvar_inline", fake_inline)

    z_grid = np.array([3.2, 3.4, 3.6])
    k_grid = np.linspace(0.001, 0.04, n_k)
    out = regenerate_param(
        gp_lf=None, gp_hf=None, param_name="ns",
        z_grid=z_grid, k_grid=k_grid, n_points=n_points,
    )
    assert out["flux_lf"].shape == (n_points, 3, n_k)
    assert out["kfkms_hf"].shape == (n_points, 3, n_k)
    assert out["params_lf"].shape == (n_points, 11)
    # z axis is axis 1: flux at z-index 1 == 3.4, hf == 2*3.6 at index 2
    np.testing.assert_allclose(out["flux_lf"][:, 1, :], 3.4)
    np.testing.assert_allclose(out["flux_hf"][:, 2, :], 2.0 * 3.6)
    np.testing.assert_allclose(out["zout"], z_grid)
