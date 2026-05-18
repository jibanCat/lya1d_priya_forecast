"""Regenerated 1pvar training-data I/O for the single-z pipeline.

`scripts/regen_1pvar.py` writes per-parameter LF/HF HDF5 files through here;
Stage C reads them back. Unlike the legacy `InferenceLyaData/1pvar/` files
(which store `k·P_F/π`), these store **raw P_F** — the quantity PySR fits —
so there is no transform to undo on load.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from priya_forecast import refit_1d_pysr as _r1d


def write_1pvar_hdf5(
    path: str | Path,
    *,
    params: np.ndarray,
    kfkms: np.ndarray,
    flux_vectors: np.ndarray,
    zout: np.ndarray,
) -> Path:
    """Write one fidelity's 1pvar sweep to HDF5.

    Parameters
    ----------
    params : (n_points, 11) — full PRIYA parameter vector per sweep point.
    kfkms : (n_points, n_z, n_k) — k-grid (s/km) per point and z-bin.
    flux_vectors : (n_points, n_z, n_k) — raw P_F (NOT k·P/π).
    zout : (n_z,) — redshift per z-bin, increasing.
    """
    path = Path(path)
    params = np.asarray(params, dtype=float)
    kfkms = np.asarray(kfkms, dtype=float)
    flux_vectors = np.asarray(flux_vectors, dtype=float)
    zout = np.asarray(zout, dtype=float)
    if params.ndim != 2 or params.shape[1] != 11:
        raise ValueError(f"params must be (n_points, 11); got {params.shape}.")
    if flux_vectors.shape != kfkms.shape:
        raise ValueError(
            f"flux_vectors {flux_vectors.shape} must match kfkms {kfkms.shape}."
        )
    if flux_vectors.ndim != 3:
        raise ValueError(f"flux_vectors must be 3-D; got {flux_vectors.shape}.")
    n_points, n_z, _ = flux_vectors.shape
    if params.shape[0] != n_points:
        raise ValueError(
            f"params n_points {params.shape[0]} != flux n_points {n_points}."
        )
    if zout.shape != (n_z,):
        raise ValueError(f"zout must be ({n_z},); got {zout.shape}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as fh:
        fh["params"] = params
        fh["kfkms"] = kfkms
        fh["flux_vectors"] = flux_vectors
        fh["zout"] = zout
        fh.attrs["flux_units"] = "raw_P_F"
    return path


def load_1pvar(
    *, param_name: str, z: float, data_dir: str | Path,
) -> dict[str, np.ndarray]:
    """Load regenerated LF+HF 1pvar data for one param, sliced to one z-bin.

    Returns a dict: `params_lf/hf` (n_points, 11), `kfkms_lf_z/hf_z`
    (n_points, n_k), `flux_lf_z/hf_z` (n_points, n_k, raw P_F),
    `kfkms_lf_min`, `kfkms_lf_max`.

    Raises FileNotFoundError if either HDF5 is absent; ValueError if `z` is
    not within 1e-3 of any stored z-bin.
    """
    data_dir = Path(data_dir)
    out: dict[str, np.ndarray] = {}
    for fidelity in ("lf", "hf"):
        path = data_dir / f"{fidelity}_{param_name}_npoints50.hdf5"
        if not path.exists():
            raise FileNotFoundError(
                f"regenerated 1pvar HDF5 not found: {path}. "
                f"Run `python scripts/regen_1pvar.py` first."
            )
        with h5py.File(path, "r") as fh:
            zout = fh["zout"][:]
            zi = int(np.argmin(np.abs(zout - z)))
            if abs(zout[zi] - z) > 1e-3:
                raise ValueError(
                    f"z={z} not in {path} zout {zout.tolist()}."
                )
            out[f"params_{fidelity}"] = fh["params"][:]
            out[f"kfkms_{fidelity}_z"] = fh["kfkms"][:, zi, :]
            out[f"flux_{fidelity}_z"] = fh["flux_vectors"][:, zi, :]
    out["kfkms_lf_min"] = float(out["kfkms_lf_z"].min())
    out["kfkms_lf_max"] = float(out["kfkms_lf_z"].max())
    return out


def regenerate_param(
    *,
    gp_lf,
    gp_hf,
    param_name: str,
    z_grid: np.ndarray,
    k_grid: np.ndarray,
    n_points: int = 50,
) -> dict[str, np.ndarray]:
    """Sweep one param at every z-bin; return stacked LF+HF arrays.

    `refit_1d_pysr._generate_1pvar_inline` is single-z, so loop it over
    `z_grid` and stack along a new z axis (axis 1). Returns `params_lf/hf`
    (n_points, 11), `kfkms_lf/hf` and `flux_lf/hf` (n_points, n_z, n_k),
    `zout` (n_z,).
    """
    z_grid = np.asarray(z_grid, dtype=float)
    k_grid = np.asarray(k_grid, dtype=float)
    flux_lf, flux_hf, kf_lf, kf_hf = [], [], [], []
    params_lf = params_hf = None
    for z in z_grid:
        gen = _r1d._generate_1pvar_inline(
            gp_lf=gp_lf, gp_hf=gp_hf, param_name=param_name,
            z=float(z), k_grid=k_grid, n_points=n_points,
        )
        flux_lf.append(np.asarray(gen["flux_lf_z"], dtype=float))
        flux_hf.append(np.asarray(gen["flux_hf_z"], dtype=float))
        kf_lf.append(np.asarray(gen["kfkms_lf_z"], dtype=float))
        kf_hf.append(np.asarray(gen["kfkms_hf_z"], dtype=float))
        params_lf = np.asarray(gen["params_lf"], dtype=float)
        params_hf = np.asarray(gen["params_hf"], dtype=float)
    return {
        "params_lf": params_lf,
        "params_hf": params_hf,
        "kfkms_lf": np.stack(kf_lf, axis=1),
        "kfkms_hf": np.stack(kf_hf, axis=1),
        "flux_lf": np.stack(flux_lf, axis=1),
        "flux_hf": np.stack(flux_hf, axis=1),
        "zout": z_grid,
    }
