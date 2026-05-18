# Single-z Stage 1 (Foundations) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the two shared foundations the single-z Stage B/C pipeline depends on — emulator-based 1pvar training-data regeneration, and the per-parameter equation combiner.

**Architecture:** Two pieces. (1) `single_z/training_data.py` + `scripts/regen_1pvar.py` sample the kodiaq-squad emulator on a 1pvar design (one parameter swept, others at fiducial) and write raw-`P_F` HDF5 files. (2) `single_z/combine.py` wraps `refit_taylor.AdditiveTaylorModel` in `local_anchored` mode to combine per-parameter equations into one `P_F(θ,k)` forward model. Both are pure, importable, and unit-tested without the emulator; only the end-to-end regen smoke needs it.

**Tech Stack:** Python 3.11, numpy, h5py, pytest + hypothesis. Reuses `refit_1d_pysr._generate_1pvar_inline`, `refit_taylor.AdditiveTaylorModel`, `models.gp_model.GPModel` / `MockGPModel`.

**Spec:** `docs/superpowers/specs/2026-05-18-single-z-stage-bc-design.md` §3 (regen) and §5.4 (combine). One refinement over the spec file map: the regen writer/loader live in a new importable module `single_z/training_data.py` (the spec's "writer and loader co-designed"); `scripts/regen_1pvar.py` is a thin CLI over it.

**Branch:** `single_z_forecast_clean` (current). All commits land there.

**Test command:** pure tests need only `src` on the path —
`PYTHONPATH=src pytest tests/test_single_z_training_data.py tests/test_single_z_combine.py -v`.
The gated regen smoke additionally needs `lyaemu` importable and `data/kodiaq_gp/` present.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/priya_forecast/single_z/training_data.py` | Write/read raw-`P_F` 1pvar HDF5s; `regenerate_param` stacks per-z sweeps. |
| `scripts/regen_1pvar.py` | CLI: load LF+HF emulators, regenerate all 11 params, write 22 HDF5s. |
| `src/priya_forecast/single_z/combine.py` | `build_combined_model` — wrap `AdditiveTaylorModel` per combine mode. |
| `tests/test_single_z_training_data.py` | Unit tests for `training_data.py` + gated regen smoke. |
| `tests/test_single_z_combine.py` | Unit tests for `combine.py`. |
| `.gitignore` | Ignore the generated `data/single_z_1pvar/`. |

---

## Task 1: 1pvar HDF5 write/load round-trip

**Files:**
- Create: `src/priya_forecast/single_z/training_data.py`
- Test: `tests/test_single_z_training_data.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_single_z_training_data.py`:

```python
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
    flux = rng.random((n_points, n_z, n_k))
    zout = np.array([3.2, 3.4, 3.6])
    for fidelity in ("lf", "hf"):
        write_1pvar_hdf5(
            tmp_path / f"{fidelity}_ns_npoints50.hdf5",
            params=params, kfkms=kfkms, flux_vectors=flux, zout=zout,
        )
    got = load_1pvar(param_name="ns", z=3.4, data_dir=tmp_path)
    # z=3.4 is z-index 1
    np.testing.assert_allclose(got["flux_lf_z"], flux[:, 1, :])
    np.testing.assert_allclose(got["flux_hf_z"], flux[:, 1, :])
    np.testing.assert_allclose(got["kfkms_lf_z"], kfkms[:, 1, :])
    assert got["params_lf"].shape == (n_points, 11)
    assert got["kfkms_lf_min"] == pytest.approx(0.001)
    assert got["kfkms_lf_max"] == pytest.approx(0.04)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_single_z_training_data.py::test_write_load_1pvar_roundtrip -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'priya_forecast.single_z.training_data'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/priya_forecast/single_z/training_data.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_single_z_training_data.py::test_write_load_1pvar_roundtrip -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/single_z/training_data.py tests/test_single_z_training_data.py
git commit -m "Stage 1: 1pvar HDF5 write/load round-trip (raw P_F)"
```

---

## Task 2: `load_1pvar` error handling

**Files:**
- Modify: `tests/test_single_z_training_data.py` (append tests)
- (No new implementation — `load_1pvar` already raises; this task verifies it.)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_single_z_training_data.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_single_z_training_data.py -k "missing_file or z_not_in_grid or bad_params" -v`
Expected: PASS (the implementation from Task 1 already raises these). If any FAIL, fix `training_data.py` to match the asserted messages.

- [ ] **Step 3: Commit**

```bash
git add tests/test_single_z_training_data.py
git commit -m "Stage 1: cover load_1pvar/write_1pvar error paths"
```

---

## Task 3: `regenerate_param` stacks per-z sweeps

**Files:**
- Modify: `tests/test_single_z_training_data.py` (append test)
- (Implementation already written in Task 1; this task adds its test.)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_single_z_training_data.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_single_z_training_data.py::test_regenerate_param_stacks_z -v`
Expected: PASS. (`regenerate_param` calls `_r1d._generate_1pvar_inline`; `monkeypatch.setattr` on the module object patches the same reference.) If FAIL, confirm `training_data.py` calls `_r1d._generate_1pvar_inline` via the module attribute, not a `from … import` binding.

- [ ] **Step 3: Commit**

```bash
git add tests/test_single_z_training_data.py
git commit -m "Stage 1: test regenerate_param per-z stacking"
```

---

## Task 4: `scripts/regen_1pvar.py` CLI + gated smoke

**Files:**
- Create: `scripts/regen_1pvar.py`
- Modify: `.gitignore`
- Modify: `tests/test_single_z_training_data.py` (append gated smoke)

- [ ] **Step 1: Write the failing (gated) smoke test**

Append to `tests/test_single_z_training_data.py`:

```python
# --------------------------------------------------------------------------
# Gated end-to-end smoke — needs the real emulator.
# --------------------------------------------------------------------------

RUN_SLOW_REGEN = os.environ.get("RUN_SLOW_REGEN_1PVAR") == "1"
GP_BASEDIR = Path(__file__).parent.parent / "data" / "kodiaq_gp"

try:
    import lyaemu  # noqa: F401

    LYAEMU_AVAILABLE = True
except ImportError:
    LYAEMU_AVAILABLE = False


@pytest.mark.skipif(
    not (RUN_SLOW_REGEN and LYAEMU_AVAILABLE and GP_BASEDIR.exists()),
    reason="gated on RUN_SLOW_REGEN_1PVAR=1 + lyaemu + data/kodiaq_gp/",
)
def test_regen_1pvar_end_to_end(tmp_path):
    """Run scripts/regen_1pvar.py for one param; load it back, check shapes."""
    import subprocess
    import sys

    repo = Path(__file__).parent.parent
    proc = subprocess.run(
        [
            sys.executable, str(repo / "scripts" / "regen_1pvar.py"),
            "--basedir", str(GP_BASEDIR),
            "--output", str(tmp_path),
            "--params", "ns",
            "--nk", "12",
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    got = load_1pvar(param_name="ns", z=3.6, data_dir=tmp_path)
    assert got["flux_lf_z"].shape == (50, 12)
    assert got["flux_hf_z"].shape == (50, 12)
    assert np.all(np.isfinite(got["flux_lf_z"]))
    assert np.all(got["flux_lf_z"] > 0)
```

- [ ] **Step 2: Run test to verify it is collected and skipped**

Run: `PYTHONPATH=src pytest tests/test_single_z_training_data.py::test_regen_1pvar_end_to_end -v`
Expected: SKIPPED with reason "gated on RUN_SLOW_REGEN_1PVAR=1 …" (the script does not exist yet, but the test is gated, so it skips cleanly rather than erroring).

- [ ] **Step 3: Write the regen CLI**

Create `scripts/regen_1pvar.py`:

```python
#!/usr/bin/env python
"""Regenerate per-parameter LF/HF 1pvar training data from the emulator.

Replaces the legacy `InferenceLyaData/1pvar/` HDF5s (Martin Fernandez's
k-range) with data sampled from the kodiaq-squad emulator at the GP basedir,
so the PySR refit trains on the same k-grid the forecast scores against.

Writes raw P_F (not k·P/π) to
`<output>/{lf,hf}_<param>_npoints50.hdf5` for all 11 params and 13 z-bins.

    python scripts/regen_1pvar.py --basedir data/kodiaq_gp \\
        --output data/single_z_1pvar
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from priya_forecast.parameters import PARAM_NAMES, fiducial_vector
from priya_forecast.single_z.training_data import regenerate_param, write_1pvar_hdf5

# The 13 kodiaq z-bins — the emulator's trained_mf/zbin* grid, increasing.
# NOT the stale 9-bin `z_grid_kodiaq` constant in refit_1d_pysr.py.
Z_GRID_13 = np.round(np.arange(2.2, 4.601, 0.2), 1)


def kodiaq_k_grid(kmin: float, kmax: float, nk: int) -> np.ndarray:
    """Log-spaced k-grid (s/km) the refit trains on."""
    return np.geomspace(kmin, kmax, nk)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--basedir", default="data/kodiaq_gp",
                   help="GP emulator basedir (cfg.gp.basedir).")
    p.add_argument("--output", default="data/single_z_1pvar",
                   help="Directory for the regenerated HDF5s.")
    p.add_argument("--kmin", type=float, default=0.001,
                   help="Min k (s/km) of the training grid.")
    p.add_argument("--kmax", type=float, default=0.04,
                   help="Max k (s/km) of the training grid.")
    p.add_argument("--nk", type=int, default=48,
                   help="Number of log-spaced k points.")
    p.add_argument("--n-points", type=int, default=50,
                   help="Sweep points per parameter.")
    p.add_argument("--params", nargs="+", default=list(PARAM_NAMES),
                   help="Subset of parameters (default: all 11).")
    args = p.parse_args()

    from priya_forecast.models.gp_model import GPModel

    k_grid = kodiaq_k_grid(args.kmin, args.kmax, args.nk)
    out_dir = Path(args.output)
    fid = np.asarray(fiducial_vector(), dtype=float)

    print(f"Loading LF + HF emulators from {args.basedir} ...")
    t0 = time.time()
    gp_lf = GPModel(basedir=args.basedir, fidelity="lf", kf=k_grid)
    gp_hf = GPModel(basedir=args.basedir, fidelity="hf", kf=k_grid)
    _ = gp_lf.predict(fid, k_grid, 3.6)  # warm the lazy load
    _ = gp_hf.predict(fid, k_grid, 3.6)
    print(f"  loaded in {time.time() - t0:.0f}s.")

    for pname in args.params:
        t0 = time.time()
        gen = regenerate_param(
            gp_lf=gp_lf, gp_hf=gp_hf, param_name=pname,
            z_grid=Z_GRID_13, k_grid=k_grid, n_points=args.n_points,
        )
        for fidelity in ("lf", "hf"):
            write_1pvar_hdf5(
                out_dir / f"{fidelity}_{pname}_npoints50.hdf5",
                params=gen[f"params_{fidelity}"],
                kfkms=gen[f"kfkms_{fidelity}"],
                flux_vectors=gen[f"flux_{fidelity}"],
                zout=gen["zout"],
            )
        print(f"  [{time.time() - t0:.1f}s] {pname} -> "
              f"{out_dir}/{{lf,hf}}_{pname}_npoints50.hdf5", flush=True)

    print(f"Done. {len(args.params)} params x 2 fidelities written to {out_dir}.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the generated data dir to `.gitignore`**

Append this line to `.gitignore` (the regenerated HDF5s are large and reproducible):

```
data/single_z_1pvar/
```

- [ ] **Step 5: Run the gated smoke (only if the emulator is available)**

If `lyaemu` is importable and `data/kodiaq_gp/` exists:
Run: `RUN_SLOW_REGEN_1PVAR=1 PYTHONPATH=src pytest tests/test_single_z_training_data.py::test_regen_1pvar_end_to_end -v`
Expected: PASS.

If the emulator is not available in this environment: Run the same command without the env var and confirm it reports SKIPPED — that is an acceptable result for this step; the smoke will be exercised later in an environment with the emulator.

- [ ] **Step 6: Run the full pure-test file**

Run: `PYTHONPATH=src pytest tests/test_single_z_training_data.py -v`
Expected: 5 PASS (round-trip, missing_file, z_not_in_grid, bad_params, regenerate_param) + 1 SKIP (the gated smoke) — or 6 PASS if the emulator is available.

- [ ] **Step 7: Commit**

```bash
git add scripts/regen_1pvar.py .gitignore tests/test_single_z_training_data.py
git commit -m "Stage 1: regen_1pvar.py CLI + gated end-to-end smoke"
```

---

## Task 5: `combine.py` — additive (`local_anchored`) path

**Files:**
- Create: `src/priya_forecast/single_z/combine.py`
- Test: `tests/test_single_z_combine.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_single_z_combine.py`:

```python
"""Unit tests for `priya_forecast.single_z.combine`.

Pure tests — `MockGPModel` stands in for the emulator, all 11 refits are
`None` (so the combine falls back to GP 1D slices). At θ=fid every per-D
deviation is zero, so the `local_anchored` combine must return the GP anchor
exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from priya_forecast.models.gp_model import MockGPModel
from priya_forecast.parameters import PARAM_NAMES, fiducial_vector
from priya_forecast.refit_taylor import AdditiveTaylorModel
from priya_forecast.single_z.combine import build_combined_model


def _fid() -> np.ndarray:
    return np.asarray(fiducial_vector(), dtype=float)


def _none_refits() -> dict:
    return {name: None for name in PARAM_NAMES}


def test_build_combined_model_additive_returns_local_anchored():
    gp = MockGPModel()
    k = np.linspace(0.001, 0.04, 20)
    model = build_combined_model(
        combine_mode="additive", gp=gp, fid=_fid(), refits=_none_refits(),
        k_grid=k, z=3.6,
    )
    assert isinstance(model, AdditiveTaylorModel)
    assert model.mode == "local_anchored"


def test_build_combined_model_additive_anchor_identity():
    """At θ=fid the additive/local_anchored combine == the GP anchor."""
    gp = MockGPModel()
    k = np.linspace(0.001, 0.04, 20)
    fid = _fid()
    model = build_combined_model(
        combine_mode="additive", gp=gp, fid=fid, refits=_none_refits(),
        k_grid=k, z=3.6,
    )
    np.testing.assert_allclose(
        model.predict(fid, k, 3.6), gp.predict(fid, k, 3.6), rtol=1e-9,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_single_z_combine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'priya_forecast.single_z.combine'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/priya_forecast/single_z/combine.py`:

```python
"""Combine per-parameter PySR equations into one multi-D forward model.

Thin wrapper over `refit_taylor.AdditiveTaylorModel`. The default `additive`
combine uses `local_anchored` mode — the combine is anchored on the GP
prediction at fiducial θ and the per-D equations supply only the deviations
(found to forecast better than the student's `multi_d` formula). The
`multiplicative` and `joint` modes are reserved in the config schema but not
yet implemented.
"""

from __future__ import annotations

import numpy as np

from priya_forecast.models.base import P1DModel
from priya_forecast.refit_taylor import AdditiveTaylorModel

VALID_COMBINE_MODES = ("additive", "multiplicative", "joint")


def build_combined_model(
    *,
    combine_mode: str,
    gp: P1DModel,
    fid: np.ndarray,
    refits: dict,
    k_grid: np.ndarray,
    z: float,
    global_norm=None,
) -> P1DModel:
    """Construct the combined P_F(θ, k) model for the given combine mode.

    Parameters
    ----------
    combine_mode : one of `VALID_COMBINE_MODES`. Only `additive` is
        implemented; `multiplicative` / `joint` raise NotImplementedError.
    gp : the HF GP emulator — the combine anchor and the perfect-1D fallback
        source for any param whose refit is `None`.
    fid : (11,) fiducial parameter vector, canonical order.
    refits : dict mapping each of the 11 param names to a `Refit1DResult`
        or `None` (None → fall back to the GP's 1D slice for that param).
    k_grid, z : the grid and redshift the combine is built on.
    global_norm : `NormalizationSpec`; only the (unimplemented) `multi_d`
        path uses it, so pass `None` for `additive`/`local_anchored`.
    """
    if combine_mode == "additive":
        return AdditiveTaylorModel(
            gp=gp,
            fid=np.asarray(fid, dtype=float),
            refits=refits,
            global_norm=global_norm,
            k_grid=np.asarray(k_grid, dtype=float),
            z=float(z),
            mode="local_anchored",
        )
    if combine_mode in ("multiplicative", "joint"):
        raise NotImplementedError(
            f"combine mode {combine_mode!r} is not implemented yet; "
            f"only 'additive' is available."
        )
    raise ValueError(
        f"unknown combine mode {combine_mode!r}; "
        f"expected one of {VALID_COMBINE_MODES}."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_single_z_combine.py -v`
Expected: 2 PASS.

If `test_build_combined_model_additive_anchor_identity` FAILS, do not paper over it — `AdditiveTaylorModel(mode="local_anchored")` is documented to return `P_GP(fid)` exactly at θ=fid. A failure means either `refits=None` is not falling back to GP slices as expected, or `MockGPModel` is non-deterministic. Read `refit_taylor.AdditiveTaylorModel.__post_init__` / `.predict` and `MockGPModel`, and fix the test setup to match real behavior before proceeding.

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/single_z/combine.py tests/test_single_z_combine.py
git commit -m "Stage 1: combine.py additive (local_anchored) wrapper"
```

---

## Task 6: `combine.py` — reject unimplemented and unknown modes

**Files:**
- Modify: `tests/test_single_z_combine.py` (append tests)
- (No new implementation — `build_combined_model` already raises; this verifies it.)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_single_z_combine.py`:

```python
@pytest.mark.parametrize("mode", ["multiplicative", "joint"])
def test_build_combined_model_unimplemented_modes(mode):
    with pytest.raises(NotImplementedError, match="not implemented"):
        build_combined_model(
            combine_mode=mode, gp=MockGPModel(), fid=_fid(),
            refits=_none_refits(), k_grid=np.linspace(0.001, 0.04, 10), z=3.6,
        )


def test_build_combined_model_unknown_mode():
    with pytest.raises(ValueError, match="unknown combine mode"):
        build_combined_model(
            combine_mode="bogus", gp=MockGPModel(), fid=_fid(),
            refits=_none_refits(), k_grid=np.linspace(0.001, 0.04, 10), z=3.6,
        )
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_single_z_combine.py -v`
Expected: 5 PASS (2 from Task 5 + 3 here). If a FAIL appears, align `build_combined_model`'s raised messages with the asserted `match=` strings.

- [ ] **Step 3: Commit**

```bash
git add tests/test_single_z_combine.py
git commit -m "Stage 1: cover combine.py unimplemented/unknown modes"
```

---

## Task 7: Stage 1 verification sweep

**Files:** none (verification only).

- [ ] **Step 1: Run both new test files**

Run: `PYTHONPATH=src pytest tests/test_single_z_training_data.py tests/test_single_z_combine.py -v`
Expected: 10 PASS + 1 SKIP (gated regen smoke), or 11 PASS if the emulator is available.

- [ ] **Step 2: Run the full single-z test suite for regressions**

Run: `PYTHONPATH=src pytest tests/test_single_z_pipeline.py tests/test_single_z_training_data.py tests/test_single_z_combine.py -v`
Expected: all Stage A `test_single_z_pipeline.py` tests still PASS (no regression), plus the new tests.

- [ ] **Step 3: Confirm clean git state**

Run: `git status --short`
Expected: no uncommitted changes under `src/priya_forecast/single_z/`, `scripts/`, or `tests/` (the `.claude/` housekeeping noise may remain).

---

## Done criteria

Stage 1 is complete when:
- `single_z/training_data.py` round-trips raw-`P_F` 1pvar HDF5s and `regenerate_param` stacks per-z sweeps.
- `scripts/regen_1pvar.py` runs end to end on the emulator (gated smoke passes where the emulator is available).
- `single_z/combine.py` builds an `additive`/`local_anchored` `AdditiveTaylorModel` and rejects unimplemented/unknown modes.
- All new tests pass; no Stage A regression.

This unblocks **Stage 2 (Stage B `forecast_only`)**: `regenerate_param`/`load_1pvar` feed the refit's training data, and `build_combined_model` is the combine the forecast scores.
