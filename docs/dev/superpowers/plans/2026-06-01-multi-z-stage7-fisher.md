# Multi-z Stage 7 — joint multi-z Fisher forecast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a multi-z Fisher forecast (`F = Σ_z F(z)`) that combines information across redshift bins to lift the IGM-thermal rank-deficiency that wrecks single-z, with Stage 6's `log(P)` target.

**Architecture:** A new `src/priya_forecast/multi_z/` package mirroring `single_z/`, reusing the shared numerical blocks. The Fisher comes from **one z-spanning `KSDataLikelihood` + the existing `fisher_matrix`** (Approach A): the likelihood already loops `z_blocks` calling `model.predict(θ, k, z)` and stacks the joint data vector, so the returned Fisher *is* `Σ_z F(z)`. The legacy per-z-sum (`compute_fisher_F_phys` + `combine_fisher_phys_arrays`) is retained only as a cross-check oracle. Stages 1–6 `single_z/` code is untouched.

**Tech Stack:** Python 3.11, numpy<2, GPy 1.13.2, PySR/Julia, pytest. Reused blocks: `MultiZAdditiveTaylorModel` (`refit_taylor.py`), `refit_1d_multiz_for_param`/`compute_local_normalization_multiz` (`refit_1d_pysr.py`), `MultiZNormalizationSpec` (`models/normalization.py`), `fisher_matrix`/`compute_fisher_F_phys`/`combine_fisher_phys_arrays` (`fisher.py`), `KSDataLikelihood`.

**Spec:** `docs/superpowers/specs/2026-06-01-multi-z-stage7-fisher-design.md`.

---

## File structure

| File | Responsibility | New/Modify |
|------|----------------|------------|
| `src/priya_forecast/refit_taylor.py` | add `log_space` branch to `MultiZAdditiveTaylorModel` | Modify (`147–256`) |
| `src/priya_forecast/models/normalization.py` | `MultiZNormalizationSpec.save_npz` / `load_npz` | Modify (after `208`) |
| `src/priya_forecast/multi_z/__init__.py` | package marker | New |
| `src/priya_forecast/multi_z/config.py` | `MultiZPipelineConfig` + `load_config` | New |
| `src/priya_forecast/multi_z/combine.py` | `build_combined_model_multiz` | New |
| `src/priya_forecast/multi_z/forecast.py` | `run_three_fisher_multiz` (joint likelihood) + refit reconstruction | New |
| `src/priya_forecast/multi_z/refit.py` | `refit_one_param_multi_z` (writes CSV + norm sidecar) | New |
| `src/priya_forecast/multi_z/pipeline.py` | 3 mode fns + DISPATCH + output writers | New |
| `scripts/refit_one_param_multi_z.py` | one-param SLURM task | New |
| `scripts/run_pipeline_multi_z.py` | CLI entry | New |
| `slurm/multi_z_refit.slurm` | 11-param array | New |
| `configs/multi_z/stage7_z2.6-4.2.yaml` | production config | New |
| `tests/test_multi_z_*.py` | unit + integration + oracle tests | New |

**Key conventions carried from single-z (read these files before starting):**
- `src/priya_forecast/single_z/config.py` — config dataclass + `load_config` pattern to mirror.
- `src/priya_forecast/single_z/forecast.py` — `run_three_fisher`, `_build_likelihood`, `_fisher_for_likelihood`, `build_refit_from_pareto`, `_filter_fisher_safe` to mirror.
- `src/priya_forecast/single_z/pipeline.py` — `run_gp_only`, `run_forecast_only`, `run_refit_and_forecast`, `_write_forecast_deliverables`, `DISPATCH` to mirror.
- `src/priya_forecast/single_z/refit.py` + `scripts/refit_one_param_single_z.py` + `slurm/single_z_refit.slurm` — the refit/SLURM pattern to mirror.

**Test invocation:** `PYTHONPATH=src pytest tests/test_multi_z_*.py -q`. Slow/emulator tests are gated by env vars (mirror single-z: `RUN_SLOW_REFIT`, `RUN_SLOW_FORECAST_ONLY`, `RUN_SLOW_GP_ONLY`) and need `PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full` + `data/kodiaq_gp/`.

**Two verification points flagged for the integration tasks (not pre-verifiable without the HPC env):**
1. **Per-z k-grid uniformity.** `MultiZAdditiveTaylorModel` is built for one fixed `k_grid` and validates `np.allclose(k, self.k_grid)` in `predict`. Approach A passes each z-block's `kept_k[sl]` to the model. This works **iff** KODIAQ-SQUAD uses a common kf grid across z-blocks in the kept range. Task 7 asserts this and raises a clear error if violated.
2. **`log_space` positivity** on real data — covered by the gated end-to-end (Task 11).

---

## Task 1: `log_space` branch in `MultiZAdditiveTaylorModel`

The one piece of genuinely new numerics. Mirror the single-z `AdditiveTaylorModel` log branch (`refit_taylor.py:329–344` for caches, `382–412` for predict) into the per-z dict structure. `Refit1DResult.predict_log(theta_phys, k, resolution, z=...)` already accepts `z`.

**Files:**
- Modify: `src/priya_forecast/refit_taylor.py:147–256`
- Test: `tests/test_multi_z_log_combine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multi_z_log_combine.py
import numpy as np
import pytest
from priya_forecast.parameters import PARAM_NAMES, fiducial_vector
from priya_forecast.refit_taylor import MultiZAdditiveTaylorModel


class _StubGP:
    """Deterministic positive GP: P_F(θ,k,z) = (1 + 0.1*Σθ) * base(k,z)."""
    def __init__(self, k_grid, z_grid):
        self.k_grid = np.asarray(k_grid, float)
        self.z_grid = np.asarray(z_grid, float)
    def predict(self, theta, k, z):
        k = np.asarray(k, float)
        base = 1.0 + 0.5 * k + 0.05 * float(z)
        return (1.0 + 0.1 * float(np.sum(theta))) * base


def _model(log_space):
    k_grid = np.linspace(0.005, 0.04, 8)
    z_grid = np.array([3.4, 3.6])
    fid = np.asarray(fiducial_vector(), float)
    gp = _StubGP(k_grid, z_grid)
    refits = {n: None for n in PARAM_NAMES}   # all GP-slice fallback
    return MultiZAdditiveTaylorModel(
        gp=gp, fid=fid, refits=refits, k_grid=k_grid, z_grid=z_grid,
        log_space=log_space,
    ), k_grid, fid


def test_log_space_predict_at_fid_equals_gp_per_z():
    m, k_grid, fid = _model(log_space=True)
    for z in (3.4, 3.6):
        got = m.predict(fid, k_grid, z)
        want = m.gp.predict(fid, k_grid, z)
        np.testing.assert_allclose(got, want, rtol=1e-12)


def test_log_and_linear_agree_at_fid():
    ml, k_grid, fid = _model(log_space=True)
    mlin, _, _ = _model(log_space=False)
    for z in (3.4, 3.6):
        np.testing.assert_allclose(
            ml.predict(fid, k_grid, z), mlin.predict(fid, k_grid, z), rtol=1e-12)


def test_log_space_positivity_guard():
    k_grid = np.linspace(0.005, 0.04, 8)
    z_grid = np.array([3.6])
    fid = np.asarray(fiducial_vector(), float)
    class _NegGP:
        def predict(self, theta, k, z):
            return -1.0 * np.ones_like(np.asarray(k, float))
    with pytest.raises(ValueError, match="non-positive"):
        MultiZAdditiveTaylorModel(
            gp=_NegGP(), fid=fid, refits={n: None for n in PARAM_NAMES},
            k_grid=k_grid, z_grid=z_grid, log_space=True,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_multi_z_log_combine.py -q`
Expected: FAIL — `MultiZAdditiveTaylorModel.__init__() got an unexpected keyword argument 'log_space'`.

- [ ] **Step 3: Add the `log_space` field**

In `refit_taylor.py`, in the `MultiZAdditiveTaylorModel` dataclass body (after `z_grid: np.ndarray` at line ~175), add:

```python
    log_space: bool = False
```

- [ ] **Step 4: Build the log caches in `__post_init__`**

At the end of `MultiZAdditiveTaylorModel.__post_init__` (after the `_eq_at_fid_pf` loop, ~line 204), append:

```python
        if self.log_space:
            self._log_p_gp_fid_per_z: dict[float, np.ndarray] = {}
            for z in self.z_grid:
                pf = self._p_gp_fid_per_z[float(z)]
                if np.any(pf <= 0):
                    raise ValueError(
                        f"log_space combine: GP P_F(fid) at z={float(z)} has "
                        f"non-positive entries — cannot take log."
                    )
                self._log_p_gp_fid_per_z[float(z)] = np.log(pf)
            self._eq_at_fid_logpf: dict[tuple[str, float], np.ndarray] = {}
            for pname, r in self.refits.items():
                if r is None:
                    continue
                i = PARAM_NAMES.index(pname)
                fid_i_phys = float(self.fid[i])
                for z in self.z_grid:
                    self._eq_at_fid_logpf[(pname, float(z))] = r.predict_log(
                        theta_phys=fid_i_phys, k=self.k_grid,
                        resolution=HF_RESOLUTION_FOR_COMBINE, z=float(z),
                    )
```

- [ ] **Step 5: Add the log branch at the top of `predict`**

In `MultiZAdditiveTaylorModel.predict`, immediately after the `z_key`/`z_grid` validation (after line ~218, before `p_gp_fid = self._p_gp_fid_per_z[z_key]`), insert:

```python
        if self.log_space:
            out_log = self._log_p_gp_fid_per_z[z_key].copy()
            for pname, r in self.refits.items():
                if r is None:
                    continue
                i = PARAM_NAMES.index(pname)
                ti, fi = float(theta[i]), float(self.fid[i])
                if abs(ti - fi) <= max(abs(fi), 1.0) * 1e-12:
                    continue
                log_at_theta = r.predict_log(
                    theta_phys=ti, k=self.k_grid,
                    resolution=HF_RESOLUTION_FOR_COMBINE, z=z_key,
                )
                out_log = out_log + (log_at_theta - self._eq_at_fid_logpf[(pname, z_key)])
            for pname, r in self.refits.items():
                if r is not None:
                    continue
                i = PARAM_NAMES.index(pname)
                ti, fi = float(theta[i]), float(self.fid[i])
                if abs(ti - fi) <= max(abs(fi), 1.0) * 1e-12:
                    continue
                t_only = self.fid.copy()
                t_only[i] = theta[i]
                p_slice = np.asarray(
                    self.gp.predict(t_only, self.k_grid, z_key), dtype=float)
                if np.any(p_slice <= 0):
                    raise ValueError(
                        f"log_space combine: GP slice for {pname!r} at z={z_key} "
                        f"has non-positive P_F — cannot take log."
                    )
                out_log = out_log + (np.log(p_slice) - self._log_p_gp_fid_per_z[z_key])
            return np.exp(out_log)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_multi_z_log_combine.py -q`
Expected: 3 passed.

- [ ] **Step 7: Verify single-z regression untouched**

Run: `PYTHONPATH=src pytest tests/test_single_z_combine.py tests/ -q -k "taylor or combine"`
Expected: all pass (no behavior change to `AdditiveTaylorModel`).

- [ ] **Step 8: Commit**

```bash
git add src/priya_forecast/refit_taylor.py tests/test_multi_z_log_combine.py
git commit -m "Stage 7: log_space branch in MultiZAdditiveTaylorModel

Per-z _log_p_gp_fid_per_z + _eq_at_fid_logpf caches and an exp(Σ log-dev)
predict path with positivity guards, transcribed from single-z Stage 6.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `MultiZNormalizationSpec` save/load (the norm sidecar)

`refit_1d_multiz_for_param` returns a `Refit1DResult` carrying a `MultiZNormalizationSpec` that can't be cheaply recomputed at forecast time. Persist it next to the Pareto CSV so `forecast_only` reconstructs the refit exactly.

**Files:**
- Modify: `src/priya_forecast/models/normalization.py` (after line 208, inside the `MultiZNormalizationSpec` class)
- Test: `tests/test_multi_z_norm_io.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multi_z_norm_io.py
import numpy as np
from priya_forecast.models.normalization import MultiZNormalizationSpec


def _spec():
    z_grid = np.array([3.4, 3.6])
    k_grid = np.linspace(0.005, 0.04, 6)
    mean = np.outer(1.0 + 0.1 * z_grid, 1.0 + k_grid)
    std = 0.2 * np.ones((2, 6))
    return MultiZNormalizationSpec(
        param_min=0.0, param_max=1.0, k_min=float(k_grid.min()),
        k_max=float(k_grid.max()), z_grid=z_grid, mean_flux=mean,
        std_flux=std, k_grid=k_grid)


def test_norm_npz_roundtrip(tmp_path):
    spec = _spec()
    path = tmp_path / "norm_ns.npz"
    spec.save_npz(path)
    back = MultiZNormalizationSpec.load_npz(path)
    np.testing.assert_allclose(back.mean_flux, spec.mean_flux)
    np.testing.assert_allclose(back.std_flux, spec.std_flux)
    np.testing.assert_allclose(back.z_grid, spec.z_grid)
    np.testing.assert_allclose(back.k_grid, spec.k_grid)
    assert (back.param_min, back.param_max) == (spec.param_min, spec.param_max)
    assert (back.k_min, back.k_max) == (spec.k_min, spec.k_max)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_multi_z_norm_io.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'save_npz'`.

- [ ] **Step 3: Implement save/load**

Add these methods to `MultiZNormalizationSpec` (after `denormalize_flux`, ~line 208). Add `from pathlib import Path` to the imports at the top of the file if absent.

```python
    def save_npz(self, path) -> None:
        np.savez(
            path,
            param_min=self.param_min, param_max=self.param_max,
            k_min=self.k_min, k_max=self.k_max,
            z_grid=self.z_grid, mean_flux=self.mean_flux,
            std_flux=self.std_flux, k_grid=self.k_grid,
        )

    @classmethod
    def load_npz(cls, path) -> "MultiZNormalizationSpec":
        d = np.load(path)
        return cls(
            param_min=float(d["param_min"]), param_max=float(d["param_max"]),
            k_min=float(d["k_min"]), k_max=float(d["k_max"]),
            z_grid=d["z_grid"], mean_flux=d["mean_flux"],
            std_flux=d["std_flux"], k_grid=d["k_grid"],
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_multi_z_norm_io.py -q`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/models/normalization.py tests/test_multi_z_norm_io.py
git commit -m "Stage 7: MultiZNormalizationSpec save_npz/load_npz sidecar

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `multi_z/config.py` — `MultiZPipelineConfig`

Mirror `single_z/config.py`, replacing the scalar `redshift: float` with `z_min`/`z_max`. Reuse the single-z sub-dataclasses (`KRange`, `DataConfig`, `GPConfig`, `NormalizationConfig`, `PySRConfig`, `FisherConfig`, `ParetoCSVsConfig`) by importing them — do not re-define them.

**Files:**
- Create: `src/priya_forecast/multi_z/__init__.py` (empty)
- Create: `src/priya_forecast/multi_z/config.py`
- Test: `tests/test_multi_z_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multi_z_config.py
import pytest
from priya_forecast.multi_z.config import MultiZPipelineConfig


def test_defaults_valid():
    cfg = MultiZPipelineConfig(mode="gp_only")
    # gp.basedir validation is skipped here by pointing at an existing dir
    cfg.gp.basedir = "."
    cfg.validate()
    assert cfg.z_min == 2.6 and cfg.z_max == 4.2


def test_rejects_inverted_z_range():
    cfg = MultiZPipelineConfig(z_min=4.2, z_max=2.6)
    cfg.gp.basedir = "."
    with pytest.raises(ValueError, match="z_min"):
        cfg.validate()


def test_rejects_multi_d_combine_on_log():
    cfg = MultiZPipelineConfig(combine="multiplicative", target_space="log")
    cfg.gp.basedir = "."
    with pytest.raises(ValueError, match="log"):
        cfg.validate()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_multi_z_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'priya_forecast.multi_z'`.

- [ ] **Step 3: Create the package marker and config**

`src/priya_forecast/multi_z/__init__.py`:

```python
"""Multi-z Fisher forecast pipeline (Stage 7)."""
```

`src/priya_forecast/multi_z/config.py`:

```python
"""YAML schema for the multi-z forecast pipeline (Stage 7).

Mirrors single_z/config.py, replacing the scalar `redshift` with a
`z_min`/`z_max` range. The Fisher is computed on one z-spanning
KSDataLikelihood; the returned Fisher is F = Σ_z F(z).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from priya_forecast.parameters import PARAM_NAMES
from priya_forecast.single_z.config import (
    DataConfig, FisherConfig, GPConfig, KRange, NormalizationConfig,
    ParetoCSVsConfig, PySRConfig, VALID_COMBINES, VALID_DATA_SOURCES,
    VALID_MODES, VALID_PARETO_SOURCES, VALID_TARGET_SPACES, _build_pareto_entries,
    _is_valid_pick,
)


@dataclass
class MultiZPipelineConfig:
    mode: str = "forecast_only"
    z_min: float = 2.6
    z_max: float = 4.2
    output_dir: str = "results/multi_z_run/"
    parameters: list[str] = field(default_factory=lambda: list(PARAM_NAMES))
    k_range: KRange = field(default_factory=KRange)
    data: DataConfig = field(default_factory=DataConfig)
    gp: GPConfig = field(default_factory=GPConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    pysr: PySRConfig = field(default_factory=PySRConfig)
    combine: str = "additive"
    pick: str = "best_loss"
    target_space: str = "linear"
    fiducial_p1d_cache: str | None = None
    pareto_csvs: ParetoCSVsConfig = field(default_factory=ParetoCSVsConfig)
    fisher: FisherConfig = field(default_factory=FisherConfig)

    def validate(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}.")
        if not 2.2 <= self.z_min <= 4.6 or not 2.2 <= self.z_max <= 4.6:
            raise ValueError(f"z_min/z_max must lie in [2.2, 4.6].")
        if self.z_min > self.z_max:
            raise ValueError(f"z_min ({self.z_min}) must be <= z_max ({self.z_max}).")
        unknown = set(self.parameters) - set(PARAM_NAMES)
        if unknown:
            raise ValueError(f"Unknown PRIYA parameters: {sorted(unknown)}.")
        if self.combine not in VALID_COMBINES:
            raise ValueError(f"combine must be one of {VALID_COMBINES}.")
        if not _is_valid_pick(self.pick):
            raise ValueError(f"pick={self.pick!r} invalid.")
        if self.target_space not in VALID_TARGET_SPACES:
            raise ValueError(f"target_space must be one of {VALID_TARGET_SPACES}.")
        if self.target_space == "log" and self.combine != "additive":
            raise ValueError(
                "target_space='log' requires combine='additive' "
                "(log-space only supports the local_anchored combine)."
            )
        self.k_range.validate()
        self.data.validate()
        self.gp.validate()
        self.normalization.validate()
        self.pysr.validate()
        self.fisher.validate()
        self.pareto_csvs.validate(self.parameters)


def load_config(path: str | Path) -> MultiZPipelineConfig:
    """Load + validate a multi-z pipeline YAML."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = MultiZPipelineConfig()
    for key, value in raw.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown top-level key in config: {key!r}.")
        if key == "k_range":
            cfg.k_range = KRange(**value)
        elif key == "data":
            cfg.data = DataConfig(**value)
        elif key == "gp":
            cfg.gp = GPConfig(**value)
        elif key == "normalization":
            cfg.normalization = NormalizationConfig(**value)
        elif key == "pysr":
            cfg.pysr = PySRConfig(**value)
        elif key == "fisher":
            cfg.fisher = FisherConfig(**value)
        elif key == "pareto_csvs":
            entries = _build_pareto_entries(value.get("per_parameter", {}))
            cfg.pareto_csvs = ParetoCSVsConfig(
                source=value.get("source", "bundled_baseline"),
                per_parameter=entries,
            )
        else:
            setattr(cfg, key, value)
    cfg.validate()
    return cfg
```

> NOTE: confirm `_build_pareto_entries` and `_is_valid_pick` are importable from `single_z.config` (they are module-level functions there). If `single_z.config` does not export `VALID_MODES` etc. as module constants, they are — see `single_z/config.py:63–70`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_multi_z_config.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/multi_z/__init__.py src/priya_forecast/multi_z/config.py tests/test_multi_z_config.py
git commit -m "Stage 7: MultiZPipelineConfig (z_min/z_max) mirroring single-z

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `multi_z/combine.py` — `build_combined_model_multiz`

Thin wrapper constructing `MultiZAdditiveTaylorModel` on a discrete `z_grid`, mirroring `single_z/combine.py`.

**Files:**
- Create: `src/priya_forecast/multi_z/combine.py`
- Test: `tests/test_multi_z_combine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multi_z_combine.py
import numpy as np
import pytest
from priya_forecast.parameters import PARAM_NAMES, fiducial_vector
from priya_forecast.multi_z.combine import build_combined_model_multiz


class _StubGP:
    def predict(self, theta, k, z):
        k = np.asarray(k, float)
        return (1.0 + 0.1 * float(np.sum(theta))) * (1.0 + 0.5 * k + 0.05 * float(z))


def test_builds_and_predicts_per_z():
    k_grid = np.linspace(0.005, 0.04, 8)
    z_grid = np.array([3.4, 3.6])
    fid = np.asarray(fiducial_vector(), float)
    m = build_combined_model_multiz(
        combine_mode="additive", gp=_StubGP(), fid=fid,
        refits={n: None for n in PARAM_NAMES}, k_grid=k_grid, z_grid=z_grid,
        log_space=False)
    out = m.predict(fid, k_grid, 3.6)
    assert out.shape == k_grid.shape


def test_rejects_unimplemented_combine():
    with pytest.raises(NotImplementedError):
        build_combined_model_multiz(
            combine_mode="multiplicative", gp=_StubGP(),
            fid=np.asarray(fiducial_vector(), float),
            refits={n: None for n in PARAM_NAMES},
            k_grid=np.linspace(0.005, 0.04, 8), z_grid=np.array([3.6]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_multi_z_combine.py -q`
Expected: FAIL — `ModuleNotFoundError: ...multi_z.combine`.

- [ ] **Step 3: Implement**

```python
# src/priya_forecast/multi_z/combine.py
"""Combine per-parameter 4-input PySR equations into one multi-z model.

Thin wrapper over `refit_taylor.MultiZAdditiveTaylorModel` (always
local_anchored). Mirrors single_z/combine.py.
"""
from __future__ import annotations

import numpy as np

from priya_forecast.models.base import P1DModel
from priya_forecast.refit_taylor import MultiZAdditiveTaylorModel
from priya_forecast.single_z.config import VALID_COMBINES as VALID_COMBINE_MODES


def build_combined_model_multiz(
    *,
    combine_mode: str,
    gp: P1DModel,
    fid: np.ndarray,
    refits: dict,
    k_grid: np.ndarray,
    z_grid: np.ndarray,
    log_space: bool = False,
) -> P1DModel:
    """Construct the multi-z combined P_F(θ, k, z) model.

    Only `additive` is implemented; `multiplicative`/`joint` raise
    NotImplementedError (mirrors the single-z combine).
    """
    if combine_mode == "additive":
        return MultiZAdditiveTaylorModel(
            gp=gp, fid=np.asarray(fid, dtype=float), refits=refits,
            k_grid=np.asarray(k_grid, dtype=float),
            z_grid=np.asarray(z_grid, dtype=float), log_space=log_space,
        )
    if combine_mode in ("multiplicative", "joint"):
        raise NotImplementedError(
            f"combine mode {combine_mode!r} is not implemented; "
            f"only 'additive' is available."
        )
    raise ValueError(
        f"unknown combine mode {combine_mode!r}; expected one of {VALID_COMBINE_MODES}."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_multi_z_combine.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/multi_z/combine.py tests/test_multi_z_combine.py
git commit -m "Stage 7: build_combined_model_multiz wrapper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `multi_z/refit.py` — refit reconstruction + driver

Two responsibilities: (a) `refit_one_param_multi_z` — run `refit_1d_multiz_for_param` with seed-retry, write `pareto_<param>.csv` + `norm_<param>.npz`; (b) `build_refit_from_pareto_multiz` — reload both into a `Refit1DResult`. Mirror `single_z/refit.py` + `single_z/forecast.py:build_refit_from_pareto`.

**Files:**
- Create: `src/priya_forecast/multi_z/refit.py`
- Test: `tests/test_multi_z_refit_reconstruct.py`

- [ ] **Step 1: Write the failing test (reconstruction round-trip, no PySR)**

```python
# tests/test_multi_z_refit_reconstruct.py
import numpy as np
import pandas as pd
from priya_forecast.models.normalization import MultiZNormalizationSpec
from priya_forecast.multi_z.refit import build_refit_from_pareto_multiz


def _write_artifacts(tmp_path, param="ns"):
    # A trivial x0-dependent 4-input equation: P depends on x0 (θ_norm).
    df = pd.DataFrame({
        "Complexity": [1, 3],
        "Loss": [1.0, 0.01],
        "Equation": ["0.5", "x0 + x1 + 0.1 * x3"],
    })
    csv = tmp_path / "pareto_ns.csv"
    df.to_csv(csv, index=False)
    z_grid = np.array([3.4, 3.6])
    k_grid = np.linspace(0.005, 0.04, 6)
    spec = MultiZNormalizationSpec(
        param_min=0.0, param_max=2.0, k_min=float(k_grid.min()),
        k_max=float(k_grid.max()), z_grid=z_grid,
        mean_flux=np.outer(1.0 + 0.1 * z_grid, 1.0 + k_grid),
        std_flux=0.2 * np.ones((2, 6)), k_grid=k_grid)
    spec.save_npz(tmp_path / "norm_ns.npz")
    return csv, tmp_path / "norm_ns.npz"


def test_reconstruct_predicts_per_z(tmp_path):
    csv, norm = _write_artifacts(tmp_path)
    r = build_refit_from_pareto_multiz(
        param_name="ns", z_min=3.4, z_max=3.6, pareto_csv=csv,
        norm_npz=norm, pick_rule="best_loss")
    assert r.is_multiz
    out = r.predict(theta_phys=r.fid_value, k=np.linspace(0.005, 0.04, 6),
                    resolution=0.8, z=3.6)
    assert out.shape == (6,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_multi_z_refit_reconstruct.py -q`
Expected: FAIL — `ModuleNotFoundError: ...multi_z.refit`.

- [ ] **Step 3: Implement `multi_z/refit.py`**

```python
# src/priya_forecast/multi_z/refit.py
"""Multi-z PySR refit driver + Pareto-CSV reconstruction.

- refit_one_param_multi_z: run refit_1d_multiz_for_param with seed-retry,
  persist pareto_<param>.csv + norm_<param>.npz.
- build_refit_from_pareto_multiz: reload both into a 4-input Refit1DResult.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from priya_forecast.models.normalization import MultiZNormalizationSpec
from priya_forecast.models.pysr_model import load_pareto_csv, pick_equation
from priya_forecast.parameters import get_param
from priya_forecast.refit_1d_pysr import (
    HF_RESOLUTION, LF_RESOLUTION, Refit1DResult, refit_1d_multiz_for_param,
)
from priya_forecast.single_z.forecast import _filter_fisher_safe


def build_refit_from_pareto_multiz(
    *,
    param_name: str,
    z_min: float,
    z_max: float,
    pareto_csv,
    norm_npz,
    pick_rule: str,
) -> Refit1DResult:
    """Reconstruct a 4-input Refit1DResult from a Pareto CSV + norm sidecar."""
    df = load_pareto_csv(pareto_csv)
    # 4 inputs: x0=θ_norm, x1=k_norm, x2=resolution, x3=z_norm.
    safe = _filter_fisher_safe(df, n_features=4)
    if safe.empty:
        raise ValueError(
            f"No x0-dependent / Fisher-safe equation in Pareto front for "
            f"({param_name}, z∈[{z_min},{z_max}]): all {len(df)} rows unusable."
        )
    equation_str, complexity, loss = pick_equation(safe, pick_rule)
    norm = MultiZNormalizationSpec.load_npz(norm_npz)
    meta = get_param(param_name)
    z_center = float((z_min + z_max) / 2.0)
    return Refit1DResult(
        param_name=param_name, z=z_center, equation_str=equation_str,
        pareto_complexity=int(complexity), pareto_loss=float(loss),
        pareto_complexities=[int(c) for c in df["Complexity"]],
        pareto_losses=[float(x) for x in df["Loss"]],
        x_param_min=float(norm.param_min), x_param_max=float(norm.param_max),
        k_min=float(norm.k_min), k_max=float(norm.k_max),
        lf_resolution=LF_RESOLUTION, hf_resolution=HF_RESOLUTION,
        fid_value=float(meta.fid), norm=norm,
        k_grid=np.asarray(norm.k_grid, dtype=float),
        wall_time_s=0.0,
        lf_train_mean_rel_err=0.0, hf_train_mean_rel_err=0.0,
        lf_train_max_rel_err=0.0, hf_train_max_rel_err=0.0,
        z_min=float(z_min), z_max=float(z_max),
    )


def _write_pareto_csv(result: Refit1DResult, csv_path: Path) -> None:
    """Write a load_pareto_csv-compatible CSV from a Refit1DResult's front."""
    import pandas as pd
    # The picked equation is the min-loss row; persist the full front so
    # forecast-time pick rules (complexity_le, etc.) still work.
    pd.DataFrame({
        "Complexity": result.pareto_complexities,
        "Loss": result.pareto_losses,
        "Equation": [result.equation_str if c == result.pareto_complexity
                     else "" for c in result.pareto_complexities],
    }).to_csv(csv_path, index=False)


def refit_one_param_multi_z(
    *,
    param_name: str,
    z_min: float,
    z_max: float,
    cfg,
    gp_lf,
    gp_hf,
    k_grid: np.ndarray,
    out_dir: str | Path,
    n_total: int = 225,
    max_retries: int = 4,
) -> Refit1DResult:
    """Refit one parameter over [z_min, z_max]; write CSV + norm sidecar.

    Retries with bumped seeds until the front has an x0-dependent,
    Fisher-safe equation, or retries are exhausted.
    """
    from priya_forecast.single_z.refit import pysr_kwargs_for_cfg

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"pareto_{param_name}.csv"
    norm_path = out_dir / f"norm_{param_name}.npz"
    pysr_kwargs = pysr_kwargs_for_cfg(cfg)
    k_grid = np.asarray(k_grid, dtype=float)

    result = None
    for attempt in range(max_retries + 1):
        result = refit_1d_multiz_for_param(
            param_name=param_name, z_min=z_min, z_max=z_max, k_grid=k_grid,
            gp_lf=gp_lf, gp_hf=gp_hf, n_total=n_total,
            pysr_kwargs=pysr_kwargs, seed=cfg.pysr.seed + attempt,
        )
        _write_pareto_csv(result, csv_path)
        result.norm.save_npz(norm_path)
        safe = _filter_fisher_safe(load_pareto_csv(csv_path), n_features=4)
        if not safe.empty:
            return result
    return result
```

> NOTE on the full-front equation column: `refit_1d_multiz_for_param` returns only the *picked* equation string, not every row's equation. `_write_pareto_csv` therefore fills only the picked complexity's `Equation`; the other rows carry an empty string. `_filter_fisher_safe` drops empty/x0-free rows, so the picked row survives and is what `best_loss` reconstructs. If a richer pick rule (e.g. `complexity_le:N`) is needed later, extend `refit_1d_multiz_for_param` to return all equations — out of scope here.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_multi_z_refit_reconstruct.py -q`
Expected: 1 passed.

- [ ] **Step 5: Verify `pick_equation`/`load_pareto_csv` tolerate empty Equation rows**

The reconstruction test already exercises a 2-row front with one usable row. Confirm no crash on the empty-string row (it should be filtered by `_filter_fisher_safe`).

- [ ] **Step 6: Commit**

```bash
git add src/priya_forecast/multi_z/refit.py tests/test_multi_z_refit_reconstruct.py
git commit -m "Stage 7: multi-z refit driver + Pareto-CSV/norm reconstruction

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `multi_z/forecast.py` — joint-likelihood three-Fisher

The heart of Approach A. Build ONE `KSDataLikelihood(z_min, z_max)` per model, run the existing `fisher_matrix`. Mirror `single_z/forecast.py:run_three_fisher` + `_fisher_for_likelihood`, but the model is multi-z and the likelihood spans the range.

**Files:**
- Create: `src/priya_forecast/multi_z/forecast.py`
- Test: `tests/test_multi_z_forecast_joint.py` (gated — needs emulator + KSData)

- [ ] **Step 1: Write the gated integration test**

```python
# tests/test_multi_z_forecast_joint.py
import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_FORECAST_ONLY"),
    reason="needs lyaemu + data/kodiaq_gp; set RUN_SLOW_FORECAST_ONLY=1",
)


def test_perfect_1d_equals_gp_joint_linear_and_log():
    from priya_forecast.models.gp_model import GPModel
    from priya_forecast.parameters import fiducial_vector, PARAM_NAMES
    from priya_forecast.multi_z.config import MultiZPipelineConfig
    from priya_forecast.multi_z.forecast import run_three_fisher_multiz

    fid = np.asarray(fiducial_vector(), float)
    for space in ("linear", "log"):
        cfg = MultiZPipelineConfig(
            mode="forecast_only", z_min=3.4, z_max=3.6,
            parameters=["ns", "Ap", "tau0"], target_space=space,
        )
        cfg.gp.basedir = "data/kodiaq_gp"
        cfg.validate()
        gp = GPModel(basedir=cfg.gp.basedir)
        refits = {n: None for n in PARAM_NAMES}   # perfect_1D == GP
        res = run_three_fisher_multiz(cfg=cfg, gp=gp, fid=fid, refits=refits)
        np.testing.assert_allclose(
            res["perfect_1D"].sigma, res["GP"].sigma, rtol=1e-3,
            err_msg=f"perfect_1D != GP in {space} space",
        )
```

- [ ] **Step 2: Run to verify it fails (collection error, module missing)**

Run: `PYTHONPATH=src pytest tests/test_multi_z_forecast_joint.py -q`
Expected: FAIL at import — `ModuleNotFoundError: ...multi_z.forecast` (test body is skipped without the env, but the import at module top must resolve; keep the `from priya_forecast.multi_z.forecast import` *inside* the test so collection still works — it does above).

- [ ] **Step 3: Implement `multi_z/forecast.py`**

```python
# src/priya_forecast/multi_z/forecast.py
"""Multi-z forecast_only: joint KSDataLikelihood over [z_min, z_max].

Approach A: one z-spanning likelihood + the existing fisher_matrix.
The likelihood loops z_blocks calling model.predict(θ,k,z) and stacks the
joint data vector, so the returned Fisher is F = Σ_z F(z).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from priya_forecast.fisher import FisherResult, fisher_matrix
from priya_forecast.ksdata_likelihood import KSDataLikelihood
from priya_forecast.likelihood import GaussianLikelihood
from priya_forecast.parameters import PARAM_NAMES, PARAMS_11D
from priya_forecast.multi_z.combine import build_combined_model_multiz
from priya_forecast.multi_z.refit import build_refit_from_pareto_multiz


def _build_likelihood(cfg, model):
    if cfg.data.source == "kodiaq":
        return KSDataLikelihood(
            model=model, z_min=cfg.z_min, z_max=cfg.z_max,
            k_min=cfg.k_range.min, k_max=cfg.k_range.max,
            cov_scale=cfg.data.cov_scale, mock_data=cfg.data.mock_data,
            conservative=cfg.data.conservative,
        )
    raise NotImplementedError(
        "multi-z forecast currently supports data.source='kodiaq' only."
    )


def shared_k_and_z_grid(like) -> tuple[np.ndarray, np.ndarray]:
    """Return (k_grid, z_grid) from a KSDataLikelihood's z_blocks.

    Asserts every z-block shares the same k-grid (required because
    MultiZAdditiveTaylorModel is built for one fixed k_grid). Raises a
    clear error if KODIAQ uses a non-uniform per-z binning in range.
    """
    kept_k = np.asarray(like.kept_k, dtype=float)
    z_grid = np.array([zv for zv, _ in like.z_blocks], dtype=float)
    blocks = [kept_k[sl] for _, sl in like.z_blocks]
    k0 = blocks[0]
    for zv, kb in zip(z_grid, blocks):
        if kb.shape != k0.shape or not np.allclose(kb, k0):
            raise ValueError(
                f"KODIAQ k-grid differs across z-blocks (z={zv}); Approach A "
                f"requires a common per-z k-grid. Block k-shapes: "
                f"{[b.shape for b in blocks]}."
            )
    return k0, z_grid


def _fisher_for_likelihood(like, *, parameters, step_frac, rel_tol):
    indices = [PARAM_NAMES.index(n) for n in parameters]
    selected = tuple(PARAMS_11D[i] for i in indices)
    theta_fid_full = np.array([p.fid for p in PARAMS_11D], dtype=float)
    return fisher_matrix(
        likelihood=like, theta_fid=theta_fid_full, params=selected,
        step_frac=step_frac, rel_tol=rel_tol, param_indices=indices,
    )


def run_three_fisher_multiz(
    *, cfg, gp, fid: np.ndarray, refits: dict,
) -> dict[str, FisherResult]:
    """σ_GP, σ_perfect_1D, σ_PySR on the joint multi-z likelihood."""
    fid = np.asarray(fid, dtype=float)
    log_space = (cfg.target_space == "log")

    like_gp = _build_likelihood(cfg, gp)
    k_grid, z_grid = shared_k_and_z_grid(like_gp)

    none_refits = {n: None for n in PARAM_NAMES}
    perfect_model = build_combined_model_multiz(
        combine_mode=cfg.combine, gp=gp, fid=fid, refits=none_refits,
        k_grid=k_grid, z_grid=z_grid, log_space=log_space,
    )
    pysr_model = build_combined_model_multiz(
        combine_mode=cfg.combine, gp=gp, fid=fid, refits=refits,
        k_grid=k_grid, z_grid=z_grid, log_space=log_space,
    )
    like_perfect = _build_likelihood(cfg, perfect_model)
    like_pysr = _build_likelihood(cfg, pysr_model)

    common = dict(parameters=cfg.parameters,
                  step_frac=cfg.fisher.step_frac, rel_tol=cfg.fisher.rel_tol)
    return {
        "GP": _fisher_for_likelihood(like_gp, **common),
        "perfect_1D": _fisher_for_likelihood(like_perfect, **common),
        "PySR": _fisher_for_likelihood(like_pysr, **common),
    }


def resolve_refit_artifacts(cfg) -> dict[str, tuple[Path, Path]]:
    """Map each parameter to (pareto_csv, norm_npz) under <output_dir>/refit.

    Only the `from_refit` layout is supported for multi-z:
    <output_dir>/refit/z{z_min}-{z_max}/pareto_{param}.csv (+ norm_{param}.npz).
    Missing parameters are omitted (caller falls back to GP slice).
    """
    base = Path(cfg.output_dir) / "refit" / f"z{cfg.z_min}-{cfg.z_max}"
    out: dict[str, tuple[Path, Path]] = {}
    for param in cfg.parameters:
        csv = base / f"pareto_{param}.csv"
        norm = base / f"norm_{param}.npz"
        if csv.exists() and norm.exists():
            out[param] = (csv, norm)
    return out


def load_refits(cfg) -> tuple[dict, list[str]]:
    """Reconstruct refits from artifacts; return (refits, dropped)."""
    refits: dict = {n: None for n in PARAM_NAMES}
    dropped: list[str] = []
    for param, (csv, norm) in resolve_refit_artifacts(cfg).items():
        try:
            refits[param] = build_refit_from_pareto_multiz(
                param_name=param, z_min=cfg.z_min, z_max=cfg.z_max,
                pareto_csv=csv, norm_npz=norm, pick_rule=cfg.pick,
            )
        except ValueError as exc:
            dropped.append(param)
            print(f"[multi_z forecast] {param}: {exc} — GP-slice fallback.")
    return refits, dropped
```

- [ ] **Step 4: Run the gated test (only if HPC env available)**

Run: `PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full RUN_SLOW_FORECAST_ONLY=1 pytest tests/test_multi_z_forecast_joint.py -q`
Expected: PASS (perfect_1D ≈ GP in both spaces). If `shared_k_and_z_grid` raises the k-grid error, STOP and report — verification point #1 has failed and the architecture needs the per-z-k handling discussed in the spec risks.

Without the env: Run `PYTHONPATH=src pytest tests/test_multi_z_forecast_joint.py -q` → 1 skipped (collection succeeds).

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/multi_z/forecast.py tests/test_multi_z_forecast_joint.py
git commit -m "Stage 7: joint multi-z three-Fisher (Approach A)

One z-spanning KSDataLikelihood + existing fisher_matrix; shared_k_and_z_grid
asserts per-z k uniformity. perfect_1D == GP gated test (linear + log).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: A-vs-B oracle equivalence test

Pin Approach A against the legacy per-z-sum (`compute_fisher_F_phys` + `combine_fisher_phys_arrays`) on a block-diagonal covariance, using a stub GP + stub likelihood so it runs fast (no emulator). This is the cross-check the spec promised.

**Files:**
- Test: `tests/test_multi_z_A_equals_B.py`

- [ ] **Step 1: Inspect the oracle signatures**

Read `src/priya_forecast/fisher.py` for `compute_fisher_F_phys` (`153–196`) and `combine_fisher_phys_arrays` (`199–236`), and `scripts/multi_z_aggregate.py:278–308` for the per-z call pattern. Confirm `compute_fisher_F_phys` takes a `likelihood` and returns an un-inverted `F_phys`.

- [ ] **Step 2: Write the test**

```python
# tests/test_multi_z_A_equals_B.py
"""Approach A (joint Fisher) == Approach B (Σ_z F_phys) on block-diagonal cov.

Uses a GaussianLikelihood per z (block-diagonal by construction) so the two
formulations must agree to numerical precision.
"""
import numpy as np
import pytest
from priya_forecast.parameters import PARAMS_11D, PARAM_NAMES, fiducial_vector
from priya_forecast.fisher import (
    compute_fisher_F_phys, combine_fisher_phys_arrays, fisher_matrix,
)
from priya_forecast.likelihood import GaussianLikelihood


# A linear-in-θ analytic model keeps the Fisher exact under finite differences.
class _LinearModel:
    def __init__(self, k_grid, slopes):
        self.k_grid = np.asarray(k_grid, float)
        self.slopes = slopes   # (11,) per-param slope
    def predict(self, theta, k, z):
        k = np.asarray(k, float)
        base = (1.0 + 0.5 * k) * (1.0 + 0.05 * float(z))
        return base * (1.0 + float(np.dot(self.slopes, theta)))


def _per_z_like(model, z):
    return GaussianLikelihood(model=model, z=z, cov_scale=1.0, mock_data="gp")


@pytest.mark.skip(reason="enable once GaussianLikelihood multi-z stub confirmed")
def test_A_equals_B_block_diagonal():
    k_grid = np.linspace(0.005, 0.04, 8)
    slopes = 0.01 * np.arange(1, 12, dtype=float)
    model = _LinearModel(k_grid, slopes)
    params = PARAMS_11D[:3]
    idx = [0, 1, 2]
    theta_fid = np.asarray(fiducial_vector(), float)
    z_bins = [3.4, 3.6]

    # B: per-z F_phys summed.
    F_list = [compute_fisher_F_phys(
        likelihood=_per_z_like(model, z), theta_fid=theta_fid,
        params=params, param_indices=idx) for z in z_bins]
    fr_B = combine_fisher_phys_arrays(
        F_list, params=params, theta_fid=theta_fid)

    # A: a single likelihood whose data vector is both z-blocks stacked.
    # (Construct via the same GaussianLikelihood machinery over a 2-z grid,
    #  or assert F_A == sum(F_list) directly — see note.)
    fr_A_sigma = fr_B.sigma   # placeholder until joint stub wired
    np.testing.assert_allclose(fr_A_sigma, fr_B.sigma, rtol=1e-6)
```

> NOTE: This test is intentionally drafted with a `skip` and a placeholder because the exact joint-stub wiring depends on whether `GaussianLikelihood` accepts a multi-z grid (check its constructor). During execution, the engineer must: (1) inspect `GaussianLikelihood.__init__`; (2) if it supports a z-range/stacked grid, build the joint likelihood and replace `fr_A_sigma` with a real `fisher_matrix(...)` call; (3) if it does not, construct the joint `F_phys` as the block-diagonal stack by hand and compare. Remove the `skip` once wired. The mathematical claim under test — joint Fisher == Σ_z F_phys for block-diagonal covariance — is unconditional.

- [ ] **Step 3: Wire the joint side and remove skip**

Inspect `GaussianLikelihood.__init__` and `KSDataLikelihood` to choose the joint construction. Implement the real Approach-A Fisher and assert equality at `rtol=1e-6`.

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_multi_z_A_equals_B.py -q`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_multi_z_A_equals_B.py
git commit -m "Stage 7: A-vs-B Fisher equivalence test on block-diagonal cov

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: `multi_z/pipeline.py` — modes + dispatch + output writers

Mirror `single_z/pipeline.py`: `run_gp_only_multiz`, `run_forecast_only_multiz`, `run_refit_and_forecast_multiz`, `_write_forecast_deliverables_multiz`, `DISPATCH`, `run`. The GP-only path uses the joint likelihood directly; forecast paths call `run_three_fisher_multiz`.

**Files:**
- Create: `src/priya_forecast/multi_z/pipeline.py`
- Test: `tests/test_multi_z_pipeline.py` (gated for the GP-touching modes; non-gated dispatch/validation test)

- [ ] **Step 1: Write the non-gated dispatch test**

```python
# tests/test_multi_z_pipeline.py
import pytest
from priya_forecast.multi_z.pipeline import DISPATCH, run
from priya_forecast.multi_z.config import MultiZPipelineConfig


def test_dispatch_has_three_modes():
    assert set(DISPATCH) == {"gp_only", "forecast_only", "refit_and_forecast"}


def test_run_validates_before_dispatch():
    cfg = MultiZPipelineConfig(mode="gp_only", z_min=5.0, z_max=6.0)
    cfg.gp.basedir = "."
    with pytest.raises(ValueError, match="z_min/z_max"):
        run(cfg)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_multi_z_pipeline.py -q`
Expected: FAIL — `ModuleNotFoundError: ...multi_z.pipeline`.

- [ ] **Step 3: Implement `multi_z/pipeline.py`**

Mirror `single_z/pipeline.py` structure. Key differences from single-z, applied throughout:
- `cfg.redshift` → `cfg.z_min`/`cfg.z_max`; likelihood built with the range.
- `run_gp_only_multiz`: build GP, build joint `KSDataLikelihood(z_min, z_max)`, call `fisher_matrix` (copy `single_z/pipeline.py:run_gp_only` verbatim, swapping the two `z_min=/z_max=` lines and the header strings to show the range).
- `run_forecast_only_multiz`: build GP + `fid`, `refits, dropped = _fc.load_refits(cfg)`, `results = _fc.run_three_fisher_multiz(...)`, write deliverables.
- `run_refit_and_forecast_multiz`: build `gp_lf`/`gp_hf` on `k_grid`, loop `cfg.parameters` calling `multi_z.refit.refit_one_param_multi_z(param_name=p, z_min=cfg.z_min, z_max=cfg.z_max, cfg=cfg, gp_lf=..., gp_hf=..., k_grid=..., out_dir=<refit_dir>)`, build refits dict from results (keep if `_fc`'s `equation_uses_param` via single_z.forecast on the equation string; else drop), then forecast.
- `_write_forecast_deliverables_multiz`: copy `single_z/pipeline.py:_write_forecast_deliverables` (`138–202`), changing the header lines from `z={cfg.redshift}` to `z∈[{cfg.z_min},{cfg.z_max}]` and the title to "multi-z". Reuse `plot_fisher_corner` and `FisherResult.save_npz` unchanged.

Full code (the new/changed parts; copy the unchanged bodies from `single_z/pipeline.py`):

```python
"""Multi-z pipeline dispatcher (Stage 7). Mirrors single_z/pipeline.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from priya_forecast.fisher import fisher_matrix
from priya_forecast.ksdata_likelihood import KSDataLikelihood
from priya_forecast.models.gp_model import GPModel
from priya_forecast.parameters import PARAM_NAMES, PARAMS_11D, fiducial_vector
from priya_forecast.diagnostics.forecast_plots import plot_fisher_corner
from priya_forecast.multi_z.config import MultiZPipelineConfig
from priya_forecast.multi_z import forecast as _fc
from priya_forecast.multi_z import refit as _refit
from priya_forecast.single_z.forecast import equation_uses_param
from priya_forecast.single_z.refit import kodiaq_k_grid


def _build_gp(cfg):
    return GPModel(basedir=cfg.gp.basedir, hires_subdir=cfg.gp.hires_subdir)


def _selected_indices(cfg):
    return [PARAM_NAMES.index(n) for n in cfg.parameters]


def run_gp_only_multiz(cfg: MultiZPipelineConfig) -> dict:
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    gp = _build_gp(cfg)
    like = KSDataLikelihood(
        model=gp, z_min=cfg.z_min, z_max=cfg.z_max,
        k_min=cfg.k_range.min, k_max=cfg.k_range.max,
        cov_scale=cfg.data.cov_scale, mock_data=cfg.data.mock_data,
        conservative=cfg.data.conservative,
    )
    indices = _selected_indices(cfg)
    selected = tuple(PARAMS_11D[i] for i in indices)
    theta_fid_full = np.asarray(fiducial_vector(), dtype=float)
    fisher = fisher_matrix(
        likelihood=like, theta_fid=theta_fid_full, params=selected,
        step_frac=cfg.fisher.step_frac, rel_tol=cfg.fisher.rel_tol,
        param_indices=indices,
    )
    sigma = fisher.sigma
    table_path = out_dir / "forecast_table.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(f"# multi-z gp_only forecast z∈[{cfg.z_min},{cfg.z_max}]\n")
        f.write(f"# data={cfg.data.source} cov_scale={cfg.data.cov_scale}\n")
        f.write(f"# {'param':<12s} {'fid':>10s} {'sigma_GP':>12s} {'rel':>10s}\n")
        for i, p in enumerate(selected):
            f.write(f"  {p.name:<12s} {p.fid:>10.4g} {sigma[i]:>12.4g} "
                    f"{sigma[i]/abs(p.fid):>10.4f}\n")
    return {"sigma_gp": sigma, "fisher": fisher, "table_path": table_path,
            "selected_params": selected}


def _write_forecast_deliverables_multiz(cfg, out_dir, results, *,
                                        pysr_available, dropped=None):
    # Copy single_z/pipeline.py:_write_forecast_deliverables verbatim,
    # replacing `z={cfg.redshift}` with `z∈[{cfg.z_min},{cfg.z_max}]` in the
    # table header and scorecard, and the titles with "Multi-z". Body
    # identical (plot_fisher_corner, forecast_table.txt, scorecard.md,
    # fisher_{label}.npz via fr.save_npz).
    ...  # IMPLEMENT by copying the single-z body with the header swaps above


def run_forecast_only_multiz(cfg: MultiZPipelineConfig) -> dict:
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    gp = _build_gp(cfg)
    fid = np.asarray(fiducial_vector(), dtype=float)
    refits, dropped = _fc.load_refits(cfg)
    pysr_available = any(v is not None for v in refits.values())
    results = _fc.run_three_fisher_multiz(cfg=cfg, gp=gp, fid=fid, refits=refits)
    deliverables = _write_forecast_deliverables_multiz(
        cfg, out_dir, results, pysr_available=pysr_available, dropped=dropped)
    return {"sigmas": {k: fr.sigma for k, fr in results.items()},
            "fisher_results": results, "pysr_available": pysr_available,
            **deliverables}


def run_refit_and_forecast_multiz(cfg: MultiZPipelineConfig) -> dict:
    out_dir = Path(cfg.output_dir)
    refit_dir = out_dir / "refit" / f"z{cfg.z_min}-{cfg.z_max}"
    refit_dir.mkdir(parents=True, exist_ok=True)
    fid = np.asarray(fiducial_vector(), dtype=float)
    k_grid = kodiaq_k_grid(cfg.k_range.min, cfg.k_range.max, 48)
    gp_lf = GPModel(basedir=cfg.gp.basedir, fidelity="lf", kf=k_grid)
    gp_hf = GPModel(basedir=cfg.gp.basedir, fidelity="hf", kf=k_grid)
    refits = {n: None for n in PARAM_NAMES}
    dropped = []
    for param in cfg.parameters:
        result = _refit.refit_one_param_multi_z(
            param_name=param, z_min=cfg.z_min, z_max=cfg.z_max, cfg=cfg,
            gp_lf=gp_lf, gp_hf=gp_hf, k_grid=k_grid, out_dir=refit_dir,
        )
        if equation_uses_param(result.equation_str):
            refits[param] = result
        else:
            dropped.append(param)
            print(f"[multi_z refit] {param}: no x0 dependence — GP-slice fallback.")
    pysr_available = bool(cfg.parameters) and len(dropped) < len(cfg.parameters)
    results = _fc.run_three_fisher_multiz(cfg=cfg, gp=gp_hf, fid=fid, refits=refits)
    deliverables = _write_forecast_deliverables_multiz(
        cfg, out_dir, results, pysr_available=pysr_available, dropped=dropped)
    return {"sigmas": {k: fr.sigma for k, fr in results.items()},
            "fisher_results": results, "pysr_available": pysr_available,
            "refit_dir": refit_dir, **deliverables}


DISPATCH = {
    "gp_only": run_gp_only_multiz,
    "forecast_only": run_forecast_only_multiz,
    "refit_and_forecast": run_refit_and_forecast_multiz,
}


def run(cfg: MultiZPipelineConfig) -> dict:
    cfg.validate()
    return DISPATCH[cfg.mode](cfg)
```

- [ ] **Step 4: Implement `_write_forecast_deliverables_multiz` by copying the single-z body**

Open `single_z/pipeline.py:138–202`, copy the body into the `...` above, and apply the header/title swaps noted in the docstring. Keep `corner.png`, `forecast_table.txt`, `scorecard.md`, and `fisher_{label}.npz` filenames identical (so existing tooling transfers).

- [ ] **Step 5: Run the non-gated test**

Run: `PYTHONPATH=src pytest tests/test_multi_z_pipeline.py -q`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/priya_forecast/multi_z/pipeline.py tests/test_multi_z_pipeline.py
git commit -m "Stage 7: multi_z pipeline (gp_only/forecast_only/refit_and_forecast)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: CLI entry + SLURM

Mirror `scripts/refit_one_param_single_z.py` + `scripts/run_pipeline.py` + `slurm/single_z_refit.slurm`.

**Files:**
- Create: `scripts/run_pipeline_multi_z.py`
- Create: `scripts/refit_one_param_multi_z.py`
- Create: `slurm/multi_z_refit.slurm`
- Test: `tests/test_multi_z_cli_smoke.py` (argparse-only, no GP)

- [ ] **Step 1: Write the CLI smoke test**

```python
# tests/test_multi_z_cli_smoke.py
import subprocess, sys


def test_run_pipeline_multi_z_help():
    out = subprocess.run(
        [sys.executable, "scripts/run_pipeline_multi_z.py", "--help"],
        capture_output=True, text=True, env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0
    assert "--config" in out.stdout


def test_refit_one_param_multi_z_help():
    out = subprocess.run(
        [sys.executable, "scripts/refit_one_param_multi_z.py", "--help"],
        capture_output=True, text=True, env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0
    assert "--z-min" in out.stdout and "--z-max" in out.stdout
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_multi_z_cli_smoke.py -q`
Expected: FAIL — scripts don't exist (returncode != 0).

- [ ] **Step 3: Create `scripts/run_pipeline_multi_z.py`**

```python
#!/usr/bin/env python
"""Run the multi-z forecast pipeline from a YAML config."""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from priya_forecast.multi_z.config import load_config
from priya_forecast.multi_z.pipeline import run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    result = run(cfg)
    print(f"[multi_z] mode={cfg.mode} z∈[{cfg.z_min},{cfg.z_max}] -> {cfg.output_dir}")
    if "table_path" in result:
        print(f"  table: {result['table_path']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `scripts/refit_one_param_multi_z.py`**

Mirror `scripts/refit_one_param_single_z.py`, swapping `--z` for `--z-min`/`--z-max` and calling `multi_z.refit.refit_one_param_multi_z`. Build a `MultiZPipelineConfig` for `cfg.pysr` knobs.

```python
#!/usr/bin/env python
"""Refit one parameter over [z_min, z_max] with multi-z PySR; write CSV+norm."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from priya_forecast.parameters import PARAM_NAMES, PARAMS_11D
from priya_forecast.multi_z.config import MultiZPipelineConfig
from priya_forecast.single_z.config import GPConfig, PySRConfig
from priya_forecast.single_z.refit import kodiaq_k_grid
from priya_forecast.multi_z import refit as _refit


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--param", required=True, choices=list(PARAM_NAMES))
    p.add_argument("--z-min", type=float, required=True)
    p.add_argument("--z-max", type=float, required=True)
    p.add_argument("--basedir", default="data/kodiaq_gp")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--kmin", type=float, default=0.001)
    p.add_argument("--kmax", type=float, default=0.04)
    p.add_argument("--n-total", type=int, default=225)
    p.add_argument("--niterations", type=int, default=50)
    p.add_argument("--maxsize", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from priya_forecast.models.gp_model import GPModel
    cfg = MultiZPipelineConfig(
        mode="refit_and_forecast", z_min=args.z_min, z_max=args.z_max,
        output_dir=args.output_dir, gp=GPConfig(basedir=args.basedir),
        pysr=PySRConfig(niterations=args.niterations, maxsize=args.maxsize,
                        seed=args.seed),
    )
    k_grid = kodiaq_k_grid(args.kmin, args.kmax, 48)
    refit_dir = Path(args.output_dir) / "refit" / f"z{args.z_min}-{args.z_max}"
    fid = np.array([pp.fid for pp in PARAMS_11D], dtype=float)

    print(f"Loading emulators from {args.basedir} ...", flush=True)
    t0 = time.time()
    gp_lf = GPModel(basedir=args.basedir, fidelity="lf", kf=k_grid)
    gp_hf = GPModel(basedir=args.basedir, fidelity="hf", kf=k_grid)
    _ = gp_lf.predict(fid, k_grid, args.z_min)
    print(f"  loaded in {time.time()-t0:.0f}s.", flush=True)

    t0 = time.time()
    result = _refit.refit_one_param_multi_z(
        param_name=args.param, z_min=args.z_min, z_max=args.z_max, cfg=cfg,
        gp_lf=gp_lf, gp_hf=gp_hf, k_grid=k_grid, out_dir=refit_dir,
        n_total=args.n_total,
    )
    print(f"[{time.time()-t0:.0f}s] {args.param} z∈[{args.z_min},{args.z_max}] "
          f"-> {refit_dir}/pareto_{args.param}.csv "
          f"(complexity={result.pareto_complexity}, loss={result.pareto_loss:.4g})",
          flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create `slurm/multi_z_refit.slurm`**

Copy `slurm/single_z_refit.slurm`, swap `Z` → `Z_MIN`/`Z_MAX`, call `refit_one_param_multi_z.py`. Keep the env block (`PYTHON_JULIAPKG_PROJECT`, `JULIA_DEPOT_PATH`, `PYTHONPATH`) verbatim. Keep `--ntasks=4 --cpus-per-task=1 --time=1:00:00` (the estimate: ~20–40 min/param).

```bash
#!/bin/bash
# GreatLakes SLURM array — multi-z PySR refit, 11 params over [Z_MIN, Z_MAX].
#   sbatch --export=ALL,REPO=$(pwd),BASEDIR=data/kodiaq_gp,\
#          OUTPUT_DIR=results/multi_z_stage7,Z_MIN=2.6,Z_MAX=4.2 \
#          --array=0-10 slurm/multi_z_refit.slurm
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1:30:00
#SBATCH --job-name=mz_refit
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
export PYTHONUNBUFFERED=1
REPO=${REPO:?}; BASEDIR=${BASEDIR:?}; OUTPUT_DIR=${OUTPUT_DIR:?}
Z_MIN=${Z_MIN:?}; Z_MAX=${Z_MAX:?}
PARAMS=(dtau0 tau0 ns Ap herei heref alphaq hub omegamh2 hireionz bhfeedback)
PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]:?}
echo "[slurm] task=${SLURM_ARRAY_TASK_ID} param=${PARAM} z∈[${Z_MIN},${Z_MAX}]"
module purge
module load gcc/10.3.0 || true
PY=/sw/pkgs/arc/mamba/py3.11/bin/python
export PYTHON_JULIAPKG_PROJECT="${HOME}/.julia_env"
export JULIA_DEPOT_PATH="${HOME}/.julia"
export PYTHONPATH="/home/mfho/student_projects/lya_emulator_full:${REPO}/src"
cd "$REPO"
"$PY" scripts/refit_one_param_multi_z.py --param "$PARAM" \
    --z-min "$Z_MIN" --z-max "$Z_MAX" \
    --basedir "$BASEDIR" --output-dir "$OUTPUT_DIR"
echo "[slurm] ${PARAM} done"
```

- [ ] **Step 6: Run the smoke test**

Run: `PYTHONPATH=src pytest tests/test_multi_z_cli_smoke.py -q`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_pipeline_multi_z.py scripts/refit_one_param_multi_z.py slurm/multi_z_refit.slurm tests/test_multi_z_cli_smoke.py
git commit -m "Stage 7: multi-z CLI entries + SLURM array

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: One-param refit smoke calibration (HPC, replaces the estimate)

Before the full 11-param array, run ONE param to measure real wall time and confirm the refit→CSV→reconstruct→forecast loop works end-to-end on real data. This de-risks the CPU-hour estimate.

**Files:** none (operational); records results in the commit message / scorecard.

- [ ] **Step 1: Run one param with reduced niter (fast)**

```bash
cd /home/mfho/lya1d_priya_forecast
PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia \
PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full \
python scripts/refit_one_param_multi_z.py --param ns --z-min 3.4 --z-max 3.6 \
    --basedir data/kodiaq_gp --output-dir results/multi_z_stage7_smoke \
    --n-total 64 --niterations 20
```
Expected: completes; prints wall time; writes `results/multi_z_stage7_smoke/refit/z3.4-3.6/pareto_ns.csv` + `norm_ns.npz`.

- [ ] **Step 2: Record the measured wall time**

Note the `[Ns]` print. Extrapolate full-niter (50) / full-n_total (225) cost = measured × (50/20) × (225/64) ≈ measured × 8.8. Confirm it lands within the ~20–40 min/param estimate; if it's >2× over, reduce `n_total` or cores in the SLURM config before launching the array.

- [ ] **Step 3: Verify the gated forecast loads the smoke refit**

```bash
PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full RUN_SLOW_FORECAST_ONLY=1 \
  pytest tests/test_multi_z_forecast_joint.py -q
```
Expected: PASS (perfect_1D ≈ GP). This also exercises verification point #1 (per-z k uniformity).

- [ ] **Step 4: Commit the calibration note** (no code; record in HANDOFF or a note file)

```bash
git commit --allow-empty -m "Stage 7: one-param refit calibration — measured wall ~<N>min/param

Confirms the ~20-40 min/param estimate and the refit->forecast loop on
real KODIAQ. Full 11-array cleared to launch.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Production run + COMPARISON.md + gated end-to-end test

**Files:**
- Create: `configs/multi_z/stage7_z2.6-4.2.yaml`
- Create: `tests/test_multi_z_end_to_end.py` (gated `RUN_SLOW_REFIT`)
- Create (by running): `results/multi_z_stage7/{corner.png,scorecard.md,COMPARISON.md,fisher_*.npz,forecast_table.txt}`

- [ ] **Step 1: Write the production config**

```yaml
# configs/multi_z/stage7_z2.6-4.2.yaml
mode: forecast_only
z_min: 2.6
z_max: 4.2
output_dir: results/multi_z_stage7/
parameters: [ns, Ap, hub, omegamh2, herei, heref, alphaq, hireionz, bhfeedback, dtau0, tau0]
k_range: {min: 0.001, max: 0.04}
data: {source: kodiaq, cov_scale: 1.0, conservative: true, mock_data: gp}
gp: {basedir: data/kodiaq_gp}
combine: additive
target_space: log
pareto_csvs: {source: bundled_baseline}
fisher: {step_frac: 0.01, rel_tol: 0.01}
```

- [ ] **Step 2: Write the gated end-to-end test (2-bin range, refit_and_forecast)**

```python
# tests/test_multi_z_end_to_end.py
import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_REFIT"),
    reason="needs PySR/Julia + emulator; set RUN_SLOW_REFIT=1",
)


def test_refit_and_forecast_two_bins(tmp_path):
    from priya_forecast.multi_z.config import MultiZPipelineConfig
    from priya_forecast.multi_z.pipeline import run

    cfg = MultiZPipelineConfig(
        mode="refit_and_forecast", z_min=3.4, z_max=3.6,
        parameters=["ns", "Ap"], target_space="log",
        output_dir=str(tmp_path / "mz"),
    )
    cfg.gp.basedir = "data/kodiaq_gp"
    cfg.pysr.niterations = 20
    res = run(cfg)
    assert "GP" in res["fisher_results"]
    # perfect_1D must equal GP even with real refits present.
    np.testing.assert_allclose(
        res["fisher_results"]["perfect_1D"].sigma,
        res["fisher_results"]["GP"].sigma, rtol=1e-3)
    assert (tmp_path / "mz" / "corner.png").exists()
```

- [ ] **Step 3: Run the gated test (HPC)**

Run: `PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full RUN_SLOW_REFIT=1 pytest tests/test_multi_z_end_to_end.py -q`
Expected: PASS.

- [ ] **Step 4: Launch the full 11-param refit array (after Task 10 calibration)**

```bash
sbatch --export=ALL,REPO=$(pwd),BASEDIR=data/kodiaq_gp,\
OUTPUT_DIR=results/multi_z_stage7,Z_MIN=2.6,Z_MAX=4.2 \
--array=0-10 slurm/multi_z_refit.slurm
```
Wait for completion (~30–45 min wall). Confirm 11 `pareto_*.csv` + `norm_*.npz` pairs in `results/multi_z_stage7/refit/z2.6-4.2/`.

- [ ] **Step 5: Run the production forecast**

Point `pareto_csvs.source` at the fresh refits (set `output_dir: results/multi_z_stage7/` and `pareto_csvs.source` handling in `resolve_refit_artifacts` already reads `<output_dir>/refit/z{z_min}-{z_max}/`). Run:

```bash
PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full \
  python scripts/run_pipeline_multi_z.py --config configs/multi_z/stage7_z2.6-4.2.yaml
```
Expected: writes `results/multi_z_stage7/{corner.png,scorecard.md,forecast_table.txt,fisher_*.npz}`.

- [ ] **Step 6: Write `COMPARISON.md` (multi-z vs single-z Stage 6)**

Create `results/multi_z_stage7/COMPARISON.md` by hand from the two scorecards: tabulate σ_GP / σ_PySR per param for single-z z=3.6 (`results/single_z_stage6_log/scorecard.md`) vs multi-z, and report: (1) IGM-thermal σ_GP no longer NaN/huge; (2) mean |log10(σ_PySR/σ_GP)| delta; (3) sub-1 / deep-Mirage counts; (4) GP-slice fallback count.

- [ ] **Step 7: Commit**

```bash
git add configs/multi_z/stage7_z2.6-4.2.yaml tests/test_multi_z_end_to_end.py results/multi_z_stage7/COMPARISON.md
git commit -m "Stage 7: production config, gated end-to-end, multi-z vs single-z COMPARISON

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Full test sweep + HANDOFF refresh

- [ ] **Step 1: Run the full fast suite**

Run: `PYTHONPATH=src pytest tests/ -q`
Expected: all prior tests pass (the one pre-existing unrelated failure `test_h6_explicit_cross_terms_*` may persist — confirm it is the SAME failure, not new) + the new multi-z fast tests pass, gated ones skip.

- [ ] **Step 2: Update `HANDOFF.md`**

Mark Stage 7 done; record the production numbers (σ_GP IGM-thermal now finite, Mirage delta), the per-z k-uniformity verification result, and set Stage 8 (Sobolev loss) as next. Add the multi-z reproduction recipe (SLURM array + forecast command).

- [ ] **Step 3: Commit**

```bash
git add HANDOFF.md
git commit -m "Stage 7 done: multi-z joint Fisher; HANDOFF refreshed, Stage 8 next

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage:** §3 Approach A → Task 6; §3 B-oracle → Task 7; §4 package layout → Tasks 3–9; §5 log-space branch → Task 1; §5 three models → Task 6; §6 config/modes → Tasks 3, 8; §6.3 4-input Pareto CSV interchange → Tasks 5, 9 (+ norm sidecar refinement, Task 2); §7 outputs → Task 8 (writers) + Task 11 (COMPARISON); §8 tests 1–6 → Tasks 1, 6, 7, 11, 5 respectively; §9 risks (k-uniformity, fallback count, env) → Tasks 6, 8, 9, 10.
- **Verification points** (k-grid uniformity; log positivity on real data) are explicit STOP-and-report gates in Tasks 6 and 10–11, not silent assumptions.
- **Type consistency:** `run_three_fisher_multiz`, `build_combined_model_multiz`, `build_refit_from_pareto_multiz`, `refit_one_param_multi_z`, `MultiZPipelineConfig`, `shared_k_and_z_grid`, `load_refits`, `resolve_refit_artifacts` are used with consistent signatures across tasks.
- **Known soft spot:** Task 7 ships with a `skip` + placeholder that the engineer MUST wire against the real `GaussianLikelihood`/`KSDataLikelihood` joint construction (the math claim is unconditional; only the stub wiring is deferred to execution). Task 5's `_write_pareto_csv` persists only the picked equation row — adequate for `best_loss`, flagged for richer pick rules.
