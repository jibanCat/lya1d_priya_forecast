# Single-z Stage 2 (forecast_only) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Implement `forecast_only` mode — load PySR Pareto CSVs, pick + reconstruct one equation per parameter, combine them, and compute three Fisher forecasts (σ_GP, σ_perfect_1D, σ_PySR) with a corner plot.

**Architecture:** A new `single_z/forecast.py` module does the equation→`Refit1DResult` reconstruction and the three Fisher runs; `run_forecast_only` in `pipeline.py` orchestrates it and writes deliverables. The per-parameter normalization is rebuilt from the regenerated 1pvar data (Stage 1's `load_1pvar`) via a **shared helper** — `per_param_local_norm` — so Stage B's reconstructed norm is identical by construction to what Stage C trains with (Stage 3 will reuse the same helper). This closes the spec's loose Stage-C→B handoff.

**Tech Stack:** Python 3.11, numpy, pandas, sympy, matplotlib, pytest. Reuses Stage 1 (`training_data.load_1pvar`, `combine.build_combined_model`), `pysr_model.{load_pareto_csv,pick_equation}`, `pareto_filters.*`, `refit_1d_pysr.Refit1DResult`, `models.normalization.NormalizationSpec`, `fisher.fisher_matrix`, `diagnostics.forecast_plots.plot_fisher_corner`.

**Spec:** `docs/superpowers/specs/2026-05-18-single-z-stage-bc-design.md` §5.

**Branch:** `single_z_forecast_clean`. Test command: `PYTHONPATH=src pytest <file> -v`.

---

## Design notes (read before implementing)

- **σ_GP** — Fisher of the GP itself (identical to `run_gp_only`). Needs no equations.
- **σ_perfect_1D** — Fisher of the combined model built with `refits = {all 11 → None}`; `AdditiveTaylorModel` then uses GP 1D slices for every parameter. Measures the constraining power lost to the additive-combine *structure*. Needs no equations.
- **σ_PySR** — Fisher of the combined model built with the reconstructed `Refit1DResult`s. Needs Pareto CSVs.
- The combined model is built on the **likelihood's k-grid** (KSData: `like.kept_k`; eBOSS: `like.inputs.k_eboss`) and `cfg.redshift`, because `AdditiveTaylorModel.predict` requires the k/z it was built with.
- The Fisher matrix is data-independent (`(∂m)ᵀC⁻¹(∂m)`), so the three likelihoods differ only in `.model`; the covariance is identical.
- Pareto CSV equations are PySR `x0,x1,x2` form (x0=θ_norm, x1=k_norm, x2=resolution). They are evaluated via `Refit1DResult.predict`, NOT `compile_equation` (which is the 2-input θ,k form). Stage B reconstructs a `Refit1DResult` directly.

## File Structure

| File | Responsibility |
|------|----------------|
| `src/priya_forecast/single_z/config.py` | flip `combine` default to `additive`; add top-level `pick` default |
| `src/priya_forecast/single_z/forecast.py` | NEW — norm helper, CSV→`Refit1DResult` reconstruction, source resolution, the 3 Fisher runs |
| `src/priya_forecast/single_z/pipeline.py` | implement `run_forecast_only` + deliverable writers |
| `src/priya_forecast/diagnostics/forecast_plots.py` | add `plot_pareto_front` (loss–complexity diagnostic) |
| `tests/test_single_z_forecast.py` | NEW — unit tests for `forecast.py` |
| `tests/test_single_z_pipeline.py` | append `forecast_only` config + gated end-to-end tests |

---

## Task 1: config — additive default + top-level `pick`

**Files:**
- Modify: `src/priya_forecast/single_z/config.py`
- Modify: `tests/test_single_z_pipeline.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_single_z_pipeline.py`:

```python
def test_combine_defaults_to_additive(tmp_path: Path):
    """forecast_only's combine default is additive (the student contract)."""
    basedir = _basedir(tmp_path)
    p = _write(tmp_path, "c.yaml", f"gp:\n  basedir: {basedir}\n")
    cfg = load_config(p)
    assert cfg.combine == "additive"


def test_top_level_pick_default_and_validation(tmp_path: Path):
    """PipelineConfig has a top-level `pick` rule, default best_loss; bad rule rejected."""
    basedir = _basedir(tmp_path)
    good = _write(tmp_path, "g.yaml", f"gp:\n  basedir: {basedir}\n")
    assert load_config(good).pick == "best_loss"
    bad = _write(tmp_path, "b.yaml",
                 f"pick: nonsense\ngp:\n  basedir: {basedir}\n")
    with pytest.raises(ValueError, match="pick"):
        load_config(bad)
```

- [ ] **Step 2: Run, expect FAIL** — `PYTHONPATH=src pytest tests/test_single_z_pipeline.py -k "combine_defaults or top_level_pick" -v` → FAIL.

- [ ] **Step 3: Implement** — in `config.py`:
  - Change `PipelineConfig.combine` default from `"multiplicative"` to `"additive"` (the field line `combine: str = "multiplicative"` → `combine: str = "additive"`).
  - Add a field to `PipelineConfig` after `combine`: `pick: str = "best_loss"`.
  - In `PipelineConfig.validate()`, after the `combine` check, add:
    ```python
        if not _is_valid_pick(self.pick):
            raise ValueError(
                f"pick={self.pick!r} invalid. "
                f"Valid: best_loss / complexity_le:N / accuracy_at:tol / row:I."
            )
    ```
  - In the YAML loader (`load_config`), ensure the top-level `pick:` key is read into `PipelineConfig.pick`. Find where `combine` is parsed from the YAML dict and add `pick` alongside it identically.

- [ ] **Step 4: Run, expect PASS** — `PYTHONPATH=src pytest tests/test_single_z_pipeline.py -k "combine_defaults or top_level_pick" -v`. Also run the whole file to confirm no regression: `PYTHONPATH=src pytest tests/test_single_z_pipeline.py -q` — note `test_shipped_example_yaml_loads_and_validates` asserts `cfg.combine == "multiplicative"`; update that assertion to `"additive"` IF the shipped `configs/single_z/example.yaml` does not pin `combine:` explicitly. If it does pin it, leave the test and instead change `example.yaml`'s `combine:` line to `additive`.

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/single_z/config.py tests/test_single_z_pipeline.py configs/single_z/example.yaml
git commit -m "Stage 2: combine defaults to additive + top-level pick rule"
```

---

## Task 2: `per_param_local_norm` helper

**Files:**
- Create: `src/priya_forecast/single_z/forecast.py`
- Create: `tests/test_single_z_forecast.py`

- [ ] **Step 1: Write the failing test** — create `tests/test_single_z_forecast.py`:

```python
"""Unit tests for `priya_forecast.single_z.forecast`."""

from __future__ import annotations

import numpy as np
import pytest

from priya_forecast.single_z.forecast import per_param_local_norm


def test_per_param_local_norm_shapes_and_values():
    """Per-k mean/std of an LF flux sweep → a valid NormalizationSpec."""
    rng = np.random.default_rng(0)
    n_points, n_k = 50, 12
    k_grid = np.linspace(0.001, 0.04, n_k)
    flux_lf = rng.random((n_points, n_k)) + 1.0  # strictly positive
    norm = per_param_local_norm(
        flux_lf_z=flux_lf, k_grid=k_grid, param_min=0.8, param_max=1.05,
    )
    assert norm.mean_flux.shape == (n_k,)
    assert norm.std_flux.shape == (n_k,)
    assert np.all(norm.std_flux > 0)
    np.testing.assert_allclose(norm.mean_flux, flux_lf.mean(axis=0))
    np.testing.assert_allclose(norm.k_grid, k_grid)
    assert norm.param_min == 0.8
    assert norm.param_max == 1.05
    assert norm.k_min == pytest.approx(0.001)
    assert norm.k_max == pytest.approx(0.04)


def test_per_param_local_norm_degenerate_std_floored():
    """A k-bin with zero variance must not produce std=0 (NormalizationSpec rejects it)."""
    k_grid = np.linspace(0.001, 0.04, 5)
    flux_lf = np.ones((10, 5)) * 3.0  # zero variance everywhere
    norm = per_param_local_norm(
        flux_lf_z=flux_lf, k_grid=k_grid, param_min=0.0, param_max=1.0,
    )
    assert np.all(norm.std_flux > 0)
```

- [ ] **Step 2: Run, expect FAIL** — `PYTHONPATH=src pytest tests/test_single_z_forecast.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement** — create `src/priya_forecast/single_z/forecast.py`:

```python
"""forecast_only mode: Pareto CSVs → equations → combined model → Fisher.

This module reconstructs a `Refit1DResult` per parameter from a picked Pareto
equation plus the regenerated 1pvar training data, builds the combined model,
and runs the three Fisher forecasts (σ_GP, σ_perfect_1D, σ_PySR).
"""

from __future__ import annotations

import numpy as np

from priya_forecast.models.normalization import NormalizationSpec


def per_param_local_norm(
    *,
    flux_lf_z: np.ndarray,
    k_grid: np.ndarray,
    param_min: float,
    param_max: float,
) -> NormalizationSpec:
    """Per-parameter local normalization from a 1pvar LF flux sweep.

    The `local_anchored` combine normalizes each per-param equation with the
    per-k mean/std of that parameter's own LF training flux. Computing it here
    — from the same regenerated 1pvar data Stage C trains on — guarantees the
    forecast-time norm matches the train-time norm.

    Parameters
    ----------
    flux_lf_z : (n_points, n_k) — LF P_F sweep at one z-bin.
    k_grid : (n_k,) — strictly increasing k-grid.
    param_min, param_max : the parameter's prior bounds.
    """
    flux_lf_z = np.asarray(flux_lf_z, dtype=float)
    k_grid = np.asarray(k_grid, dtype=float)
    mean_flux = flux_lf_z.mean(axis=0)
    std_flux = flux_lf_z.std(axis=0, ddof=0)
    std_flux = np.where(std_flux > 0, std_flux, 1.0)
    return NormalizationSpec(
        param_min=float(param_min),
        param_max=float(param_max),
        k_min=float(k_grid.min()),
        k_max=float(k_grid.max()),
        mean_flux=mean_flux,
        std_flux=std_flux,
        k_grid=k_grid,
    )
```

- [ ] **Step 4: Run, expect PASS** — `PYTHONPATH=src pytest tests/test_single_z_forecast.py -v`.

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/single_z/forecast.py tests/test_single_z_forecast.py
git commit -m "Stage 2: per_param_local_norm helper"
```

---

## Task 3: `build_refit_from_pareto` — CSV → Refit1DResult

**Files:**
- Modify: `src/priya_forecast/single_z/forecast.py`
- Modify: `tests/test_single_z_forecast.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_single_z_forecast.py`:

```python
def test_build_refit_from_pareto(tmp_path):
    """A Pareto CSV + regenerated 1pvar data → a usable Refit1DResult."""
    import pandas as pd

    from priya_forecast.single_z.training_data import write_1pvar_hdf5
    from priya_forecast.single_z.forecast import build_refit_from_pareto

    # synthetic 1pvar data for param 'ns' across 3 z-bins
    n_points, n_z, n_k = 50, 3, 8
    k_grid = np.linspace(0.001, 0.04, n_k)
    kfkms = np.broadcast_to(k_grid, (n_points, n_z, n_k)).copy()
    rng = np.random.default_rng(1)
    flux = rng.random((n_points, n_z, n_k)) + 1.0
    params = np.tile(np.array([p.fid for p in __import__(
        "priya_forecast.parameters", fromlist=["PARAMS_11D"]).PARAMS_11D]),
        (n_points, 1))
    zout = np.array([3.2, 3.4, 3.6])
    for fid in ("lf", "hf"):
        write_1pvar_hdf5(tmp_path / f"{fid}_ns_npoints50.hdf5",
                         params=params, kfkms=kfkms, flux_vectors=flux, zout=zout)

    # a minimal Pareto CSV: a safe linear equation in x0 (θ_norm)
    csv = tmp_path / "pareto_ns.csv"
    pd.DataFrame({
        "Complexity": [1, 3, 5],
        "Loss": [1.0, 0.1, 0.05],
        "Equation": ["x0", "x0 + x1", "x0 + x1 + 0.1*x2"],
    }).to_csv(csv, index=False)

    refit = build_refit_from_pareto(
        param_name="ns", z=3.6, pareto_csv=csv, pick_rule="best_loss",
        data_1pvar_dir=tmp_path,
    )
    assert refit.param_name == "ns"
    assert refit.z == 3.6
    # best_loss picks the min-Loss row that survives the safety filter
    assert refit.equation_str in {"x0 + x1", "x0 + x1 + 0.1*x2"}
    # the reconstructed result evaluates without error
    pred = refit.predict(theta_phys=0.98, k=k_grid)
    assert pred.shape == k_grid.shape
    assert np.all(np.isfinite(pred))


def test_build_refit_from_pareto_all_filtered_raises(tmp_path):
    """If every Pareto row is Fisher-pathological, fail loud naming the param."""
    import pandas as pd

    from priya_forecast.single_z.training_data import write_1pvar_hdf5
    from priya_forecast.single_z.forecast import build_refit_from_pareto
    from priya_forecast.parameters import PARAMS_11D

    n_points, n_z, n_k = 50, 1, 6
    k_grid = np.linspace(0.001, 0.04, n_k)
    kfkms = np.broadcast_to(k_grid, (n_points, n_z, n_k)).copy()
    flux = np.ones((n_points, n_z, n_k)) + 1.0
    params = np.tile(np.array([p.fid for p in PARAMS_11D]), (n_points, 1))
    for fid in ("lf", "hf"):
        write_1pvar_hdf5(tmp_path / f"{fid}_ns_npoints50.hdf5",
                         params=params, kfkms=kfkms, flux_vectors=flux,
                         zout=np.array([3.6]))
    csv = tmp_path / "pareto_ns.csv"
    # equations with a huge pathological constant — all filtered out
    pd.DataFrame({
        "Complexity": [3], "Loss": [0.01],
        "Equation": ["x0 + 1e9"],
    }).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="ns"):
        build_refit_from_pareto(
            param_name="ns", z=3.6, pareto_csv=csv, pick_rule="best_loss",
            data_1pvar_dir=tmp_path,
        )
```

- [ ] **Step 2: Run, expect FAIL** — `PYTHONPATH=src pytest tests/test_single_z_forecast.py -k build_refit -v`.

- [ ] **Step 3: Implement** — append to `src/priya_forecast/single_z/forecast.py` (add imports at the top of the file):

```python
from pathlib import Path

from priya_forecast.models.pysr_model import load_pareto_csv, pick_equation
from priya_forecast.pareto_filters import (
    has_pathological_constant,
    is_eq_well_behaved,
    is_fisher_stencil_safe,
)
from priya_forecast.parameters import get_param
from priya_forecast.refit_1d_pysr import HF_RESOLUTION, LF_RESOLUTION, Refit1DResult
from priya_forecast.single_z.training_data import load_1pvar


def _filter_fisher_safe(df, n_features: int):
    """Drop Fisher-pathological Pareto rows; return the surviving sub-frame.

    Mirrors `scripts/refit_one_param.py`: an equation is kept only if it has
    no pathological constant and is Fisher-stencil-safe. `is_eq_well_behaved`
    needs a training matrix, so here we use the lighter stencil + constant
    guards, which are the two that protect Fisher conditioning.
    """
    eq = df["Equation"].astype(str)
    pathological = eq.apply(has_pathological_constant)
    stencil_safe = eq.apply(
        lambda s: is_fisher_stencil_safe(s, n_features=n_features)
    )
    return df[(~pathological) & stencil_safe].reset_index(drop=True)


def build_refit_from_pareto(
    *,
    param_name: str,
    z: float,
    pareto_csv: str | Path,
    pick_rule: str,
    data_1pvar_dir: str | Path,
) -> Refit1DResult:
    """Reconstruct a `Refit1DResult` from a Pareto CSV + regenerated 1pvar data.

    Filter-then-pick: drop Fisher-pathological rows, then apply `pick_rule`.
    The per-parameter normalization comes from `per_param_local_norm` on the
    regenerated LF flux — identical to what Stage C trains with.
    """
    df = load_pareto_csv(pareto_csv)
    # PySR equations here have 3 inputs (x0=θ_norm, x1=k_norm, x2=resolution).
    safe = _filter_fisher_safe(df, n_features=3)
    if safe.empty:
        raise ValueError(
            f"No Fisher-safe equation in Pareto front for ({param_name}, z={z}): "
            f"all {len(df)} rows were pathological or stencil-unsafe."
        )
    equation_str, complexity, loss = pick_equation(safe, pick_rule)

    d = load_1pvar(param_name=param_name, z=z, data_dir=data_1pvar_dir)
    k_grid = d["kfkms_lf_z"][0]
    meta = get_param(param_name)
    norm = per_param_local_norm(
        flux_lf_z=d["flux_lf_z"], k_grid=k_grid,
        param_min=float(meta.prior[0]), param_max=float(meta.prior[1]),
    )
    return Refit1DResult(
        param_name=param_name,
        z=float(z),
        equation_str=equation_str,
        pareto_complexity=int(complexity),
        pareto_loss=float(loss),
        pareto_complexities=[int(c) for c in df["Complexity"]],
        pareto_losses=[float(x) for x in df["Loss"]],
        x_param_min=float(meta.prior[0]),
        x_param_max=float(meta.prior[1]),
        k_min=float(k_grid.min()),
        k_max=float(k_grid.max()),
        lf_resolution=LF_RESOLUTION,
        hf_resolution=HF_RESOLUTION,
        fid_value=float(meta.fid),
        norm=norm,
        k_grid=np.asarray(k_grid, dtype=float),
        wall_time_s=0.0,
        lf_train_mean_rel_err=0.0,
        hf_train_mean_rel_err=0.0,
        lf_train_max_rel_err=0.0,
        hf_train_max_rel_err=0.0,
    )
```

- [ ] **Step 4: Run, expect PASS** — `PYTHONPATH=src pytest tests/test_single_z_forecast.py -k build_refit -v`. If the first test's `best_loss` pick lands on a row the safety filter dropped, that is correct behavior — adjust the assertion only if it contradicts the filter, never weaken the filter.

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/single_z/forecast.py tests/test_single_z_forecast.py
git commit -m "Stage 2: reconstruct Refit1DResult from Pareto CSV (filter-then-pick)"
```

---

## Task 4: `resolve_pareto_csvs` — the three sources

**Files:**
- Modify: `src/priya_forecast/single_z/forecast.py`
- Modify: `tests/test_single_z_forecast.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_single_z_forecast.py`:

```python
def test_resolve_pareto_csvs_per_parameter(tmp_path):
    from priya_forecast.single_z.config import (
        PipelineConfig, ParetoCSVsConfig, ParetoEntry,
    )
    from priya_forecast.single_z.forecast import resolve_pareto_csvs

    csv = tmp_path / "ns.csv"
    csv.write_text("Complexity,Loss,Equation\n1,0.1,x0\n")
    cfg = PipelineConfig(
        mode="forecast_only", parameters=["ns"],
        pareto_csvs=ParetoCSVsConfig(
            source="per_parameter",
            per_parameter={"ns": ParetoEntry(pareto_csv=str(csv))},
        ),
    )
    paths = resolve_pareto_csvs(cfg)
    assert paths["ns"] == csv


def test_resolve_pareto_csvs_from_refit(tmp_path):
    from priya_forecast.single_z.config import PipelineConfig, ParetoCSVsConfig
    from priya_forecast.single_z.forecast import resolve_pareto_csvs

    refit_dir = tmp_path / "out" / "refit" / "z3.6"
    refit_dir.mkdir(parents=True)
    (refit_dir / "pareto_ns.csv").write_text("Complexity,Loss,Equation\n1,0.1,x0\n")
    cfg = PipelineConfig(
        mode="forecast_only", redshift=3.6, parameters=["ns"],
        output_dir=str(tmp_path / "out"),
        pareto_csvs=ParetoCSVsConfig(source="from_refit"),
    )
    paths = resolve_pareto_csvs(cfg)
    assert paths["ns"] == refit_dir / "pareto_ns.csv"


def test_resolve_pareto_csvs_missing_raises(tmp_path):
    from priya_forecast.single_z.config import PipelineConfig, ParetoCSVsConfig
    from priya_forecast.single_z.forecast import resolve_pareto_csvs

    cfg = PipelineConfig(
        mode="forecast_only", redshift=3.6, parameters=["ns"],
        output_dir=str(tmp_path / "out"),
        pareto_csvs=ParetoCSVsConfig(source="from_refit"),
    )
    with pytest.raises(FileNotFoundError, match="ns"):
        resolve_pareto_csvs(cfg)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** — append to `forecast.py` (add `from priya_forecast.single_z.config import PipelineConfig` to imports):

```python
# Vendored baseline Pareto CSVs (populated once Stage C produces them).
_BUNDLED_BASELINE_DIR = (
    Path(__file__).resolve().parents[2]
    / "priya_forecast" / "_vendored" / "data" / "pareto_baseline"
)


def resolve_pareto_csvs(cfg: PipelineConfig) -> dict[str, Path]:
    """Map each selected parameter to its Pareto-CSV path, per `pareto_csvs.source`.

    - `per_parameter` → the path in each `ParetoEntry`.
    - `from_refit`    → `<output_dir>/refit/z{z}/pareto_{param}.csv`.
    - `bundled_baseline` → the vendored `_vendored/data/pareto_baseline/z{z}/`.

    Raises FileNotFoundError naming the parameter if a CSV is absent.
    """
    src = cfg.pareto_csvs.source
    z_tag = f"z{cfg.redshift}"
    out: dict[str, Path] = {}
    for param in cfg.parameters:
        if src == "per_parameter":
            entry = cfg.pareto_csvs.per_parameter[param]
            path = Path(entry.pareto_csv)
        elif src == "from_refit":
            path = Path(cfg.output_dir) / "refit" / z_tag / f"pareto_{param}.csv"
        elif src == "bundled_baseline":
            path = _BUNDLED_BASELINE_DIR / z_tag / f"pareto_{param}.csv"
        else:  # pragma: no cover - config.validate already guards this
            raise ValueError(f"unknown pareto_csvs.source {src!r}.")
        if not path.exists():
            raise FileNotFoundError(
                f"Pareto CSV for parameter {param!r} not found at {path} "
                f"(pareto_csvs.source={src!r})."
            )
        out[param] = path
    return out
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/single_z/forecast.py tests/test_single_z_forecast.py
git commit -m "Stage 2: resolve Pareto-CSV paths for the three sources"
```

---

## Task 5: `run_three_fisher` — σ_GP, σ_perfect_1D, σ_PySR

**Files:**
- Modify: `src/priya_forecast/single_z/forecast.py`
- Modify: `tests/test_single_z_forecast.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_single_z_forecast.py`:

```python
def test_run_three_fisher_with_mock_gp():
    """run_three_fisher returns 3 FisherResults; refits=None path == perfect_1D."""
    from priya_forecast.models.gp_model import MockGPModel
    from priya_forecast.parameters import PARAM_NAMES, fiducial_vector
    from priya_forecast.fisher import FisherResult
    from priya_forecast.single_z.forecast import run_three_fisher

    gp = MockGPModel()
    fid = np.asarray(fiducial_vector(), dtype=float)
    results = run_three_fisher(
        gp=gp, fid=fid, refits={n: None for n in PARAM_NAMES},
        parameters=["ns", "Ap"], redshift=3.6,
        k_range=(0.001, 0.04), combine_mode="additive",
        step_frac=0.05, rel_tol=0.05,
    )
    assert set(results) == {"GP", "perfect_1D", "PySR"}
    for label, fr in results.items():
        assert isinstance(fr, FisherResult)
        assert fr.sigma.shape == (2,)
        assert np.all(np.isfinite(fr.sigma))
        assert np.all(fr.sigma > 0)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** — append to `forecast.py`. Add imports: `from priya_forecast.fisher import FisherResult, fisher_matrix`, `from priya_forecast.likelihood import GaussianLikelihood`, `from priya_forecast.parameters import PARAM_NAMES, PARAMS_11D`, `from priya_forecast.single_z.combine import build_combined_model`.

```python
def _fisher_for_model(model, *, parameters, redshift, k_range, step_frac, rel_tol):
    """Run `fisher_matrix` for a forward `model` over a parameter subset.

    Uses a single-z `GaussianLikelihood` (eBOSS covariance) so this works
    without the KSData/lyaemu dependency. `run_forecast_only` (Task 6) uses
    the KSData likelihood when `cfg.data.source == 'kodiaq'`; the Fisher call
    pattern is identical.
    """
    like = GaussianLikelihood(model=model, z=redshift)
    indices = [PARAM_NAMES.index(n) for n in parameters]
    selected = tuple(PARAMS_11D[i] for i in indices)
    theta_fid_full = np.array([p.fid for p in PARAMS_11D], dtype=float)
    return fisher_matrix(
        likelihood=like, theta_fid=theta_fid_full, params=selected,
        step_frac=step_frac, rel_tol=rel_tol, param_indices=indices,
    )


def run_three_fisher(
    *,
    gp,
    fid: np.ndarray,
    refits: dict,
    parameters: list[str],
    redshift: float,
    k_range: tuple[float, float],
    combine_mode: str,
    step_frac: float = 0.01,
    rel_tol: float = 0.01,
) -> dict[str, FisherResult]:
    """Compute σ_GP, σ_perfect_1D, σ_PySR as a dict of FisherResults.

    - GP         : Fisher of the raw GP emulator.
    - perfect_1D : combine built with all-None refits (GP 1D-slice fallback).
    - PySR       : combine built with the reconstructed `refits`.
    """
    k_grid = np.linspace(k_range[0], k_range[1], 48)
    fid = np.asarray(fid, dtype=float)
    none_refits = {n: None for n in PARAM_NAMES}
    perfect_model = build_combined_model(
        combine_mode=combine_mode, gp=gp, fid=fid, refits=none_refits,
        k_grid=k_grid, z=redshift,
    )
    pysr_model = build_combined_model(
        combine_mode=combine_mode, gp=gp, fid=fid, refits=refits,
        k_grid=k_grid, z=redshift,
    )
    common = dict(parameters=parameters, redshift=redshift, k_range=k_range,
                  step_frac=step_frac, rel_tol=rel_tol)
    return {
        "GP": _fisher_for_model(gp, **common),
        "perfect_1D": _fisher_for_model(perfect_model, **common),
        "PySR": _fisher_for_model(pysr_model, **common),
    }
```

Note: `GaussianLikelihood` bins the model onto its own eBOSS k-grid via `model.predict`. `AdditiveTaylorModel.predict` requires the exact k_grid it was built with — so `_fisher_for_model` for the combined models must pass `GaussianLikelihood(model=model, z=redshift, k_grid=model.k_grid)`. Update `_fisher_for_model` to accept and forward an optional `k_grid` and have `run_three_fisher` pass `k_grid=k_grid` for the two combined models (and leave it `None` for the raw GP). If `GaussianLikelihood` does not accept a `k_grid` kwarg in this codebase, instead build the combined models on the likelihood's native k-grid: construct the GP likelihood first, read `like.inputs.k_eboss`, and build the combine on that. Pick whichever the actual `GaussianLikelihood.__init__` supports — it has a `k_grid` parameter per the spec extraction, so prefer passing `k_grid`.

- [ ] **Step 4: Run, expect PASS** — `PYTHONPATH=src pytest tests/test_single_z_forecast.py -k three_fisher -v`. If the combined-model k-grid mismatch raises, apply the k_grid fix described in the note above and re-run.

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/single_z/forecast.py tests/test_single_z_forecast.py
git commit -m "Stage 2: run_three_fisher (sigma_GP / perfect_1D / PySR)"
```

---

## Task 6: `run_forecast_only` + deliverables

**Files:**
- Modify: `src/priya_forecast/single_z/pipeline.py`
- Modify: `tests/test_single_z_pipeline.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_single_z_pipeline.py`:

```python
RUN_SLOW_FORECAST = os.environ.get("RUN_SLOW_FORECAST_ONLY") == "1"


@pytest.mark.skipif(
    not (RUN_SLOW_FORECAST and LYAEMU_AVAILABLE and GP_BASEDIR.exists()),
    reason="gated on RUN_SLOW_FORECAST_ONLY=1 + lyaemu + data/kodiaq_gp/",
)
def test_forecast_only_perfect_1d_end_to_end(tmp_path: Path):
    """forecast_only with no equations still yields σ_GP and σ_perfect_1D."""
    import numpy as np
    from priya_forecast.single_z.pipeline import run

    cfg = PipelineConfig(
        mode="forecast_only", redshift=3.6,
        output_dir=str(tmp_path / "out"),
        gp=GPConfig(basedir=str(GP_BASEDIR)),
        parameters=["ns", "Ap"],
        k_range=KRange(min=0.001, max=0.04),
        data=DataConfig(source="eboss_dr14"),
    )
    result = run(cfg)
    for label in ("GP", "perfect_1D"):
        s = result["sigmas"][label]
        assert s.shape == (2,)
        assert np.all(np.isfinite(s)) and np.all(s > 0)
    assert (tmp_path / "out" / "forecast_table.txt").exists()
    assert (tmp_path / "out" / "corner.png").exists()
```

- [ ] **Step 2: Run, expect SKIP** — `PYTHONPATH=src pytest tests/test_single_z_pipeline.py::test_forecast_only_perfect_1d_end_to_end -v` → SKIPPED.

- [ ] **Step 3: Implement `run_forecast_only`** — in `src/priya_forecast/single_z/pipeline.py`, replace the `run_forecast_only` stub. Add to the imports at the top: `from priya_forecast.single_z import forecast as _fc`, `from priya_forecast.diagnostics.forecast_plots import plot_fisher_corner`. Implementation:

```python
def run_forecast_only(cfg: PipelineConfig) -> dict:
    """Student CSVs → equations → combined model → σ_GP / σ_perfect_1D / σ_PySR.

    σ_GP and σ_perfect_1D need no equations. σ_PySR needs Pareto CSVs; if none
    are available the run still emits σ_GP and σ_perfect_1D and notes σ_PySR
    as unavailable in the scorecard.
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gp = _build_gp(cfg)
    fid = np.asarray(fiducial_vector(), dtype=float)

    # Reconstruct per-parameter refits from Pareto CSVs if available.
    refits: dict = {name: None for name in PARAM_NAMES}
    pysr_available = True
    try:
        csv_paths = _fc.resolve_pareto_csvs(cfg)
        data_dir = "data/single_z_1pvar"
        for param, csv in csv_paths.items():
            refits[param] = _fc.build_refit_from_pareto(
                param_name=param, z=cfg.redshift, pareto_csv=csv,
                pick_rule=cfg.pick, data_1pvar_dir=data_dir,
            )
    except FileNotFoundError:
        pysr_available = False

    results = _fc.run_three_fisher(
        gp=gp, fid=fid, refits=refits, parameters=cfg.parameters,
        redshift=cfg.redshift, k_range=(cfg.k_range.min, cfg.k_range.max),
        combine_mode=cfg.combine, step_frac=cfg.fisher.step_frac,
        rel_tol=cfg.fisher.rel_tol,
    )

    sigmas = {label: fr.sigma for label, fr in results.items()}
    corner_labels = ["GP", "perfect_1D"] + (["PySR"] if pysr_available else [])
    corner_path = out_dir / "corner.png"
    plot_fisher_corner(
        fisher_results={lab: results[lab] for lab in corner_labels},
        outpath=corner_path,
        param_subset=cfg.parameters[: min(5, len(cfg.parameters))],
    )

    table_path = out_dir / "forecast_table.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(f"# single-z forecast_only at z={cfg.redshift}\n")
        f.write(f"# combine={cfg.combine}  pysr_equations={'yes' if pysr_available else 'NONE'}\n")
        f.write(f"# {'param':<12s} {'sigma_GP':>12s} {'sigma_perf1D':>14s} {'sigma_PySR':>12s}\n")
        for i, name in enumerate(cfg.parameters):
            sp = f"{sigmas['PySR'][i]:>12.4g}" if pysr_available else f"{'n/a':>12s}"
            f.write(f"  {name:<12s} {sigmas['GP'][i]:>12.4g} "
                    f"{sigmas['perfect_1D'][i]:>14.4g} {sp}\n")

    scorecard_path = out_dir / "scorecard.md"
    with open(scorecard_path, "w", encoding="utf-8") as f:
        f.write(f"# Single-z forecast scorecard — forecast_only\n\n")
        f.write(f"- z = {cfg.redshift}\n")
        f.write(f"- combine = {cfg.combine}\n")
        f.write(f"- PySR equations: {'available' if pysr_available else 'NOT available — σ_PySR omitted'}\n\n")
        f.write(f"## σ per parameter\n\n")
        f.write(f"| param | σ_GP | σ_perfect_1D | σ_PySR |\n|---|---|---|---|\n")
        for i, name in enumerate(cfg.parameters):
            sp = f"{sigmas['PySR'][i]:.4g}" if pysr_available else "n/a"
            f.write(f"| {name} | {sigmas['GP'][i]:.4g} | "
                    f"{sigmas['perfect_1D'][i]:.4g} | {sp} |\n")

    return {
        "sigmas": sigmas,
        "fisher_results": results,
        "pysr_available": pysr_available,
        "table_path": table_path,
        "scorecard_path": scorecard_path,
        "corner_path": corner_path,
    }
```

- [ ] **Step 4: Run** — `PYTHONPATH=src pytest tests/test_single_z_pipeline.py -q` (the new gated test SKIPs cleanly; no regression). If `lyaemu` + `data/kodiaq_gp/` are available, run `RUN_SLOW_FORECAST_ONLY=1 PYTHONPATH=src:/home/mfho/lya_emulator_full pytest tests/test_single_z_pipeline.py::test_forecast_only_perfect_1d_end_to_end -v` and confirm PASS.

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/single_z/pipeline.py tests/test_single_z_pipeline.py
git commit -m "Stage 2: run_forecast_only — three Fisher forecasts + deliverables"
```

---

## Task 7: Stage 2 verification sweep

**Files:** none (verification only).

- [ ] **Step 1** — `PYTHONPATH=src pytest tests/test_single_z_forecast.py tests/test_single_z_combine.py tests/test_single_z_training_data.py tests/test_single_z_pipeline.py -v`. Expect all pure tests PASS, gated tests SKIP, no regression.

- [ ] **Step 2** — confirm `git status --short` shows no stray uncommitted changes under `src/`, `tests/`.

- [ ] **Step 3** — if `lyaemu` is available, run the gated `forecast_only` end-to-end test and capture the σ_GP / σ_perfect_1D numbers; record them in the commit message of a final verification commit (or report them).

---

## Done criteria

- `forecast_only` runs end to end: σ_GP and σ_perfect_1D always; σ_PySR when Pareto CSVs exist.
- `combine` defaults to `additive`; the top-level `pick` rule is wired.
- Deliverables written: `forecast_table.txt`, `scorecard.md`, `corner.png`.
- All new tests pass; no Stage 1 / Stage A regression.

Carryover to Stage 3: `build_refit_from_pareto` and `per_param_local_norm` are the reconstruction path; Stage C must write `pareto_{param}.csv` files into `<output_dir>/refit/z{z}/` so `forecast_only` with `source=from_refit` consumes them. The loss–complexity diagnostic plot (`plot_pareto_front`) is deferred to Stage 3, where the Pareto fronts are actually produced.
