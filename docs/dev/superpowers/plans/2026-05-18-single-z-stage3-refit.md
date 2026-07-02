# Single-z Stage 3 (refit_and_forecast) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox (`- [ ]`) steps.

**Goal:** Implement `refit_and_forecast` — run a single-z PySR refit per parameter from the emulator, emit Pareto CSVs, then forecast (σ_GP / σ_perfect_1D / σ_PySR).

**Architecture:** The existing `refit_1d_pysr.refit_1d_for_param` already does the single-z refit correctly (inline 1pvar generation + `compute_local_normalization`, which is byte-identical to Stage 2's `per_param_local_norm`). Stage 3 adds (1) optional Pareto-CSV emission to `refit_1d_for_param`, (2) a thin `single_z/refit.py` wrapper, (3) `run_refit_and_forecast` looping 11 params then forecasting, (4) a SLURM array script + CLI for the batched all-z-bins case.

**Tech Stack:** Python 3.11, numpy, pandas, PySR (Julia). Reuses `refit_1d_pysr.{refit_1d_for_param, SMART_REFIT_PYSR_KWARGS, DEFAULT_PYSR_KWARGS}`, `models.gp_model.GPModel`, `single_z.forecast.run_three_fisher`.

**Spec:** `docs/superpowers/specs/2026-05-18-single-z-stage-bc-design.md` §4.

**Branch:** `single_z_forecast_clean`. Test command: `PYTHONPATH=src pytest <file> -v`. PySR tests are gated behind `RUN_SLOW_REFIT=1` (Julia cold-start is slow).

---

## Design notes

- `refit_1d_for_param(*, param_name, z, k_grid, gp_lf, gp_hf, norm=None, pysr_kwargs, seed)` with `norm=None` computes the per-param local norm via `compute_local_normalization` — the SAME `flux_lf_z.std(axis=0, ddof=0)` / `flux_lf_z.mean(axis=0)` formula as `single_z.forecast.per_param_local_norm`. So the refit's bundled norm and Stage 2's reconstructed norm agree.
- `run_refit_and_forecast` feeds the `Refit1DResult`s **directly** to `run_three_fisher` (no CSV round-trip needed for the forecast). The Pareto CSVs are emitted as an inspection/reuse artifact.
- k-grid for the refit: `np.geomspace(cfg.k_range.min, cfg.k_range.max, 48)` — the same grid `scripts/regen_1pvar.py` used.
- PySR kwargs: `SMART_REFIT_PYSR_KWARGS` when `cfg.pysr.smart_kwargs` (default), else `DEFAULT_PYSR_KWARGS`; `niterations/maxsize/populations/procs` overridden from `cfg.pysr`. (`SMART_REFIT_PYSR_KWARGS` carries the ANOVA loss — that is the documented "smart" behavior the user asked for.)

## File Structure

| File | Responsibility |
|------|----------------|
| `src/priya_forecast/refit_1d_pysr.py` | add optional `pareto_csv_out` to `refit_1d_for_param` |
| `src/priya_forecast/single_z/refit.py` | NEW — `kodiaq_k_grid`, `pysr_kwargs_for_cfg`, `refit_one_param_single_z` |
| `src/priya_forecast/single_z/pipeline.py` | extract `_write_forecast_deliverables`; implement `run_refit_and_forecast` |
| `scripts/refit_one_param_single_z.py` | NEW — CLI: refit one (param, z) |
| `slurm/single_z_refit.slurm` | NEW — array job, 11 params, parametrized by Z |
| `tests/test_single_z_refit.py` | NEW — unit tests + gated PySR smoke |

---

## Task 1: `refit_1d_for_param` — optional Pareto-CSV emission

**Files:**
- Modify: `src/priya_forecast/refit_1d_pysr.py`
- Modify: `tests/test_refit_1d_pysr.py` (or create `tests/test_single_z_refit.py` if no such file — check first)

- [ ] **Step 1: Read** `refit_1d_for_param` in `src/priya_forecast/refit_1d_pysr.py` (around lines 895-1000) to see the full body — where `model.equations_` is available and where the function returns.

- [ ] **Step 2: Implement** — add a keyword-only parameter `pareto_csv_out: str | Path | None = None` to `refit_1d_for_param`'s signature (after `seed`). After `model.fit(...)` and after `pareto = model.equations_` is available, before the function returns, add:

```python
    if pareto_csv_out is not None:
        pareto_csv_out = Path(pareto_csv_out)
        pareto_csv_out.parent.mkdir(parents=True, exist_ok=True)
        # PySR's equations_ has lowercase columns; write them capitalized so
        # `load_pareto_csv` reads them without case coercion.
        _pareto_out = model.equations_.rename(
            columns={"complexity": "Complexity", "loss": "Loss",
                     "equation": "Equation"}
        )
        _pareto_out.to_csv(pareto_csv_out, index=False)
```

Update the docstring to mention `pareto_csv_out`.

- [ ] **Step 3: Write a focused test** — append to the test file a test of just the CSV-writing contract using a tiny fake `equations_` frame, to avoid a PySR run. Add:

```python
def test_pareto_csv_out_is_load_pareto_csv_compatible(tmp_path):
    """A frame written via the pareto_csv_out path round-trips through load_pareto_csv."""
    import pandas as pd
    from priya_forecast.models.pysr_model import load_pareto_csv

    # Mimic the rename + to_csv the pareto_csv_out branch performs.
    eqs = pd.DataFrame({"complexity": [1, 3], "loss": [0.5, 0.1],
                        "equation": ["x0", "x0 + x1"]})
    out = tmp_path / "pareto_ns.csv"
    eqs.rename(columns={"complexity": "Complexity", "loss": "Loss",
                        "equation": "Equation"}).to_csv(out, index=False)
    df = load_pareto_csv(out)
    assert list(df.columns[:3]) == ["Complexity", "Loss", "Equation"]
    assert len(df) == 2
```

- [ ] **Step 4: Run** — `PYTHONPATH=src pytest <testfile>::test_pareto_csv_out_is_load_pareto_csv_compatible -v` → PASS. Also run the existing `refit_1d_pysr` tests if any, to confirm the new optional arg breaks nothing.

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/refit_1d_pysr.py <testfile>
git commit -m "Stage 3: refit_1d_for_param can emit a Pareto CSV"
```

---

## Task 2: `single_z/refit.py` — refit wrapper

**Files:**
- Create: `src/priya_forecast/single_z/refit.py`
- Create: `tests/test_single_z_refit.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_single_z_refit.py`:

```python
"""Unit tests for `priya_forecast.single_z.refit`."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from priya_forecast.single_z.refit import kodiaq_k_grid, pysr_kwargs_for_cfg


def test_kodiaq_k_grid_is_log_spaced_in_range():
    k = kodiaq_k_grid(0.001, 0.04, 48)
    assert k.shape == (48,)
    assert k[0] == pytest.approx(0.001)
    assert k[-1] == pytest.approx(0.04)
    # log-spaced → constant ratio between consecutive points
    ratios = k[1:] / k[:-1]
    assert np.allclose(ratios, ratios[0])


def test_pysr_kwargs_for_cfg_smart_and_default():
    from priya_forecast.single_z.config import PipelineConfig, PySRConfig

    smart = pysr_kwargs_for_cfg(PipelineConfig(
        pysr=PySRConfig(smart_kwargs=True, niterations=7, maxsize=15)))
    assert smart["niterations"] == 7
    assert smart["maxsize"] == 15
    # SMART restricts unary operators to exp/log/square
    assert set(smart["unary_operators"]) == {"exp", "log", "square"}

    plain = pysr_kwargs_for_cfg(PipelineConfig(
        pysr=PySRConfig(smart_kwargs=False, niterations=9)))
    assert plain["niterations"] == 9
    # DEFAULT keeps the wider operator set
    assert "sqrt" in plain["unary_operators"]
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** — create `src/priya_forecast/single_z/refit.py`:

```python
"""refit_and_forecast mode: single-z PySR refit per parameter.

Thin wrapper over `refit_1d_pysr.refit_1d_for_param` (inline 1pvar path).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from priya_forecast.refit_1d_pysr import (
    DEFAULT_PYSR_KWARGS,
    SMART_REFIT_PYSR_KWARGS,
    refit_1d_for_param,
)
from priya_forecast.single_z.config import PipelineConfig


def kodiaq_k_grid(kmin: float, kmax: float, nk: int = 48) -> np.ndarray:
    """Log-spaced k-grid (s/km) — the grid the regen + refit share."""
    return np.geomspace(kmin, kmax, nk)


def pysr_kwargs_for_cfg(cfg: PipelineConfig) -> dict:
    """Assemble the PySR kwargs dict from `cfg.pysr`.

    `smart_kwargs` selects SMART (restricted operators + ANOVA loss) vs the
    default operator set; the search-budget fields are taken from `cfg.pysr`.
    """
    base = dict(
        SMART_REFIT_PYSR_KWARGS if cfg.pysr.smart_kwargs else DEFAULT_PYSR_KWARGS
    )
    base["niterations"] = cfg.pysr.niterations
    base["maxsize"] = cfg.pysr.maxsize
    base["populations"] = cfg.pysr.populations
    base["procs"] = cfg.pysr.procs
    return base


def refit_one_param_single_z(
    *,
    param_name: str,
    z: float,
    cfg: PipelineConfig,
    gp_lf,
    gp_hf,
    k_grid: np.ndarray,
    out_dir: str | Path,
):
    """Refit one parameter at one z-bin; write `pareto_{param}.csv`.

    Returns the `Refit1DResult`. `out_dir` is `<output_dir>/refit/z{z}/`.
    """
    out_dir = Path(out_dir)
    return refit_1d_for_param(
        param_name=param_name,
        z=z,
        k_grid=np.asarray(k_grid, dtype=float),
        gp_lf=gp_lf,
        gp_hf=gp_hf,
        pysr_kwargs=pysr_kwargs_for_cfg(cfg),
        seed=cfg.pysr.seed,
        pareto_csv_out=out_dir / f"pareto_{param_name}.csv",
    )
```

- [ ] **Step 4: Run, expect PASS** — `PYTHONPATH=src pytest tests/test_single_z_refit.py -v` (the 2 pure tests; `refit_one_param_single_z` itself is exercised in Task 5's gated smoke).

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/single_z/refit.py tests/test_single_z_refit.py
git commit -m "Stage 3: single_z/refit.py wrapper (kodiaq_k_grid, pysr_kwargs, refit_one_param_single_z)"
```

---

## Task 3: `run_refit_and_forecast` + shared deliverable writer

**Files:**
- Modify: `src/priya_forecast/single_z/pipeline.py`
- Modify: `tests/test_single_z_pipeline.py`

- [ ] **Step 1: Read** `run_forecast_only` in `pipeline.py` — it writes `forecast_table.txt`, `scorecard.md`, `corner.png` inline. Stage 3 reuses that exact logic, so extract it into a helper.

- [ ] **Step 2: Refactor** — in `pipeline.py`, add a helper `_write_forecast_deliverables(cfg, out_dir, results, pysr_available)` containing the corner-plot + table + scorecard writing currently inside `run_forecast_only`. It returns `dict(table_path=..., scorecard_path=..., corner_path=...)`. Then replace that block inside `run_forecast_only` with a call to the helper. Run `PYTHONPATH=src pytest tests/test_single_z_pipeline.py -q` to confirm `run_forecast_only` still behaves (no regression; gated tests skip).

- [ ] **Step 3: Write the failing (gated) test** — append to `tests/test_single_z_pipeline.py`:

```python
RUN_SLOW_REFIT = os.environ.get("RUN_SLOW_REFIT") == "1"


@pytest.mark.skipif(
    not (RUN_SLOW_REFIT and LYAEMU_AVAILABLE and GP_BASEDIR.exists()),
    reason="gated on RUN_SLOW_REFIT=1 + lyaemu + data/kodiaq_gp/ (runs PySR)",
)
def test_refit_and_forecast_end_to_end(tmp_path: Path):
    """refit_and_forecast refits a 2-param subset and forecasts σ_PySR."""
    import numpy as np
    from priya_forecast.single_z.pipeline import run

    cfg = PipelineConfig(
        mode="refit_and_forecast", redshift=3.6,
        output_dir=str(tmp_path / "out"),
        gp=GPConfig(basedir=str(GP_BASEDIR)),
        parameters=["ns", "Ap"],
        k_range=KRange(min=0.001, max=0.04),
        data=DataConfig(source="kodiaq"),
    )
    result = run(cfg)
    assert result["pysr_available"] is True
    for label in ("GP", "perfect_1D", "PySR"):
        s = result["sigmas"][label]
        assert s.shape == (2,)
        assert np.all(np.isfinite(s)) and np.all(s > 0)
    # the Pareto CSVs were emitted
    for p in ("ns", "Ap"):
        assert (tmp_path / "out" / "refit" / "z3.6" / f"pareto_{p}.csv").exists()
    assert (tmp_path / "out" / "corner.png").exists()
```

- [ ] **Step 4: Implement `run_refit_and_forecast`** — in `pipeline.py`, replace the `run_refit_and_forecast` stub. Add imports at the top: `from priya_forecast.single_z import refit as _refit`, `from priya_forecast.models.gp_model import GPModel`. Implementation:

```python
def run_refit_and_forecast(cfg: PipelineConfig) -> dict:
    """Refit single-z PySR per parameter, emit Pareto CSVs, then forecast.

    Loops `refit_one_param_single_z` over `cfg.parameters` in-process, then
    runs the three Fisher forecasts on the fresh refits.
    """
    out_dir = Path(cfg.output_dir)
    refit_dir = out_dir / "refit" / f"z{cfg.redshift}"
    refit_dir.mkdir(parents=True, exist_ok=True)
    fid = np.asarray(fiducial_vector(), dtype=float)

    k_grid = _refit.kodiaq_k_grid(cfg.k_range.min, cfg.k_range.max, 48)
    gp_lf = GPModel(basedir=cfg.gp.basedir, fidelity="lf", kf=k_grid)
    gp_hf = GPModel(basedir=cfg.gp.basedir, fidelity="hf", kf=k_grid)

    refits: dict = {name: None for name in PARAM_NAMES}
    for param in cfg.parameters:
        refits[param] = _refit.refit_one_param_single_z(
            param_name=param, z=cfg.redshift, cfg=cfg,
            gp_lf=gp_lf, gp_hf=gp_hf, k_grid=k_grid, out_dir=refit_dir,
        )

    results = _fc.run_three_fisher(cfg=cfg, gp=gp_hf, fid=fid, refits=refits)
    deliverables = _write_forecast_deliverables(
        cfg, out_dir, results, pysr_available=True,
    )
    return {
        "sigmas": {label: fr.sigma for label, fr in results.items()},
        "fisher_results": results,
        "pysr_available": True,
        "refit_dir": refit_dir,
        **deliverables,
    }
```

- [ ] **Step 5: Run** — `PYTHONPATH=src pytest tests/test_single_z_pipeline.py -q`. The new gated test SKIPs; check whether a `refit_and_forecast`-raises-NotImplementedError stub test exists and, if so, replace its body with an "is implemented" assertion (same as was done for `forecast_only`). No regression elsewhere.

- [ ] **Step 6: Commit**

```bash
git add src/priya_forecast/single_z/pipeline.py tests/test_single_z_pipeline.py
git commit -m "Stage 3: run_refit_and_forecast — 11-param refit then forecast"
```

---

## Task 4: SLURM array script + CLI

**Files:**
- Create: `scripts/refit_one_param_single_z.py`
- Create: `slurm/single_z_refit.slurm`

- [ ] **Step 1: Create the CLI** `scripts/refit_one_param_single_z.py`:

```python
#!/usr/bin/env python
"""Refit one (parameter, z-bin) with single-z PySR; write its Pareto CSV.

One SLURM array task runs one parameter. Used by slurm/single_z_refit.slurm.

    python scripts/refit_one_param_single_z.py --param ns --z 3.6 \\
        --basedir data/kodiaq_gp --output-dir results/single_z_run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from priya_forecast.parameters import PARAM_NAMES
from priya_forecast.single_z.config import PipelineConfig, GPConfig, PySRConfig
from priya_forecast.single_z import refit as _refit


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--param", required=True, choices=list(PARAM_NAMES))
    p.add_argument("--z", type=float, required=True)
    p.add_argument("--basedir", default="data/kodiaq_gp")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--kmin", type=float, default=0.001)
    p.add_argument("--kmax", type=float, default=0.04)
    p.add_argument("--niterations", type=int, default=50)
    p.add_argument("--maxsize", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from priya_forecast.models.gp_model import GPModel

    cfg = PipelineConfig(
        mode="refit_and_forecast", redshift=args.z,
        output_dir=args.output_dir, gp=GPConfig(basedir=args.basedir),
        pysr=PySRConfig(niterations=args.niterations, maxsize=args.maxsize,
                        seed=args.seed),
    )
    k_grid = _refit.kodiaq_k_grid(args.kmin, args.kmax, 48)
    refit_dir = Path(args.output_dir) / "refit" / f"z{args.z}"

    print(f"Loading emulators from {args.basedir} ...", flush=True)
    t0 = time.time()
    gp_lf = GPModel(basedir=args.basedir, fidelity="lf", kf=k_grid)
    gp_hf = GPModel(basedir=args.basedir, fidelity="hf", kf=k_grid)
    fid = np.array([p.fid for p in __import__(
        "priya_forecast.parameters", fromlist=["PARAMS_11D"]).PARAMS_11D])
    _ = gp_lf.predict(fid, k_grid, args.z)
    _ = gp_hf.predict(fid, k_grid, args.z)
    print(f"  loaded in {time.time() - t0:.0f}s.", flush=True)

    t0 = time.time()
    result = _refit.refit_one_param_single_z(
        param_name=args.param, z=args.z, cfg=cfg,
        gp_lf=gp_lf, gp_hf=gp_hf, k_grid=k_grid, out_dir=refit_dir,
    )
    print(f"[{time.time() - t0:.0f}s] {args.param} z={args.z} -> "
          f"{refit_dir}/pareto_{args.param}.csv "
          f"(eq complexity={result.pareto_complexity}, "
          f"loss={result.pareto_loss:.4g})", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create the SLURM array script** `slurm/single_z_refit.slurm` — adapt `slurm/refit_array.slurm` (read it first). One array job per z-bin; the array indexes the 11 params; `Z` and `OUTPUT_DIR` and `REPO` and `BASEDIR` come via `--export`:

```bash
#!/bin/bash
# GreatLakes SLURM array — single-z PySR refit, 11 params for one z-bin.
#
# Submit one array job per z-bin (the batch driver does this 13×):
#   sbatch --export=ALL,REPO=/home/mfho/lya1d_priya_forecast,\
#                    BASEDIR=data/kodiaq_gp,\
#                    OUTPUT_DIR=results/single_z_run,Z=3.6 \
#          --array=0-10 \
#          slurm/single_z_refit.slurm

#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=sz_refit
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO=${REPO:?REPO must be set via --export}
BASEDIR=${BASEDIR:?BASEDIR must be set via --export}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR must be set via --export}
Z=${Z:?Z must be set via --export}

PARAMS=(dtau0 tau0 ns Ap herei heref alphaq hub omegamh2 hireionz bhfeedback)
PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]:?Array index out of range}

echo "[slurm] task=${SLURM_ARRAY_TASK_ID} param=${PARAM} z=${Z}"

module purge
module load gcc/10.3.0 || true
PY=/sw/pkgs/arc/mamba/py3.11/bin/python

export PYTHON_JULIAPKG_PROJECT="${HOME}/.julia_env"
export JULIA_DEPOT_PATH="${HOME}/.julia"
export PYTHONPATH="/home/mfho/lya_emulator_full:${REPO}/src"

cd "$REPO"
"$PY" scripts/refit_one_param_single_z.py \
    --param "$PARAM" --z "$Z" \
    --basedir "$BASEDIR" --output-dir "$OUTPUT_DIR"

echo "[slurm] ${PARAM} z=${Z} done"
```

- [ ] **Step 3: Smoke-check the CLI parses** — `python -c "import ast; ast.parse(open('scripts/refit_one_param_single_z.py').read())"` succeeds, and `PYTHONPATH=src python scripts/refit_one_param_single_z.py --help` prints usage without error.

- [ ] **Step 4: Commit**

```bash
git add scripts/refit_one_param_single_z.py slurm/single_z_refit.slurm
git commit -m "Stage 3: single-z refit CLI + SLURM array script"
```

---

## Task 5: Stage 3 verification + gated PySR smoke

**Files:** none (verification).

- [ ] **Step 1** — `PYTHONPATH=src pytest tests/test_single_z_refit.py tests/test_single_z_forecast.py tests/test_single_z_pipeline.py tests/test_single_z_combine.py tests/test_single_z_training_data.py -q`. All pure tests PASS, gated tests SKIP, no regression.

- [ ] **Step 2 (if PySR available)** — run the gated end-to-end:
`RUN_SLOW_REFIT=1 PYTHONPATH=src:/home/mfho/lya_emulator_full pytest tests/test_single_z_pipeline.py::test_refit_and_forecast_end_to_end -v`.
Expect PASS (this runs PySR — minutes). If PySR/Julia is unavailable, SKIP is acceptable.

- [ ] **Step 3** — `git status --short` clean under `src/`, `scripts/`, `slurm/`, `tests/`.

---

## Done criteria

- `refit_and_forecast` runs end to end: 11 single-z PySR refits → Pareto CSVs → σ_GP / σ_perfect_1D / σ_PySR + corner plot.
- `refit_1d_for_param` can emit a `load_pareto_csv`-compatible Pareto CSV.
- SLURM array script + CLI ready for the batched all-13-z-bins case (Stage 4 driver invokes them).
- No regression.

Carryover to Stage 4: `run_batch.py` submits `slurm/single_z_refit.slurm` 13× (one per z-bin) and `aggregate_z.py` collects the per-z forecasts.
