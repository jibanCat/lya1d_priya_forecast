# Stage 8 cheap levers — `aq` operator + derivative-validation gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Fisher's-Mirage (derivative-unfaithful PySR equations biasing σ_PySR) by (a) replacing raw `/` with the pole-free `aq(x,y)=x/√(1+y²)` operator and (b) adding a finite-difference derivative-validation gate to single-z equation selection.

**Architecture:** A shared custom-operator registry (`custom_operators.py`) provides the `aq` Julia def, PySR sympy mappings, and a numpy `LAMBDIFY_MODULES` dict (covering existing `inv` + new `aq`) threaded into every `sympy.lambdify(equation_str, …)` site. The gate compares each candidate equation's finite-diff `∂P/∂θ` (same stencil Fisher uses) against the GP's, rejecting equations whose median relative gradient error exceeds `derivative_tol`.

**Tech Stack:** Python 3.11 (project `.venv`, numpy<2), PySR 1.5.10 / Julia, sympy, pytest. Run tests with `PYTHONPATH=src .venv/bin/python -m pytest <path> -q`.

**Spec:** `docs/superpowers/specs/2026-06-04-stage8-cheap-levers-design.md`.

---

## File structure

| File | Responsibility | New/Modify |
|------|----------------|------------|
| `src/priya_forecast/custom_operators.py` | registry: `AQ_JULIA`, `EXTRA_SYMPY_MAPPINGS`, `LAMBDIFY_MODULES` | New |
| `src/priya_forecast/refit_1d_pysr.py` | use `LAMBDIFY_MODULES` in `predict_normalized`; `aq` in `DEFAULT/SMART` kwargs | Modify (`67–126`, `206–221`) |
| `src/priya_forecast/pareto_filters.py` | use `LAMBDIFY_MODULES` in the two lambdify sites | Modify (`67–69`, `138–143`) |
| `src/priya_forecast/derivative_gate.py` | `gp_param_gradient`, `equation_param_gradient`, `derivative_faithful` | New |
| `src/priya_forecast/single_z/forecast.py` | thread the gate into `build_refit_from_pareto` selection | Modify (`105–162`) |
| `src/priya_forecast/single_z/config.py` | `derivative_tol`, `use_aq_operator` config | Modify |
| `src/priya_forecast/single_z/pipeline.py` | pass GP target gradient into selection | Modify |
| `configs/single_z/stage8_z3.6.yaml` | production config | New |
| `tests/test_stage8_*.py` | unit + integration + gated tests | New |

**Read before starting:** `src/priya_forecast/refit_1d_pysr.py:62–126` (PySR kwargs) and `:190–249` (`predict_normalized`); `src/priya_forecast/pareto_filters.py:33–160` (`has_pathological_constant`, `is_fisher_stencil_safe`); `src/priya_forecast/single_z/forecast.py:105–162` (`build_refit_from_pareto`, `_filter_fisher_safe`); `src/priya_forecast/fisher.py:130–150` (`_stencil_derivative`, the finite-diff convention to mirror).

**Key facts (verified):**
- `predict_normalized` and `is_fisher_stencil_safe` both do `sp.sympify(eq_str)` then `sp.lambdify(syms, expr, modules=[{"inv": lambda x: 1.0/x}, "numpy"])`. Bare `sympify` leaves a custom op as an *undefined Function* (no error — spike-confirmed); `lambdify`'s `modules` dict maps it numerically. So the fix is the `modules` dict, NOT sympify locals.
- `Refit1DResult.predict(theta_phys, k, resolution, z)` returns physical `P_F`; `predict_normalized` returns flux-norm. The gate uses `predict` (physical units, log or linear per `log_space`).
- `GPModel.predict(theta, k, z)` returns `P_F` at the full 11-vector `theta`.

---

## Task 1: custom-operator registry

**Files:**
- Create: `src/priya_forecast/custom_operators.py`
- Test: `tests/test_stage8_custom_operators.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage8_custom_operators.py
import numpy as np
import sympy as sp
from priya_forecast.custom_operators import (
    AQ_JULIA, EXTRA_SYMPY_MAPPINGS, LAMBDIFY_MODULES,
)


def test_aq_julia_def_shape():
    assert AQ_JULIA.startswith("aq(") and "sqrt(1 + y^2)" in AQ_JULIA.replace(" ", "")[:40] or "sqrt(1+y^2)" in AQ_JULIA.replace(" ", "")


def test_lambdify_modules_cover_inv_and_aq():
    assert "inv" in LAMBDIFY_MODULES and "aq" in LAMBDIFY_MODULES
    assert LAMBDIFY_MODULES["inv"](2.0) == 0.5
    np.testing.assert_allclose(LAMBDIFY_MODULES["aq"](1.0, 2.0), 1.0 / np.sqrt(5.0))


def test_extra_sympy_mappings_cover_inv_and_aq():
    assert "inv" in EXTRA_SYMPY_MAPPINGS and "aq" in EXTRA_SYMPY_MAPPINGS
    x, y = sp.symbols("x y")
    assert sp.simplify(EXTRA_SYMPY_MAPPINGS["aq"](x, y) - x / sp.sqrt(1 + y**2)) == 0


def test_aq_roundtrip_through_lambdify():
    # The crux the feasibility spike found: a raw equation string with aq(...)
    # must lambdify + differentiate via LAMBDIFY_MODULES.
    expr = sp.sympify("aq(x0, 2*x1)")
    x0, x1 = sp.Symbol("x0"), sp.Symbol("x1")
    fn = sp.lambdify([x0, x1], expr, modules=[LAMBDIFY_MODULES, "numpy"])
    assert np.isclose(float(fn(0.5, 0.5)), 0.5 / np.sqrt(1 + 1.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_stage8_custom_operators.py -q`
Expected: FAIL — `ModuleNotFoundError: ...custom_operators`.

- [ ] **Step 3: Implement the registry**

```python
# src/priya_forecast/custom_operators.py
"""Custom PySR operators and the sympy/lambdify mappings to evaluate them.

Single source of truth so adding an operator is one entry. `aq` is the
analytic quotient x/sqrt(1+y^2) — a bounded, pole-free replacement for raw
division (raw `/` creates poles whose derivatives wreck the Fisher matrix;
see docs/SR_EMULATOR_LITERATURE_NOTES.md). `inv` is the pre-existing 1/x op.
"""
from __future__ import annotations

import numpy as np
import sympy as sp

# Julia definition for PySR `binary_operators`.
AQ_JULIA = "aq(x, y) = x / sqrt(1 + y^2)"

# sympy-backed mappings for PySRRegressor(extra_sympy_mappings=...) — used by
# PySR's own .sympy() expansion.
EXTRA_SYMPY_MAPPINGS = {
    "inv": lambda x: 1 / x,
    "aq": lambda x, y: x / sp.sqrt(1 + y**2),
}

# numpy-backed mappings for sympy.lambdify(..., modules=[LAMBDIFY_MODULES, "numpy"]).
# Threaded into every equation-evaluation site so a raw equation string
# containing inv(...) or aq(...) evaluates and differentiates numerically.
LAMBDIFY_MODULES = {
    "inv": lambda x: 1.0 / x,
    "aq": lambda x, y: x / np.sqrt(1.0 + y**2),
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_stage8_custom_operators.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/custom_operators.py tests/test_stage8_custom_operators.py
git commit -m "Stage 8: custom-operator registry (aq + inv mappings)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: thread `LAMBDIFY_MODULES` into all eval/filter parse sites

Replace the inline `{"inv": lambda x: 1.0/x}` lambdify modules with the shared `LAMBDIFY_MODULES` in `predict_normalized` and the two `pareto_filters` sites, so `aq` equations evaluate everywhere.

**Files:**
- Modify: `src/priya_forecast/refit_1d_pysr.py:218–221`
- Modify: `src/priya_forecast/pareto_filters.py:67–69`, `:140–143`
- Test: `tests/test_stage8_aq_eval.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage8_aq_eval.py
import numpy as np
from priya_forecast.refit_1d_pysr import Refit1DResult
from priya_forecast.models.normalization import NormalizationSpec
from priya_forecast.pareto_filters import is_fisher_stencil_safe


def _aq_refit():
    k_grid = np.linspace(0.005, 0.04, 6)
    norm = NormalizationSpec(
        param_min=0.0, param_max=2.0, k_min=float(k_grid.min()),
        k_max=float(k_grid.max()), mean_flux=np.zeros(6), std_flux=np.ones(6),
        k_grid=k_grid)
    return Refit1DResult(
        param_name="ns", z=3.6, equation_str="aq(x0, x1)",
        pareto_complexity=3, pareto_loss=0.01, pareto_complexities=[3],
        pareto_losses=[0.01], x_param_min=0.0, x_param_max=2.0,
        k_min=float(k_grid.min()), k_max=float(k_grid.max()),
        lf_resolution=0.4, hf_resolution=0.8, fid_value=1.0, norm=norm,
        k_grid=k_grid, wall_time_s=0.0, lf_train_mean_rel_err=0.0,
        hf_train_mean_rel_err=0.0, lf_train_max_rel_err=0.0,
        hf_train_max_rel_err=0.0)


def test_predict_normalized_handles_aq():
    r = _aq_refit()
    k_grid = np.linspace(0.005, 0.04, 6)
    out = r.predict_normalized(theta_phys=1.0, k=k_grid, resolution=0.8)
    # x0 = (1-0)/(2-0) = 0.5; x1 = k_norm; aq(0.5, x1) = 0.5/sqrt(1+x1^2)
    k_norm = (k_grid - k_grid.min()) / (k_grid.max() - k_grid.min())
    np.testing.assert_allclose(out, 0.5 / np.sqrt(1 + k_norm**2), rtol=1e-10)


def test_is_fisher_stencil_safe_handles_aq():
    # aq is bounded everywhere -> stencil-safe (no blow-up).
    assert is_fisher_stencil_safe("aq(x0, x1)", n_features=3) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_stage8_aq_eval.py -q`
Expected: FAIL — `predict_normalized` lambdify has no `aq` mapping → `NameError`/`TypeError` evaluating `aq`.

- [ ] **Step 3: Update the lambdify sites**

In `refit_1d_pysr.py`, add the import near the top (with the other intra-package imports):

```python
from priya_forecast.custom_operators import LAMBDIFY_MODULES
```

Replace `predict_normalized`'s lambdify (lines ~218–221):

```python
        fn = sp.lambdify(
            all_syms, expr,
            modules=[LAMBDIFY_MODULES, "numpy"],
        )
```

In `pareto_filters.py`, add at top: `from priya_forecast.custom_operators import LAMBDIFY_MODULES`. Replace BOTH lambdify calls (the one in `has_pathological_constant` ~line 67–69 region if it lambdifies, and `is_fisher_stencil_safe` lines ~140–143):

```python
        fn = sp.lambdify(
            all_syms, expr,
            modules=[LAMBDIFY_MODULES, "numpy"],
        )
```

(If `has_pathological_constant` does not lambdify — only inspects constants — leave it; only update lambdify sites. Verify by reading lines 60–105.)

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_stage8_aq_eval.py -q`
Expected: 2 passed.

- [ ] **Step 5: No-regression on existing eval/filter tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q -k "pareto or refit or filter or predict"`
Expected: all pass (the `inv` path still works — `LAMBDIFY_MODULES` includes it).

- [ ] **Step 6: Commit**

```bash
git add src/priya_forecast/refit_1d_pysr.py src/priya_forecast/pareto_filters.py tests/test_stage8_aq_eval.py
git commit -m "Stage 8: thread LAMBDIFY_MODULES into eval/filter sites (aq support)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `aq` operator in the PySR kwargs (drop raw `/`)

**Files:**
- Modify: `src/priya_forecast/refit_1d_pysr.py:62–126`
- Test: `tests/test_stage8_pysr_kwargs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage8_pysr_kwargs.py
from priya_forecast.refit_1d_pysr import DEFAULT_PYSR_KWARGS, SMART_REFIT_PYSR_KWARGS
from priya_forecast.custom_operators import AQ_JULIA


def test_raw_division_dropped_aq_added():
    for kw in (DEFAULT_PYSR_KWARGS, SMART_REFIT_PYSR_KWARGS):
        ops = kw["binary_operators"]
        assert "/" not in ops, f"raw / must be dropped, got {ops}"
        assert AQ_JULIA in ops, f"aq must be present, got {ops}"


def test_extra_sympy_mappings_have_aq():
    for kw in (DEFAULT_PYSR_KWARGS, SMART_REFIT_PYSR_KWARGS):
        assert "aq" in kw["extra_sympy_mappings"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_stage8_pysr_kwargs.py -q`
Expected: FAIL — `/` still present, no `aq`.

- [ ] **Step 3: Update the kwargs**

In `DEFAULT_PYSR_KWARGS` (line 67) change `binary_operators` and `extra_sympy_mappings`:

```python
    binary_operators=["+", "-", "*", AQ_JULIA, "^"],
    unary_operators=["exp", "log", "square", "sqrt", "inv(x) = 1/x"],
    extra_sympy_mappings=dict(EXTRA_SYMPY_MAPPINGS),
```

Add imports near the top of `refit_1d_pysr.py`:

```python
from priya_forecast.custom_operators import AQ_JULIA, EXTRA_SYMPY_MAPPINGS, LAMBDIFY_MODULES
```

In the `SMART_REFIT_PYSR_KWARGS` block (lines 115, 117) change:

```python
SMART_REFIT_PYSR_KWARGS["binary_operators"] = ["+", "-", "*", AQ_JULIA, "^"]
SMART_REFIT_PYSR_KWARGS["unary_operators"] = ["exp", "log", "square"]
SMART_REFIT_PYSR_KWARGS["extra_sympy_mappings"] = dict(EXTRA_SYMPY_MAPPINGS)
```

Leave `constraints={"^": (-1, 0)}` and `complexity_of_operators={"^": 3}` unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_stage8_pysr_kwargs.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/refit_1d_pysr.py tests/test_stage8_pysr_kwargs.py
git commit -m "Stage 8: replace raw / with aq operator in PySR kwargs

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: GP target gradient + equation gradient (finite-diff)

**Files:**
- Create: `src/priya_forecast/derivative_gate.py`
- Test: `tests/test_stage8_gradients.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage8_gradients.py
import numpy as np
from priya_forecast.derivative_gate import gp_param_gradient, equation_param_gradient
from priya_forecast.refit_1d_pysr import Refit1DResult
from priya_forecast.models.normalization import NormalizationSpec


class _LinGP:
    # P_F(theta,k,z) = base(k) * (1 + s*theta[i]); dP/dtheta_i = base(k)*s
    def __init__(self, i, s=0.3):
        self.i, self.s = i, s
    def predict(self, theta, k, z):
        k = np.asarray(k, float)
        return (1.0 + 0.5 * k) * (1.0 + self.s * float(theta[self.i]))


def test_gp_param_gradient_matches_analytic():
    k = np.linspace(0.005, 0.04, 6)
    fid = np.zeros(11)
    g = gp_param_gradient(gp=_LinGP(2), fid=fid, k_grid=k, z=3.6, param_idx=2, h=1e-3)
    np.testing.assert_allclose(g, (1.0 + 0.5 * k) * 0.3, rtol=1e-4)


def test_equation_param_gradient_matches_predict_fd():
    k = np.linspace(0.005, 0.04, 6)
    norm = NormalizationSpec(param_min=0.0, param_max=2.0, k_min=float(k.min()),
        k_max=float(k.max()), mean_flux=np.zeros(6), std_flux=np.ones(6), k_grid=k)
    r = Refit1DResult(param_name="ns", z=3.6, equation_str="x0 + x1",
        pareto_complexity=3, pareto_loss=0.01, pareto_complexities=[3],
        pareto_losses=[0.01], x_param_min=0.0, x_param_max=2.0, k_min=float(k.min()),
        k_max=float(k.max()), lf_resolution=0.4, hf_resolution=0.8, fid_value=1.0,
        norm=norm, k_grid=k, wall_time_s=0.0, lf_train_mean_rel_err=0.0,
        hf_train_mean_rel_err=0.0, lf_train_max_rel_err=0.0, hf_train_max_rel_err=0.0)
    g = equation_param_gradient(refit=r, fid_value=1.0, k_grid=k, z=3.6, h=1e-3)
    assert g.shape == (6,) and np.all(np.isfinite(g))
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_stage8_gradients.py -q`
Expected: FAIL — `ModuleNotFoundError: ...derivative_gate`.

- [ ] **Step 3: Implement the gradients**

```python
# src/priya_forecast/derivative_gate.py
"""Finite-difference derivative-validation gate for PySR equations.

Compares a candidate equation's central-difference dP/dtheta at fid against
the GP's, using the SAME stencil the Fisher matrix consumes (fisher.py).
Equations whose gradient is unfaithful (the "Fisher's-Mirage" pathology) are
rejected before best_loss selection.
"""
from __future__ import annotations

import numpy as np

from priya_forecast.refit_1d_pysr import HF_RESOLUTION, Refit1DResult


def gp_param_gradient(*, gp, fid: np.ndarray, k_grid: np.ndarray, z: float,
                      param_idx: int, h: float = 1e-3) -> np.ndarray:
    """Central-difference dP_GP/dtheta_param at fid, per k-bin."""
    fid = np.asarray(fid, dtype=float)
    k_grid = np.asarray(k_grid, dtype=float)
    tp, tm = fid.copy(), fid.copy()
    step = h * max(abs(float(fid[param_idx])), 1.0)
    tp[param_idx] += step
    tm[param_idx] -= step
    pp = np.asarray(gp.predict(tp, k_grid, z), dtype=float)
    pm = np.asarray(gp.predict(tm, k_grid, z), dtype=float)
    return (pp - pm) / (2.0 * step)


def equation_param_gradient(*, refit: Refit1DResult, fid_value: float,
                            k_grid: np.ndarray, z: float, h: float = 1e-3,
                            resolution: float = HF_RESOLUTION) -> np.ndarray:
    """Central-difference dP_eq/dtheta at fid via the refit's own predict()."""
    k_grid = np.asarray(k_grid, dtype=float)
    step = h * max(abs(float(fid_value)), 1.0)
    pp = np.asarray(refit.predict(theta_phys=fid_value + step, k=k_grid,
                                  resolution=resolution, z=z), dtype=float)
    pm = np.asarray(refit.predict(theta_phys=fid_value - step, k=k_grid,
                                  resolution=resolution, z=z), dtype=float)
    return (pp - pm) / (2.0 * step)
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_stage8_gradients.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/derivative_gate.py tests/test_stage8_gradients.py
git commit -m "Stage 8: GP + equation finite-diff parameter gradients

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: the derivative-faithfulness gate predicate

**Files:**
- Modify: `src/priya_forecast/derivative_gate.py`
- Test: `tests/test_stage8_gate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage8_gate.py
import numpy as np
from priya_forecast.derivative_gate import derivative_faithful


def test_faithful_passes():
    target = np.array([1.0, 2.0, 3.0, 4.0])
    cand = target * 1.05               # 5% off -> median rel err 0.05 < 0.25
    assert derivative_faithful(cand_grad=cand, target_grad=target, tol=0.25) is True


def test_unfaithful_rejected():
    target = np.array([1.0, 2.0, 3.0, 4.0])
    cand = target * -0.5               # wrong sign + magnitude -> rel err ~1.5
    assert derivative_faithful(cand_grad=cand, target_grad=target, tol=0.25) is False


def test_near_zero_target_bins_masked():
    # bins where the GP gradient is ~0 must not blow up the ratio
    target = np.array([1.0, 1e-12, 1e-12, 1.0])
    cand = np.array([1.05, 5.0, -3.0, 1.05])   # huge rel err only in masked bins
    assert derivative_faithful(cand_grad=cand, target_grad=target, tol=0.25) is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_stage8_gate.py -q`
Expected: FAIL — `derivative_faithful` not defined.

- [ ] **Step 3: Implement the predicate**

Append to `src/priya_forecast/derivative_gate.py`:

```python
def derivative_faithful(*, cand_grad: np.ndarray, target_grad: np.ndarray,
                        tol: float = 0.25, floor_frac: float = 1e-3) -> bool:
    """True if median_k |cand/target - 1| <= tol over non-negligible bins.

    Bins where |target_grad| is below `floor_frac` times its own max are
    masked out (a ~zero GP gradient makes the ratio meaningless / explosive).
    If every bin is masked, returns False (no usable gradient to validate).
    """
    cand = np.asarray(cand_grad, dtype=float)
    target = np.asarray(target_grad, dtype=float)
    amax = float(np.max(np.abs(target)))
    if amax == 0.0:
        return False
    keep = np.abs(target) >= floor_frac * amax
    if not np.any(keep):
        return False
    rel = np.abs(cand[keep] / target[keep] - 1.0)
    return bool(np.median(rel) <= tol)
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_stage8_gate.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/derivative_gate.py tests/test_stage8_gate.py
git commit -m "Stage 8: derivative-faithfulness gate predicate (masked median rel-err)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: wire the gate into single-z selection

`build_refit_from_pareto` (`single_z/forecast.py`) currently filters the Pareto frame for Fisher-safety, then `pick_equation(best_loss)`, then reconstructs ONE refit. Restructure to a derivative-aware pick: iterate the Fisher-safe rows in best-loss order, reconstruct each candidate, accept the first whose gradient passes the gate; if none pass, raise `ValueError` (→ existing GP-slice fallback in the caller).

**Files:**
- Modify: `src/priya_forecast/single_z/forecast.py:105–162` (`build_refit_from_pareto`)
- Modify: `src/priya_forecast/single_z/config.py` (add `derivative_tol`)
- Modify: `src/priya_forecast/single_z/pipeline.py` (compute GP target gradient, pass it through)
- Test: `tests/test_stage8_selection.py`

- [ ] **Step 1: Read the current selection + add config**

Read `build_refit_from_pareto` (`forecast.py:105–162`) and `run_forecast_only`/`run_refit_and_forecast` (`pipeline.py`). In `config.py`, add to `FisherConfig` or `PipelineConfig`:

```python
    derivative_tol: float = 0.25
```
(Place on `PipelineConfig`, validated `> 0` in `validate()`.)

- [ ] **Step 2: Write the failing test**

```python
# tests/test_stage8_selection.py
import numpy as np, pandas as pd
from priya_forecast.single_z.forecast import build_refit_from_pareto_gated


class _LinGP:
    def predict(self, theta, k, z):
        k = np.asarray(k, float)
        return (1.0 + 0.5 * k) * (1.0 + 0.3 * float(theta[0]))   # param ns=idx0? use real idx in impl


def test_gate_picks_faithful_over_lower_loss(tmp_path, monkeypatch):
    # Two x0-dependent, Fisher-safe equations: row A lower loss but wrong
    # gradient; row B higher loss but faithful. Gate must pick B.
    # (This test calls the gated selection with a stub GP target gradient.)
    ...  # SEE NOTE — implement against the real build_refit_from_pareto_gated signature
```

> NOTE: this test must be written against the real `build_refit_from_pareto_gated` signature defined in Step 3. Because the selection reconstructs a `Refit1DResult` (which needs 1pvar training data via `load_1pvar`), the unit test stubs `load_1pvar` (monkeypatch) to return a fixed LF flux sweep, and passes a synthetic `gp_target_grad`. Write it so it asserts: given two candidate rows where the lower-loss one has a gradient far from `gp_target_grad` and the higher-loss one matches, the returned refit's `equation_str` is the faithful (higher-loss) one. Keep the two equations simple (e.g. `"x0*x1"` vs `"x0 + x1"`) and choose `gp_target_grad` to match one of them.

- [ ] **Step 3: Implement `build_refit_from_pareto_gated`**

Add a new function in `forecast.py` (keep the old `build_refit_from_pareto` for callers that don't gate; the gated one wraps it per-candidate):

```python
def build_refit_from_pareto_gated(
    *, param_name, z, pareto_csv, pick_rule, data_1pvar_dir,
    gp_target_grad, derivative_tol=0.25, log_space=False,
):
    """Filter-then-derivative-gate-then-pick.

    Iterate Fisher-safe Pareto rows in best-loss order; reconstruct each into
    a Refit1DResult and accept the first whose finite-diff dP/dtheta passes the
    derivative gate vs `gp_target_grad`. Raise ValueError if none pass (caller
    falls back to the GP slice).
    """
    from priya_forecast.derivative_gate import equation_param_gradient, derivative_faithful
    df = load_pareto_csv(pareto_csv)
    safe = _filter_fisher_safe(df, n_features=3)
    if safe.empty:
        raise ValueError(f"No Fisher-safe equation for ({param_name}, z={z}).")
    # Reconstruct each candidate (best-loss order) and gate on derivative.
    safe = safe.sort_values("Loss").reset_index(drop=True)
    meta = get_param(param_name)
    d = load_1pvar(param_name=param_name, z=z, data_dir=data_1pvar_dir)
    k_grid = d["kfkms_lf_z"][0]
    norm = per_param_local_norm(
        flux_lf_z=d["flux_lf_z"], k_grid=k_grid,
        param_min=float(meta.prior[0]), param_max=float(meta.prior[1]),
        log_space=log_space)
    for _, row in safe.iterrows():
        cand = _refit_from_row(  # small helper: builds a Refit1DResult from one row + norm
            row=row, param_name=param_name, z=z, df=df, meta=meta,
            k_grid=k_grid, norm=norm, log_space=log_space)
        g = equation_param_gradient(refit=cand, fid_value=float(meta.fid),
                                    k_grid=np.asarray(k_grid, float), z=float(z))
        if derivative_faithful(cand_grad=g, target_grad=gp_target_grad,
                               tol=derivative_tol):
            return cand
    raise ValueError(
        f"No derivative-faithful equation for ({param_name}, z={z}) "
        f"at tol={derivative_tol} — GP-slice fallback.")
```

Refactor the existing `build_refit_from_pareto` body that constructs the `Refit1DResult` into a `_refit_from_row(row, …)` helper so both the old and gated paths share it (DRY). The old `build_refit_from_pareto` calls `_refit_from_row` on the `pick_equation`-selected row; the gated one calls it per-candidate.

- [ ] **Step 4: Thread the GP target gradient through the pipeline**

In `pipeline.py` `run_forecast_only`/`run_refit_and_forecast`, before reconstructing refits, compute per-param GP target gradients and call the gated builder:

```python
from priya_forecast.derivative_gate import gp_param_gradient
# gp is the HF GPModel; k_grid from the likelihood; fid the 11-vector.
target_grads = {
    p: gp_param_gradient(gp=gp, fid=fid, k_grid=k_grid, z=cfg.redshift,
                         param_idx=PARAM_NAMES.index(p))
    for p in cfg.parameters
}
# then per param:
refits[param] = _fc.build_refit_from_pareto_gated(
    param_name=param, z=cfg.redshift, pareto_csv=csv, pick_rule=cfg.pick,
    data_1pvar_dir="data/single_z_1pvar", gp_target_grad=target_grads[param],
    derivative_tol=cfg.derivative_tol, log_space=(cfg.target_space == "log"))
```
Keep the existing `try/except ValueError → dropped/GP-slice` wrapper so a failed gate falls back exactly as a failed Fisher-safe filter does today. Note `k_grid` here must be the GP-refit k-grid (the `kodiaq_k_grid` used in refit), consistent with what `equation_param_gradient` uses; if `run_forecast_only`'s likelihood k-grid differs, compute `target_grads` on the refit k-grid (`_refit.kodiaq_k_grid(cfg.k_range.min, cfg.k_range.max, 48)`).

- [ ] **Step 5: Run the selection test**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_stage8_selection.py -q`
Expected: PASS (gate picks the faithful equation over the lower-loss one).

- [ ] **Step 6: No-regression**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`
Expected: existing suite passes (gated builder is additive; old path intact).

- [ ] **Step 7: Commit**

```bash
git add src/priya_forecast/single_z/forecast.py src/priya_forecast/single_z/config.py src/priya_forecast/single_z/pipeline.py tests/test_stage8_selection.py
git commit -m "Stage 8: derivative-validation gate in single-z selection

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: production config + gated end-to-end + COMPARISON

**Files:**
- Create: `configs/single_z/stage8_z3.6.yaml`
- Create: `tests/test_stage8_end_to_end.py` (gated `RUN_SLOW_REFIT`)
- Create (by running): `results/single_z_stage8/COMPARISON.md`

- [ ] **Step 1: Write the production config**

```yaml
# configs/single_z/stage8_z3.6.yaml — single-z z=3.6 with aq + derivative gate.
mode: refit_and_forecast
redshift: 3.6
output_dir: results/single_z_stage8/
parameters: [ns, Ap, hub, omegamh2, herei, heref, alphaq, hireionz, bhfeedback, dtau0, tau0]
k_range: {min: 0.001, max: 0.04}
data: {source: kodiaq, cov_scale: 1.0, conservative: true, mock_data: gp}
gp: {basedir: data/kodiaq_gp}
combine: additive
target_space: log
derivative_tol: 0.25
fisher: {step_frac: 0.01, rel_tol: 0.01}
```

- [ ] **Step 2: Write the gated end-to-end test**

```python
# tests/test_stage8_end_to_end.py
import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_REFIT"),
    reason="needs PySR/Julia + emulator; set RUN_SLOW_REFIT=1",
)


def test_aq_gate_refit_forecast_two_params(tmp_path):
    from priya_forecast.single_z.config import PipelineConfig
    from priya_forecast.single_z.pipeline import run
    cfg = PipelineConfig(mode="refit_and_forecast", redshift=3.6,
        parameters=["ns", "Ap"], target_space="log",
        output_dir=str(tmp_path / "s8"))
    cfg.gp.basedir = "data/kodiaq_gp"
    cfg.pysr.niterations = 20
    res = run(cfg)
    assert "GP" in res["fisher_results"]
    np.testing.assert_allclose(
        res["fisher_results"]["perfect_1D"].sigma,
        res["fisher_results"]["GP"].sigma, rtol=1e-3)
```

- [ ] **Step 3: Run the gated test on the cluster venv**

Run: `PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full RUN_SLOW_REFIT=1 .venv/bin/python -m pytest tests/test_stage8_end_to_end.py -q`
Expected: PASS. (Confirms aq equations refit, the gate runs, perfect_1D==GP.)

- [ ] **Step 4: Production run (cluster) + COMPARISON**

Submit the 11-param refit (reuse `slurm/single_z_refit.slurm`, now venv-based, with `%3`):
```bash
sbatch --account=yueyingn0 --export=ALL,REPO=$(pwd),BASEDIR=data/kodiaq_gp,\
OUTPUT_DIR=results/single_z_stage8,Z=3.6,TARGET_SPACE=log \
--array=0-10%3 slurm/single_z_refit.slurm
```
Then the forecast:
```bash
PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full \
  .venv/bin/python scripts/run_pipeline.py --config configs/single_z/stage8_z3.6.yaml
```
Write `results/single_z_stage8/COMPARISON.md`: mean |log10(σ_PySR/σ_GP)|, sub-1 Mirage count, deep-Mirage count, and GP-slice fallback count vs Stage 6 (`results/single_z_stage6_log/scorecard.md`: 0.366 / 7 / 0). Win = lower mean |log10|, fewer sub-1, fallbacks no worse.

> NOTE: `single_z_refit.slurm` calls `refit_one_param_single_z.py`, which uses `SMART_REFIT_PYSR_KWARGS` (now aq-based). The gate runs at forecast time (`run_forecast_only`/`run_refit_and_forecast`), so the production refit array produces aq equations and the forecast applies the gate. Confirm `refit_one_param_single_z.py` needs no change (it reads the kwargs via `pysr_kwargs_for_cfg`).

- [ ] **Step 5: Commit**

```bash
git add configs/single_z/stage8_z3.6.yaml tests/test_stage8_end_to_end.py results/single_z_stage8/COMPARISON.md
git commit -m "Stage 8: production config, gated e2e, aq+gate vs Stage 6 COMPARISON

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: full sweep + HANDOFF

- [ ] **Step 1: Full fast suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`
Expected: all pass + the new Stage 8 unit tests; gated tests skip.

- [ ] **Step 2: Update HANDOFF.md**

Add a Stage 8 section: aq operator + derivative gate done; the z=3.6 Mirage delta vs Stage 6; whether the gate increased GP-slice fallbacks; and the decision on whether lever #1 (ratio target) / the Sobolev loss is still warranted. Note multi-z application is the next follow-up.

- [ ] **Step 3: Commit**

```bash
git add HANDOFF.md
git commit -m "Stage 8 (cheap levers) done: HANDOFF refresh

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage:** §3.1 registry → Task 1; §3.1 parse-site threading → Task 2 (realized as `LAMBDIFY_MODULES` in lambdify `modules`, matching existing code, rather than sympify locals — a refinement of the spec's intent); §3.2 aq operator → Task 3; §3.3 GP/equation gradients → Task 4, gate predicate → Task 5, selection integration → Task 6; §3.4 config (`derivative_tol`, `use_aq_operator`) → Task 6 (note: `use_aq_operator` toggle deferred — aq is on by default via the kwargs; add the toggle only if a param regresses, per §6 risk); §4 success metric → Task 7; §5 tests 1–8 → Tasks 1,2,4,5,6,7; §6 risks (near-zero mask, fallback) → Task 5/6.
- **Refinement flagged:** the spec's `sympify_equation(locals=…)` is implemented as `LAMBDIFY_MODULES` in the lambdify `modules` arg — the existing code maps custom ops at lambdify time, not sympify time (bare sympify tolerates undefined functions, spike-confirmed). Same effect, matches the codebase.
- **Deferred from spec:** `use_aq_operator` config toggle (YAGNI until a regression appears; aq is the default). Noted, not silently dropped.
- **Type consistency:** `gp_param_gradient`, `equation_param_gradient`, `derivative_faithful`, `build_refit_from_pareto_gated`, `_refit_from_row`, `LAMBDIFY_MODULES`, `AQ_JULIA`, `EXTRA_SYMPY_MAPPINGS` used consistently across tasks.
- **Known soft spot:** Task 6 Step 2 test ships as a NOTE-guided stub — the engineer writes it against the real `build_refit_from_pareto_gated` signature + a monkeypatched `load_1pvar`. The selection-restructure (shared `_refit_from_row`) is the riskiest change; Task 6 Step 6 runs the whole suite to catch regressions.
