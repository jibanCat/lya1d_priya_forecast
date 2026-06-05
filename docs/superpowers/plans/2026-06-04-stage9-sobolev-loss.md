# Stage 9 — Sobolev derivative loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the PySR search *generate* derivative-faithful equations (fixing the ns/hub Fisher's-Mirage stragglers) by adding a Sobolev term `λ·MSE(∂eq/∂θ, ∂logP_GP/∂θ)` to the PySR loss.

**Architecture:** Unchanged pipeline (per-1D refits, inputs θ/k/z/resolution, additive combine, forecast). The change is a custom Julia `loss_function` that finite-differences the tree's θ-derivative *in-loss* and matches it to a per-point GP target gradient delivered via PySR's `weights` channel. Spike-confirmed it runs.

**Tech Stack:** Python 3.11 (`.venv`, numpy<2), PySR 1.5.10 / SymbolicRegression.jl, pytest. Run tests with `PYTHONPATH=src .venv/bin/python -m pytest <path> -q`; PySR/GP runs also need `PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia` and `PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full`.

**Spec:** `docs/superpowers/specs/2026-06-04-stage9-sobolev-loss-design.md`.

---

## File structure

| File | Responsibility | New/Modify |
|------|----------------|------------|
| `src/priya_forecast/sobolev_loss.py` | `make_sobolev_loss(lam, h)` → Julia loss string; `sobolev_target_weights(...)` → per-row target gradient | New |
| `src/priya_forecast/refit_1d_pysr.py` | wire Sobolev loss + weights into `refit_1d_for_param` | Modify (`962+`) |
| `src/priya_forecast/single_z/config.py` | `use_sobolev`, `sobolev_lambda` on `PySRConfig` | Modify |
| `src/priya_forecast/single_z/refit.py` | thread the flag through `pysr_kwargs_for_cfg` / `refit_one_param_single_z` | Modify |
| `scripts/refit_one_param_single_z.py` | `--use-sobolev`, `--sobolev-lambda` CLI | Modify |
| `slurm/single_z_refit.slurm` | `USE_SOBOLEV` env passthrough | Modify |
| `configs/single_z/stage9_z3.6.yaml` | production config | New |
| `tests/test_stage9_*.py` | unit + gated | New |

**Read before starting:**
- `src/priya_forecast/dim_balanced_loss.py:203–256` (`JULIA_LOSS_FUNCTION_ANOVA`) — the Julia `loss_function(tree, dataset::Dataset{T,L}, options)` pattern; `eval_tree_array(tree, dataset.X, options)`; `dataset.X` is **(n_features × n_points)**; `dataset.y`, `dataset.weights` available; returns an `L` value.
- `src/priya_forecast/refit_1d_pysr.py:344–445` (`_build_training_matrix`): `X_act = vstack([X_lf, X_hf])`, rows `[θ_norm, k_norm, resolution]`; **LF rows first then HF**; within a fidelity, **point-major / k-minor** (`x_param_lf_norm.ravel()` over `(n_points, n_k)`); `ranges = {x_param_min, x_param_max, k_min, k_max}`; normalization `flux_norm = (logP − mean_k)/std_k`.
- `src/priya_forecast/refit_1d_pysr.py:962+` (`refit_1d_for_param`): generates `payload` via `_generate_1pvar_inline`, computes `norm`, builds matrix, fits `model.fit(X_act, Y_act.reshape(-1,1))`.
- `src/priya_forecast/derivative_gate.py` (Stage 8): `gp_param_gradient`, `equation_param_gradient`, `derivative_faithful` — reused for validation.

**Key normalization fact (from the spec — the lever-#1 lesson):** the in-loss finite-diff `∂eq/∂x0` is a derivative of *normalized* logP w.r.t. *normalized* θ. So the target weight at a point is
```
weight = (∂logP_GP/∂θ_phys) · (x_param_max − x_param_min) / std_k
```
computed **per fidelity** (LF gradient for LF rows, HF for HF rows).

---

## Task 1: the Sobolev Julia loss string

**Files:**
- Create: `src/priya_forecast/sobolev_loss.py`
- Test: `tests/test_stage9_loss_string.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage9_loss_string.py
from priya_forecast.sobolev_loss import make_sobolev_loss


def test_loss_string_has_terms_and_constants():
    s = make_sobolev_loss(lam=2.5, h=1e-4)
    # value term + finite-diff derivative + weights reference + injected constants
    assert "eval_tree_array(tree, dataset.X, options)" in s
    assert "dataset.y" in s and "dataset.weights" in s
    assert "X2[1, :]" in s                    # shift x0 (feature row 1)
    assert "2.5" in s                          # lambda injected
    assert "0.0001" in s or "1.0e-4" in s or "1e-4" in s  # h injected
    assert s.strip().startswith("function loss_function")


def test_lambda_changes_string():
    assert make_sobolev_loss(lam=1.0, h=1e-4) != make_sobolev_loss(lam=9.0, h=1e-4)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_stage9_loss_string.py -q`
Expected: `ModuleNotFoundError: ...sobolev_loss`.

- [ ] **Step 3: Implement `make_sobolev_loss`**

```python
# src/priya_forecast/sobolev_loss.py (part 1)
"""Sobolev derivative-matching loss for PySR refits.

Adds  λ·MSE( ∂eq/∂θ_norm , target_grad )  to the value MSE, where ∂eq/∂θ_norm
is finite-differenced INSIDE the loss (eval the tree at X and at X shifted by
+h in the θ-feature row) and `target_grad` is the GP's gradient delivered via
PySR's per-point `weights` channel. Spike-confirmed to run in PySR 1.5.10.
"""
from __future__ import annotations


def make_sobolev_loss(lam: float, h: float = 1e-4) -> str:
    """Return a Julia `loss_function` string with λ and h injected as literals."""
    return (
        "function loss_function(tree, dataset::Dataset{T,L}, options) where {T,L}\n"
        "    prediction, complete = eval_tree_array(tree, dataset.X, options)\n"
        "    if !complete || any(isnan, prediction) || any(isinf, prediction)\n"
        "        return L(Inf)\n"
        "    end\n"
        "    n = length(prediction)\n"
        "    residual = prediction .- dataset.y\n"
        "    mse = sum(residual .^ 2) / n\n"
        f"    h = T({h!r})\n"
        "    X2 = copy(dataset.X)\n"
        "    @inbounds X2[1, :] .+= h\n"
        "    pred2, complete2 = eval_tree_array(tree, X2, options)\n"
        "    if !complete2 || any(isnan, pred2) || any(isinf, pred2)\n"
        "        return L(Inf)\n"
        "    end\n"
        "    grad = (pred2 .- prediction) ./ h\n"
        "    gdiff = grad .- dataset.weights\n"
        "    gmse = sum(gdiff .^ 2) / n\n"
        f"    return mse + L({float(lam)!r}) * gmse\n"
        "end\n"
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_stage9_loss_string.py -q`
Expected: 2 passed. (If the `h` literal assertion fails because `repr(1e-4)` prints `0.0001`, the test already accepts that form.)

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/sobolev_loss.py tests/test_stage9_loss_string.py
git commit -m "Stage 9: make_sobolev_loss Julia loss-string builder

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: the per-fidelity target-gradient weights

The lever-#1-gotcha-prone piece: build the per-row target gradient in the SAME row order as `_build_training_matrix` (LF then HF; point-major/k-minor), using each row's OWN fidelity GP, normalized to the (x0, std_k) space.

**Files:**
- Modify: `src/priya_forecast/sobolev_loss.py`
- Test: `tests/test_stage9_weights.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage9_weights.py
import numpy as np
from priya_forecast.sobolev_loss import sobolev_target_weights


class _GP:
    """logP = s*theta[idx] + base(k); ∂logP/∂theta = s (k-independent)."""
    def __init__(self, idx, s):
        self.idx, self.s = idx, s
    def predict(self, theta, k, z):
        k = np.asarray(k, float)
        return np.exp(self.s * float(theta[self.idx]) + 0.5 * k)  # P; logP linear in theta


def test_per_fidelity_normalized_gradient():
    # 2 sweep points, 3 k-bins, param idx 0. LF slope 0.2, HF slope 0.4.
    nk = 3
    k = np.linspace(0.01, 0.04, nk)
    params = np.zeros((2, 11)); params[1, 0] = 1.0   # two theta values for idx 0
    payload = {
        "params_lf": params, "params_hf": params,
        "kfkms_lf_z": np.tile(k, (2, 1)), "kfkms_hf_z": np.tile(k, (2, 1)),
    }
    # std_k per the norm; x range width = (x_param_max - x_param_min)
    norm_std = np.full(nk, 2.0)                     # std_flux on the k_grid
    w = sobolev_target_weights(
        payload=payload, param_idx=0, gp_lf=_GP(0, 0.2), gp_hf=_GP(0, 0.4),
        z=3.6, x_param_min=0.0, x_param_max=1.0, std_flux=norm_std, norm_k_grid=k, h=1e-3)
    # order: LF rows (2 points x 3 k) then HF rows. LF grad_phys=0.2 -> /std=2 *width=1 => 0.1
    n_lf = 2 * nk
    np.testing.assert_allclose(w[:n_lf], 0.1, rtol=1e-3)   # LF: 0.2 * 1.0 / 2.0
    np.testing.assert_allclose(w[n_lf:], 0.2, rtol=1e-3)   # HF: 0.4 * 1.0 / 2.0
    assert w.shape == (2 * n_lf,)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_stage9_weights.py -q`
Expected: FAIL — `sobolev_target_weights` not defined.

- [ ] **Step 3: Implement `sobolev_target_weights`**

Append to `src/priya_forecast/sobolev_loss.py`:

```python
import numpy as np


def _fidelity_grad_weights(*, params, kfkms, gp, param_idx, z, width, std_on_k, norm_k_grid, h):
    """Per-row normalized target gradient for one fidelity, point-major/k-minor.

    weight = (∂logP/∂θ_phys) · width / std_k   (width = x_param_max − x_param_min)
    Rows are ordered point-major (k varies fastest), matching _build_training_matrix.
    """
    n_points = params.shape[0]
    rows = []
    for j in range(n_points):
        k_j = np.asarray(kfkms[j], dtype=float)
        theta = np.asarray(params[j], dtype=float)
        step = h * max(abs(float(theta[param_idx])), 1.0)
        tp = theta.copy(); tp[param_idx] += step
        tm = theta.copy(); tm[param_idx] -= step
        lp_p = np.log(np.asarray(gp.predict(tp, k_j, z), dtype=float))
        lp_m = np.log(np.asarray(gp.predict(tm, k_j, z), dtype=float))
        grad_phys = (lp_p - lp_m) / (2.0 * step)             # ∂logP/∂θ_phys per k
        std_k = np.interp(k_j, np.asarray(norm_k_grid, float), np.asarray(std_on_k, float))
        rows.append(grad_phys * width / std_k)                # normalized to (x0, std)
    return np.concatenate(rows)


def sobolev_target_weights(*, payload, param_idx, gp_lf, gp_hf, z,
                           x_param_min, x_param_max, std_flux, norm_k_grid, h=1e-3):
    """Per-row Sobolev target gradient matching X_act row order (LF rows then HF)."""
    width = float(x_param_max) - float(x_param_min)
    w_lf = _fidelity_grad_weights(
        params=np.asarray(payload["params_lf"], float), kfkms=payload["kfkms_lf_z"],
        gp=gp_lf, param_idx=param_idx, z=z, width=width, std_on_k=std_flux,
        norm_k_grid=norm_k_grid, h=h)
    w_hf = _fidelity_grad_weights(
        params=np.asarray(payload["params_hf"], float), kfkms=payload["kfkms_hf_z"],
        gp=gp_hf, param_idx=param_idx, z=z, width=width, std_on_k=std_flux,
        norm_k_grid=norm_k_grid, h=h)
    return np.concatenate([w_lf, w_hf])
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_stage9_weights.py -q`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/sobolev_loss.py tests/test_stage9_weights.py
git commit -m "Stage 9: per-fidelity normalized Sobolev target-gradient weights

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: config knobs

**Files:**
- Modify: `src/priya_forecast/single_z/config.py` (`PySRConfig`)
- Test: `tests/test_stage9_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage9_config.py
from priya_forecast.single_z.config import PySRConfig


def test_sobolev_defaults_off():
    c = PySRConfig()
    assert c.use_sobolev is False and c.sobolev_lambda == 1.0
    c.validate()


def test_sobolev_lambda_validated():
    import pytest
    c = PySRConfig(use_sobolev=True, sobolev_lambda=-1.0)
    with pytest.raises(ValueError, match="sobolev_lambda"):
        c.validate()
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_stage9_config.py -q`
Expected: FAIL — `PySRConfig` has no `use_sobolev`.

- [ ] **Step 3: Add the fields**

In `PySRConfig` (config.py) add fields:
```python
    use_sobolev: bool = False
    sobolev_lambda: float = 1.0
```
In `PySRConfig.validate()` add:
```python
        if self.use_sobolev and self.sobolev_lambda <= 0:
            raise ValueError("sobolev_lambda must be > 0 when use_sobolev=True.")
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_stage9_config.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/single_z/config.py tests/test_stage9_config.py
git commit -m "Stage 9: PySRConfig use_sobolev + sobolev_lambda

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: wire Sobolev into `refit_1d_for_param`

When `use_sobolev`, set `loss_function = make_sobolev_loss(λ, h)` (drop `elementwise_loss`/`loss_function` conflicts like the ANOVA path), build the weights via `sobolev_target_weights` (using `payload`, `ranges`, the computed `norm`, and the gps), and pass them to `model.fit(..., weights=...)`.

**Files:**
- Modify: `src/priya_forecast/refit_1d_pysr.py` (`refit_1d_for_param`)
- Test: `tests/test_stage9_refit_wiring.py`

- [ ] **Step 1: Read the exact fit call**

In `refit_1d_for_param`, find: the `norm` computed (carries `mean_flux`,`std_flux`,`k_grid`), the `_build_training_matrix(...)` call returning `(X_act, Y_act, ranges, fidelity_arrays)`, the `args = dict(...); args.update(pysr_kwargs or {})`, the `model = PySRRegressor(**args)`, and `model.fit(X_act, Y_act.reshape(-1,1))`. Add a kwarg `use_sobolev: bool = False, sobolev_lambda: float = 1.0, sobolev_h: float = 1e-4` to `refit_1d_for_param`'s signature.

- [ ] **Step 2: Write the failing test** (stubbed — no PySR; verify the loss string + weights are assembled and passed)

```python
# tests/test_stage9_refit_wiring.py
import numpy as np
from priya_forecast import refit_1d_pysr as R


def test_refit_assembles_sobolev_loss_and_weights(monkeypatch):
    captured = {}

    class _FakePySR:
        def __init__(self, **kw): captured["kwargs"] = kw
        def fit(self, X, y, **kw): captured["fit"] = (X.shape, kw)
        @property
        def equations_(self):
            import pandas as pd
            return pd.DataFrame({"equation": ["x0"], "complexity": [1], "loss": [0.0]})
    monkeypatch.setattr(R, "PySRRegressor", _FakePySR, raising=False)

    # minimal payload via a stub _generate_1pvar_inline + stub gps
    class _GP:
        def predict(self, theta, k, z):
            k = np.asarray(k, float); return np.exp(0.3*float(theta[0]) + 0.5*k)
    k = np.linspace(0.01, 0.04, 4)
    params = np.zeros((5, 11)); params[:, 0] = np.linspace(0.8, 1.05, 5)
    payload = {"params_lf": params, "params_hf": params,
               "flux_lf_z": np.exp(0.3*params[:, :1] + 0.5*k[None, :]),
               "flux_hf_z": np.exp(0.3*params[:, :1] + 0.5*k[None, :]),
               "kfkms_lf_z": np.tile(k, (5, 1)), "kfkms_hf_z": np.tile(k, (5, 1))}
    monkeypatch.setattr(R, "_generate_1pvar_inline", lambda **kw: payload, raising=False)

    R.refit_1d_for_param(
        param_name="ns", z=3.6, k_grid=k, gp_lf=_GP(), gp_hf=_GP(),
        log_space=True, use_sobolev=True, sobolev_lambda=3.0, sobolev_h=1e-4)

    assert "loss_function" in captured["kwargs"]
    assert "3.0" in captured["kwargs"]["loss_function"]    # lambda injected
    assert "elementwise_loss" not in captured["kwargs"]    # conflict dropped
    w = captured["fit"][1].get("weights")
    assert w is not None and w.shape[0] == 2 * (5 * 4)     # LF+HF rows
```

(If `_generate_1pvar_inline`'s real return dict has extra required keys the matrix builder needs, extend the stub `payload` to include them — read `_generate_1pvar_inline` to match.)

- [ ] **Step 3: Run to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_stage9_refit_wiring.py -q`
Expected: FAIL — `refit_1d_for_param` has no `use_sobolev` / doesn't set the loss/weights.

- [ ] **Step 4: Implement the wiring**

In `refit_1d_for_param`, after `norm` and `(X_act, Y_act, ranges, fidelity_arrays)` are available and before constructing `args`:

```python
    sobolev_weights = None
    if use_sobolev:
        from priya_forecast.sobolev_loss import make_sobolev_loss, sobolev_target_weights
        sobolev_weights = sobolev_target_weights(
            payload=payload, param_idx=param_idx, gp_lf=gp_lf, gp_hf=gp_hf, z=z,
            x_param_min=ranges["x_param_min"], x_param_max=ranges["x_param_max"],
            std_flux=norm.std_flux, norm_k_grid=norm.k_grid, h=sobolev_h)
```

When building `args`, when `use_sobolev`, set the loss and drop the conflict:
```python
    if use_sobolev:
        args["loss_function"] = make_sobolev_loss(sobolev_lambda, sobolev_h)
        args.pop("elementwise_loss", None)
```
(Keep the existing `if args.get("loss_function") is not None: args.pop("elementwise_loss", None)` guard — it already handles the conflict; the explicit pop is belt-and-braces.)

Change the fit call:
```python
    if sobolev_weights is not None:
        model.fit(X_act, Y_act.reshape(-1, 1), weights=sobolev_weights)
    else:
        model.fit(X_act, Y_act.reshape(-1, 1))
```

- [ ] **Step 5: Run to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_stage9_refit_wiring.py -q`
Expected: 1 passed.

- [ ] **Step 6: No-regression** — `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q -k "refit or pysr or training"` — all pass (Sobolev off by default leaves behavior unchanged).

- [ ] **Step 7: Commit**

```bash
git add src/priya_forecast/refit_1d_pysr.py tests/test_stage9_refit_wiring.py
git commit -m "Stage 9: wire Sobolev loss + per-fidelity weights into refit_1d_for_param

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: thread the flag (refit module + CLI + SLURM)

**Files:**
- Modify: `src/priya_forecast/single_z/refit.py` (`pysr_kwargs_for_cfg`, `refit_one_param_single_z`)
- Modify: `scripts/refit_one_param_single_z.py`
- Modify: `slurm/single_z_refit.slurm`
- Test: `tests/test_stage9_thread.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage9_thread.py
import inspect
from priya_forecast.single_z import refit as _r


def test_refit_one_param_passes_sobolev(monkeypatch):
    seen = {}
    def _fake_refit_1d_for_param(**kw):
        seen.update(kw)
        class _R:  # minimal stand-in
            equation_str = "x0"; pareto_complexity = 1; pareto_loss = 0.0
        return _R()
    monkeypatch.setattr(_r, "refit_1d_for_param", _fake_refit_1d_for_param, raising=True)
    monkeypatch.setattr(_r, "load_pareto_csv", lambda p: __import__("pandas").DataFrame(
        {"Equation": ["x0"], "Complexity": [1], "Loss": [0.0]}), raising=True)
    from priya_forecast.single_z.config import PipelineConfig
    cfg = PipelineConfig(mode="refit_and_forecast", redshift=3.6, target_space="log")
    cfg.pysr.use_sobolev = True
    cfg.pysr.sobolev_lambda = 4.0
    import numpy as np, tempfile
    _r.refit_one_param_single_z(param_name="ns", z=3.6, cfg=cfg, gp_lf=object(),
        gp_hf=object(), k_grid=np.linspace(0.01,0.04,4), out_dir=tempfile.mkdtemp())
    assert seen.get("use_sobolev") is True and seen.get("sobolev_lambda") == 4.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_stage9_thread.py -q`
Expected: FAIL — `use_sobolev` not threaded.

- [ ] **Step 3: Thread the flag**

In `single_z/refit.py: refit_one_param_single_z`, in the `refit_1d_for_param(...)` call, add:
```python
            use_sobolev=cfg.pysr.use_sobolev,
            sobolev_lambda=cfg.pysr.sobolev_lambda,
```
(The retry loop calls `refit_1d_for_param` — add the two kwargs to that call.)

In `scripts/refit_one_param_single_z.py`, add args:
```python
    p.add_argument("--use-sobolev", action="store_true")
    p.add_argument("--sobolev-lambda", type=float, default=1.0)
```
and set them on the `PySRConfig(...)` it builds: `use_sobolev=args.use_sobolev, sobolev_lambda=args.sobolev_lambda`.

In `slurm/single_z_refit.slurm`, after the `TARGET_SPACE` line add:
```bash
USE_SOBOLEV=${USE_SOBOLEV:-0}
SOBOLEV_ARGS=""
[ "$USE_SOBOLEV" = "1" ] && SOBOLEV_ARGS="--use-sobolev --sobolev-lambda ${SOBOLEV_LAMBDA:-1.0}"
```
and append `$SOBOLEV_ARGS` to the `"$PY" scripts/refit_one_param_single_z.py ...` invocation.

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_stage9_thread.py -q`
Expected: 1 passed. Also `bash -n slurm/single_z_refit.slurm` (no syntax error).

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/single_z/refit.py scripts/refit_one_param_single_z.py slurm/single_z_refit.slurm tests/test_stage9_thread.py
git commit -m "Stage 9: thread use_sobolev through refit module + CLI + SLURM

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: gated end-to-end — ns must become faithful

The decisive test: with Sobolev on, ns's refit equation passes `derivative_faithful` (what the value loss could not achieve).

**Files:**
- Test: `tests/test_stage9_end_to_end.py` (gated `RUN_SLOW_REFIT`)

- [ ] **Step 1: Write the gated test**

```python
# tests/test_stage9_end_to_end.py
import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_REFIT"),
    reason="needs PySR/Julia + emulator; set RUN_SLOW_REFIT=1",
)


def test_sobolev_makes_ns_gradient_faithful(tmp_path):
    from pathlib import Path
    import numpy as np
    from priya_forecast.models.gp_model import GPModel
    from priya_forecast.parameters import PARAM_NAMES, get_param, fiducial_vector
    from priya_forecast.single_z.config import PipelineConfig
    from priya_forecast.single_z import refit as _r
    from priya_forecast.single_z.refit import kodiaq_k_grid
    from priya_forecast.models.pysr_model import load_pareto_csv
    from priya_forecast.single_z.forecast import (
        _filter_fisher_safe, per_param_local_norm, _refit_from_row)
    from priya_forecast.derivative_gate import (
        gp_param_gradient, equation_param_gradient, derivative_faithful)
    from priya_forecast.single_z.training_data import load_1pvar

    k = kodiaq_k_grid(0.001, 0.04, 48)
    gp_lf = GPModel(basedir="data/kodiaq_gp", fidelity="lf", kf=k)
    gp_hf = GPModel(basedir="data/kodiaq_gp", fidelity="hf", kf=k)
    cfg = PipelineConfig(mode="refit_and_forecast", redshift=3.6, target_space="log")
    cfg.pysr.use_sobolev = True; cfg.pysr.sobolev_lambda = 1.0; cfg.pysr.niterations = 40
    _r.refit_one_param_single_z(param_name="ns", z=3.6, cfg=cfg, gp_lf=gp_lf,
        gp_hf=gp_hf, k_grid=k, out_dir=str(tmp_path))

    df = load_pareto_csv(Path(tmp_path) / "pareto_ns.csv")
    safe = _filter_fisher_safe(df, n_features=3).sort_values("Loss")
    meta = get_param("ns"); fid = np.asarray(fiducial_vector(), float)
    tgt = gp_param_gradient(gp=gp_hf, fid=fid, k_grid=k, z=3.6,
                            param_idx=PARAM_NAMES.index("ns"))
    d = load_1pvar(param_name="ns", z=3.6, data_dir="data/single_z_1pvar")
    kg = d["kfkms_lf_z"][0]
    norm = per_param_local_norm(flux_lf_z=d["flux_lf_z"], k_grid=kg,
        param_min=float(meta.prior[0]), param_max=float(meta.prior[1]), log_space=True)
    passed = False
    for _, row in safe.iterrows():
        cand = _refit_from_row(equation_str=str(row["Equation"]),
            complexity=int(row["Complexity"]), loss=float(row["Loss"]), df=df,
            param_name="ns", z=3.6, meta=meta, k_grid=kg, norm=norm, log_space=True)
        g = equation_param_gradient(refit=cand, fid_value=float(meta.fid),
            k_grid=np.asarray(kg, float), z=3.6)
        if derivative_faithful(cand_grad=g, target_grad=tgt, tol=0.25):
            passed = True; break
    assert passed, "Sobolev refit produced no derivative-faithful ns equation"
```

- [ ] **Step 2: Run gated (cluster venv)**

Run: `PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full RUN_SLOW_REFIT=1 .venv/bin/python -m pytest tests/test_stage9_end_to_end.py -q`
Expected: PASS. **If it FAILS, this is the λ-tuning signal** — re-run sweeping `sobolev_lambda` ∈ {0.3, 1, 3, 10} and pick the smallest λ that makes ns pass without wrecking value accuracy; record the chosen λ. Do NOT proceed to production until ns passes.

- [ ] **Step 3: Commit**

```bash
git add tests/test_stage9_end_to_end.py
git commit -m "Stage 9: gated e2e — Sobolev makes ns gradient-faithful

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: production run + COMPARISON + sweep + HANDOFF

**Files:**
- Create: `configs/single_z/stage9_z3.6.yaml`
- Create (by running): `results/single_z_stage9/COMPARISON.md`
- Modify: `HANDOFF.md`

- [ ] **Step 1: Production config**

```yaml
# configs/single_z/stage9_z3.6.yaml — single-z z=3.6, Sobolev refits + gate.
mode: forecast_only
redshift: 3.6
output_dir: results/single_z_stage9/
parameters: [ns, Ap, hub, omegamh2, herei, heref, alphaq, hireionz, bhfeedback, dtau0, tau0]
k_range: {min: 0.001, max: 0.04}
data: {source: kodiaq, cov_scale: 1.0, conservative: true, mock_data: gp}
gp: {basedir: data/kodiaq_gp}
combine: additive
target_space: log
derivative_tol: 0.25
pareto_csvs: {source: from_refit}
fisher: {step_frac: 0.01, rel_tol: 0.01}
```

- [ ] **Step 2: Cluster refit array (Sobolev on) + forecast**

```bash
sbatch --account=yueyingn0 --export=ALL,REPO=$(pwd),BASEDIR=data/kodiaq_gp,\
OUTPUT_DIR=results/single_z_stage9,Z=3.6,TARGET_SPACE=log,USE_SOBOLEV=1,SOBOLEV_LAMBDA=<chosen> \
--array=0-10%3 slurm/single_z_refit.slurm
# after it finishes:
PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full \
  .venv/bin/python scripts/run_pipeline.py --config configs/single_z/stage9_z3.6.yaml
```

- [ ] **Step 3: Write `results/single_z_stage9/COMPARISON.md`**

Per-param gradient error (median |∂eq/∂θ ÷ ∂logP_GP/∂θ − 1| at fid) under: value-loss (Stage 8) vs ratio-response spike vs **Sobolev**; the gate pass-count; and which params (if any) still GP-slice. **Success = ns AND hub pass**, no regression on the params that already passed. Note the chosen λ.

- [ ] **Step 4: Full sweep + HANDOFF**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q` (all pass; gated skip). Update `HANDOFF.md`: Stage 9 done; the per-param gradient results; chosen λ; whether the multi-z mirror is warranted next.

- [ ] **Step 5: Commit**

```bash
git add configs/single_z/stage9_z3.6.yaml results/single_z_stage9/COMPARISON.md HANDOFF.md
git commit -m "Stage 9: production config, Sobolev vs value-loss COMPARISON, HANDOFF

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage:** §3.1 loss → Task 1; §3.2 per-fidelity + normalized weights → Task 2 (the chosen convention: target = `(∂logP_GP/∂θ_phys)·width/std_k`, matching the in-loss `∂eq/∂x0`); §3.3 config + wiring → Tasks 3–5; §3.4 reuse (gate validates) → Task 6; §4 evaluation (ns+hub gradient faithfulness, not single-z σ) → Tasks 6–7; §5 tests 1–4 → Tasks 1,2,6,4; §6 risks (λ tuning → Task 6 Step 2 sweep; per-fidelity → Task 2; weights-hijack → Task 4) covered.
- **Convention pinned (spec §3.2 offered two):** target gradient is `∂(normalized logP)/∂x0` computed as `(∂logP_GP/∂θ_phys)·(x_param_max−x_param_min)/std_k`. The in-loss `grad = (eq(x0+h)−eq(x0))/h` is the matching `∂eq/∂x0`. Both are derivatives w.r.t. the *normalized* x0, so they are directly comparable — no leftover chain-rule factor.
- **Type consistency:** `make_sobolev_loss(lam,h)`, `sobolev_target_weights(payload,param_idx,gp_lf,gp_hf,z,x_param_min,x_param_max,std_flux,norm_k_grid,h)`, the `refit_1d_for_param(..., use_sobolev, sobolev_lambda, sobolev_h)` kwargs, and `PySRConfig.use_sobolev/sobolev_lambda` are used consistently across tasks.
- **Known soft spots:** (1) Task 4's stub test must match the real `_generate_1pvar_inline` payload keys the matrix builder reads — flagged in the task. (2) `λ` is the live unknown; Task 6 Step 2 is an explicit sweep-and-pick gate before production. (3) the in-loss finite-diff doubles tree-eval cost — acceptable on SLURM, watch wall-time.
