# Single-z Stage 6 (log(P) target) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox (`- [ ]`) steps.

**Goal:** Switch the per-parameter PySR symbolic-regression target from `P_F` to `log(P_F)`, with a log-space combine, behind a `target_space: linear | log` config flag.

**Architecture:** A `log_space: bool` flag, default `False`, threaded through the refit (`refit_1d_pysr`), the result container (`Refit1DResult`), the combine (`AdditiveTaylorModel`), and the forecast (`forecast.py`). When set, the SR equation is trained on normalized `log(P_F)` and the additive-Taylor combine works in log-space (additive in log = multiplicative in P). The linear path is untouched (default), so the two are directly comparable.

**Tech Stack:** Python 3.11, numpy, sympy, PySR, pytest. Modifies `refit_1d_pysr.py`, `refit_taylor.py`, `single_z/{config,forecast,combine,refit,pipeline}.py`.

**Spec:** `docs/superpowers/specs/2026-05-19-single-z-stage6-log-target-design.md`.

**Branch:** `single_z_forecast_clean`. Test command: `PYTHONPATH=src pytest <file> -v`.

---

## File Structure

| File | Change |
|------|--------|
| `src/priya_forecast/single_z/config.py` | `target_space` field + `VALID_TARGET_SPACES` + validation |
| `src/priya_forecast/refit_1d_pysr.py` | `log_space` param on `compute_local_normalization`, `_build_training_matrix`, `refit_1d_for_param`; `log_space` field on `Refit1DResult` + `predict_log` |
| `src/priya_forecast/refit_taylor.py` | `log_space` field on `AdditiveTaylorModel` + log-space combine branch |
| `src/priya_forecast/single_z/combine.py` | thread `log_space` into `build_combined_model` |
| `src/priya_forecast/single_z/forecast.py` | `log_space` on `per_param_local_norm` + `build_refit_from_pareto`; `run_three_fisher` threads it |
| `src/priya_forecast/single_z/refit.py` + `pipeline.py` | thread `log_space` from `cfg.target_space` |
| `configs/single_z/example.yaml` | `target_space: log` |

---

## Task 1: config — `target_space` flag

**Files:** Modify `src/priya_forecast/single_z/config.py`, `configs/single_z/example.yaml`, `tests/test_single_z_pipeline.py`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_single_z_pipeline.py`:

```python
def test_target_space_default_and_validation(tmp_path: Path):
    """PipelineConfig has target_space, default 'linear'; bad value rejected."""
    basedir = _basedir(tmp_path)
    good = _write(tmp_path, "t.yaml", f"gp:\n  basedir: {basedir}\n")
    assert load_config(good).target_space == "linear"
    logc = _write(tmp_path, "tl.yaml",
                  f"target_space: log\ngp:\n  basedir: {basedir}\n")
    assert load_config(logc).target_space == "log"
    bad = _write(tmp_path, "tb.yaml",
                 f"target_space: sqrt\ngp:\n  basedir: {basedir}\n")
    with pytest.raises(ValueError, match="target_space"):
        load_config(bad)
```

- [ ] **Step 2: Run, expect FAIL** — `PYTHONPATH=src pytest tests/test_single_z_pipeline.py -k target_space -v`.

- [ ] **Step 3: Implement** — in `config.py`:
  - Add a module-level constant near the other `VALID_*` tuples:
    ```python
    VALID_TARGET_SPACES = ("linear", "log")
    ```
  - Add a field to `PipelineConfig` immediately after `pick`:
    ```python
        target_space: str = "linear"
    ```
  - In `PipelineConfig.validate()`, after the `pick` check, add:
    ```python
            if self.target_space not in VALID_TARGET_SPACES:
                raise ValueError(
                    f"target_space must be one of {VALID_TARGET_SPACES}."
                )
    ```
  - `load_config` reads flat top-level scalar keys via its generic `else: setattr` branch (same as `combine`/`pick`) — no loader change needed; verify by reading `load_config`.

- [ ] **Step 4: Update `configs/single_z/example.yaml`** — add a line near `combine:`:
  ```yaml
  target_space: log           # linear | log  — fit log(P) (Stage 6, recommended)
  ```

- [ ] **Step 5: Run, expect PASS** — `PYTHONPATH=src pytest tests/test_single_z_pipeline.py -k target_space -v`, then the whole file for no regression. If `test_shipped_example_yaml_loads_and_validates` checks fields, it will still pass (it does not assert `target_space`); if you want, add `assert cfg.target_space == "log"` to it.

- [ ] **Step 6: Commit**
```bash
git add src/priya_forecast/single_z/config.py configs/single_z/example.yaml tests/test_single_z_pipeline.py
git commit -m "Stage 6: target_space config flag (linear | log)"
```

---

## Task 2: `compute_local_normalization` — log-space

**Files:** Modify `src/priya_forecast/refit_1d_pysr.py`, `tests/test_normalization.py`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_normalization.py`:

```python
def test_compute_local_normalization_log_space():
    """log_space=True normalizes log(P_F): mean/std are of log-flux."""
    import numpy as np
    from priya_forecast.refit_1d_pysr import compute_local_normalization

    rng = np.random.default_rng(0)
    k = np.linspace(0.001, 0.04, 8)
    flux = rng.random((50, 8)) + 1.0  # strictly positive
    norm = compute_local_normalization(
        flux_lf_z=flux, k_grid=k, log_space=True,
        param_min=0.8, param_max=1.05,
    )
    np.testing.assert_allclose(norm.mean_flux, np.log(flux).mean(axis=0))
    np.testing.assert_allclose(norm.std_flux, np.log(flux).std(axis=0, ddof=0))


def test_compute_local_normalization_log_space_rejects_nonpositive():
    import numpy as np
    import pytest
    from priya_forecast.refit_1d_pysr import compute_local_normalization

    k = np.linspace(0.001, 0.04, 4)
    flux = np.ones((10, 4))
    flux[3, 2] = -0.5  # a non-positive entry
    with pytest.raises(ValueError, match="positive"):
        compute_local_normalization(flux_lf_z=flux, k_grid=k, log_space=True)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** — in `compute_local_normalization` (`refit_1d_pysr.py:723`), add `log_space: bool = False` as the last keyword parameter. Right after the `flux_lf_z` / `k_grid` `np.asarray` + shape check, insert:

```python
    if log_space:
        if np.any(flux_lf_z <= 0):
            raise ValueError(
                "log_space=True requires strictly positive flux; "
                "flux_lf_z has non-positive entries."
            )
        target = np.log(flux_lf_z)
    else:
        target = flux_lf_z
```

Then replace the two `flux_lf_z`-based statistics with `target`:
- `std_k_local = flux_lf_z.std(axis=0, ddof=0)` → `std_k_local = target.std(axis=0, ddof=0)`
- `mean_k = flux_lf_z.mean(axis=0)` → `mean_k = target.mean(axis=0)`

The `std_k_local = np.where(std_k_local > 0, std_k_local, 1.0)` floor stays as-is (it now floors log-space std too). The `mean_flux_global` branch is unchanged (single-z always passes `None`); leave it. Update the docstring to note log_space.

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit**
```bash
git add src/priya_forecast/refit_1d_pysr.py tests/test_normalization.py
git commit -m "Stage 6: compute_local_normalization log_space option"
```

---

## Task 3: `_build_training_matrix` + `refit_1d_for_param` — log-space

**Files:** Modify `src/priya_forecast/refit_1d_pysr.py`, `tests/test_refit_1d_pysr_pareto.py`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_refit_1d_pysr_pareto.py`:

```python
def test_build_training_matrix_log_space():
    """_build_training_matrix log_space normalizes log(flux)."""
    import numpy as np
    from priya_forecast.refit_1d_pysr import _build_training_matrix
    from priya_forecast.models.normalization import NormalizationSpec

    n_pts, n_k = 50, 6
    k = np.linspace(0.001, 0.04, n_k)
    flux = np.geomspace(10.0, 80.0, n_pts * n_k).reshape(n_pts, n_k)
    params = np.tile(np.linspace(0.8, 1.05, n_pts)[:, None], (1, 11))
    payload = dict(
        flux_lf_z=flux, flux_hf_z=flux,
        kfkms_lf_z=np.tile(k, (n_pts, 1)), kfkms_hf_z=np.tile(k, (n_pts, 1)),
        params_lf=params, params_hf=params,
    )
    log_flux = np.log(flux)
    norm = NormalizationSpec(
        param_min=0.0, param_max=1.0, k_min=float(k.min()), k_max=float(k.max()),
        mean_flux=log_flux.mean(axis=0),
        std_flux=np.where(log_flux.std(axis=0) > 0, log_flux.std(axis=0), 1.0),
        k_grid=k,
    )
    X, Y, ranges, farr = _build_training_matrix(
        payload=payload, param_idx=2, global_norm=norm, log_space=True,
    )
    # Y is normalized log-flux → mean ≈ 0
    assert abs(float(Y.mean())) < 1e-6
    # fidelity_arrays still expose raw flux for diagnostics
    np.testing.assert_allclose(farr["flux_lf"], flux)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `_build_training_matrix`** — add `log_space: bool = False` as the last keyword parameter (`refit_1d_pysr.py:321`). After `flux_lf = payload["flux_lf_z"]` / `flux_hf = payload["flux_hf_z"]` are loaded, insert the log transform used for normalization (keep the raw `flux_lf`/`flux_hf` names for the `fidelity_arrays` diagnostics — introduce separate `*_t` arrays):

```python
    if log_space:
        if np.any(flux_lf <= 0) or np.any(flux_hf <= 0):
            raise ValueError(
                "log_space=True requires strictly positive flux in the "
                "1pvar payload."
            )
        flux_lf_t = np.log(flux_lf)
        flux_hf_t = np.log(flux_hf)
    else:
        flux_lf_t = flux_lf
        flux_hf_t = flux_hf
```

Then change the two normalization lines to use the `*_t` arrays:
- `flux_lf_norm = (flux_lf - mean_k_lf[None, :]) / std_k_lf[None, :]` → `(flux_lf_t - mean_k_lf[None, :]) / std_k_lf[None, :]`
- `flux_hf_norm = (flux_hf - mean_k_hf[None, :]) / std_k_hf[None, :]` → `(flux_hf_t - mean_k_hf[None, :]) / std_k_hf[None, :]`

Leave the `fidelity_arrays` return (`flux_lf=flux_lf, flux_hf=flux_hf, ...`) on the **raw** arrays — `_validate_per_fidelity` compares against raw P_F via `result.predict()`.

- [ ] **Step 4: Implement `refit_1d_for_param`** — add `log_space: bool = False` as the last keyword parameter (`refit_1d_pysr.py:895`). In the body:
  - the `compute_local_normalization(...)` call (when `norm is None`) → add `log_space=log_space`.
  - the `_build_training_matrix(...)` call → add `log_space=log_space`.
  - Do **not** touch the `Refit1DResult(...)` constructor call in this task.
    Task 4 adds the `log_space` field to `Refit1DResult` *and* adds
    `log_space=log_space` to that constructor call. After this task,
    `refit_1d_for_param` accepts `log_space` and threads it into the
    normalization + training matrix (the testable behavior) but does not
    yet store it on the result — Task 4 completes that. This intermediate
    state is valid and tested.

- [ ] **Step 5: Run** — `PYTHONPATH=src pytest tests/test_refit_1d_pysr_pareto.py -k log_space -v` → PASS. Confirm `import priya_forecast.refit_1d_pysr` is clean.

- [ ] **Step 6: Commit**
```bash
git add src/priya_forecast/refit_1d_pysr.py tests/test_refit_1d_pysr_pareto.py
git commit -m "Stage 6: _build_training_matrix + refit_1d_for_param log_space"
```

---

## Task 4: `Refit1DResult` — `log_space` field + `predict_log`

**Files:** Modify `src/priya_forecast/refit_1d_pysr.py`, `tests/test_refit_1d_pysr_pareto.py`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_refit_1d_pysr_pareto.py`:

```python
def _hand_refit_log(equation_str, k, *, log_space):
    """A Refit1DResult with a hand-written equation, in linear or log space."""
    import numpy as np
    from priya_forecast.models.normalization import NormalizationSpec
    from priya_forecast.refit_1d_pysr import Refit1DResult, HF_RESOLUTION, LF_RESOLUTION

    nk = len(k)
    norm = NormalizationSpec(
        param_min=0.8, param_max=1.05, k_min=float(k.min()), k_max=float(k.max()),
        mean_flux=np.full(nk, 2.0 if log_space else 30.0),
        std_flux=np.full(nk, 0.5 if log_space else 5.0),
        k_grid=np.asarray(k, dtype=float),
    )
    return Refit1DResult(
        param_name="ns", z=3.6, equation_str=equation_str,
        pareto_complexity=3, pareto_loss=0.0,
        pareto_complexities=[3], pareto_losses=[0.0],
        x_param_min=0.8, x_param_max=1.05,
        k_min=float(k.min()), k_max=float(k.max()),
        lf_resolution=LF_RESOLUTION, hf_resolution=HF_RESOLUTION,
        fid_value=0.983, norm=norm, k_grid=np.asarray(k, dtype=float),
        wall_time_s=0.0, lf_train_mean_rel_err=0.0, hf_train_mean_rel_err=0.0,
        lf_train_max_rel_err=0.0, hf_train_max_rel_err=0.0,
        log_space=log_space,
    )


def test_refit1dresult_predict_log_consistency():
    """predict and predict_log are exp/log consistent in both spaces."""
    import numpy as np
    k = np.linspace(0.001, 0.04, 10)
    for log_space in (False, True):
        r = _hand_refit_log("x0 + x1", k, log_space=log_space)
        p = r.predict(theta_phys=0.98, k=k)
        plog = r.predict_log(theta_phys=0.98, k=k)
        assert np.all(p > 0)                       # raw P_F positive
        np.testing.assert_allclose(plog, np.log(p), rtol=1e-9)
        np.testing.assert_allclose(p, np.exp(plog), rtol=1e-9)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** — in `refit_1d_pysr.py`:
  - Add a field to the `Refit1DResult` dataclass, after `z_max`:
    ```python
        log_space: bool = False
    ```
  - In `refit_1d_for_param`, add `log_space=log_space` to the
    `Refit1DResult(...)` constructor call at the end of the function
    (Task 3 left this line untouched; this completes the threading so the
    flag is stored on the result).
  - Add a private denorm helper and `predict_log`, and make `predict` branch. The current `predict` (`refit_1d_pysr.py:247`) is:
    ```python
    def predict(self, theta_phys, k, resolution=HF_RESOLUTION, z=None):
        flux_norm = self.predict_normalized(theta_phys, k, resolution=resolution, z=z)
        return self.norm.denormalize_flux(
            flux_norm, np.asarray(k, dtype=float),
            z=(z if z is not None else self.z),
        )
    ```
    Replace it with:
    ```python
    def _predict_denorm(self, theta_phys, k, resolution=HF_RESOLUTION, z=None):
        """Denormalized prediction in the training space (log P_F if
        log_space, else raw P_F)."""
        flux_norm = self.predict_normalized(theta_phys, k, resolution=resolution, z=z)
        return self.norm.denormalize_flux(
            flux_norm, np.asarray(k, dtype=float),
            z=(z if z is not None else self.z),
        )

    def predict(self, theta_phys, k, resolution=HF_RESOLUTION, z=None):
        """Raw P_F. exp() applied when the equation was trained on log(P)."""
        val = self._predict_denorm(theta_phys, k, resolution=resolution, z=z)
        return np.exp(val) if self.log_space else val

    def predict_log(self, theta_phys, k, resolution=HF_RESOLUTION, z=None):
        """log(P_F). For a log-trained equation this is the native output;
        for a linear-trained one it is log() of the raw prediction."""
        val = self._predict_denorm(theta_phys, k, resolution=resolution, z=z)
        return val if self.log_space else np.log(val)
    ```

- [ ] **Step 4: Run, expect PASS** — `PYTHONPATH=src pytest tests/test_refit_1d_pysr_pareto.py -k predict_log -v`. Also confirm `import priya_forecast.refit_1d_pysr` is clean and the Task 3 log-space tests still pass (the `Refit1DResult(log_space=...)` line is now in place).

- [ ] **Step 5: Commit**
```bash
git add src/priya_forecast/refit_1d_pysr.py tests/test_refit_1d_pysr_pareto.py
git commit -m "Stage 6: Refit1DResult log_space field + predict_log"
```

---

## Task 5: `AdditiveTaylorModel` — log-space combine

**Files:** Modify `src/priya_forecast/refit_taylor.py`, `tests/test_refit_taylor.py`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_refit_taylor.py`:

```python
def test_additive_taylor_log_space_anchor_identity():
    """log_space combine returns the GP anchor exactly at θ=fid."""
    import numpy as np
    from priya_forecast.models.gp_model import MockGPModel
    from priya_forecast.parameters import PARAM_NAMES, fiducial_vector
    from priya_forecast.refit_taylor import AdditiveTaylorModel

    gp = MockGPModel()
    k = np.linspace(0.001, 0.04, 16)
    fid = np.asarray(fiducial_vector(), dtype=float)
    model = AdditiveTaylorModel(
        gp=gp, fid=fid, refits={n: None for n in PARAM_NAMES},
        global_norm=None, k_grid=k, z=3.6, mode="local_anchored",
        log_space=True,
    )
    np.testing.assert_allclose(
        model.predict(fid, k, 3.6), gp.predict(fid, k, 3.6), rtol=1e-9,
    )


def test_additive_taylor_log_space_rejects_multi_d():
    """log_space is only valid with the local_anchored mode."""
    import numpy as np
    import pytest
    from priya_forecast.models.gp_model import MockGPModel
    from priya_forecast.parameters import PARAM_NAMES, fiducial_vector
    from priya_forecast.refit_taylor import AdditiveTaylorModel

    with pytest.raises(ValueError, match="log_space"):
        AdditiveTaylorModel(
            gp=MockGPModel(), fid=np.asarray(fiducial_vector(), dtype=float),
            refits={n: None for n in PARAM_NAMES}, global_norm=None,
            k_grid=np.linspace(0.001, 0.04, 8), z=3.6, mode="multi_d",
            log_space=True,
        )
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** — in `refit_taylor.py`, `AdditiveTaylorModel`:
  - Add a field after `mode`:
    ```python
        log_space: bool = False
    ```
  - In `__post_init__`, after the existing `mode` validity check, add:
    ```python
            if self.log_space and self.mode != "local_anchored":
                raise ValueError(
                    "log_space=True is only supported with "
                    "mode='local_anchored'."
                )
    ```
  - After `self._p_gp_fid` is cached, add the log-space caches (only when `log_space`):
    ```python
            if self.log_space:
                if np.any(self._p_gp_fid <= 0):
                    raise ValueError(
                        "log_space combine: GP P_F(fid) has non-positive "
                        "entries — cannot take log."
                    )
                self._log_p_gp_fid = np.log(self._p_gp_fid)
                self._eq_at_fid_logpf: dict[str, np.ndarray] = {}
                for pname, r in self.refits.items():
                    if r is None:
                        continue
                    i = PARAM_NAMES.index(pname)
                    self._eq_at_fid_logpf[pname] = r.predict_log(
                        theta_phys=float(self.fid[i]), k=self.k_grid,
                        resolution=HF_RESOLUTION_FOR_COMBINE,
                    )
    ```
  - In `predict`, after the `k`/`z` guards and BEFORE the existing `if self.mode == "local_anchored":` block, add a log-space branch:
    ```python
            if self.log_space:
                out_log = self._log_p_gp_fid.copy()
                for pname, r in self.refits.items():
                    if r is None:
                        continue
                    i = PARAM_NAMES.index(pname)
                    if float(theta[i]) == float(self.fid[i]):
                        continue
                    log_at_theta = r.predict_log(
                        theta_phys=float(theta[i]), k=self.k_grid,
                        resolution=HF_RESOLUTION_FOR_COMBINE,
                    )
                    out_log = out_log + (log_at_theta - self._eq_at_fid_logpf[pname])
                for pname, r in self.refits.items():
                    if r is not None:
                        continue
                    i = PARAM_NAMES.index(pname)
                    if float(theta[i]) == float(self.fid[i]):
                        continue
                    t_only = self.fid.copy()
                    t_only[i] = theta[i]
                    p_slice = np.asarray(
                        self.gp.predict(t_only, self.k_grid, self.z), dtype=float
                    )
                    if np.any(p_slice <= 0):
                        raise ValueError(
                            f"log_space combine: GP slice for {pname!r} has "
                            f"non-positive P_F — cannot take log."
                        )
                    out_log = out_log + (np.log(p_slice) - self._log_p_gp_fid)
                return np.exp(out_log)
    ```

- [ ] **Step 4: Run, expect PASS** — `PYTHONPATH=src pytest tests/test_refit_taylor.py -k log_space -v`. The existing linear `AdditiveTaylorModel` tests must still pass (the `log_space` field defaults `False`).

- [ ] **Step 5: Commit**
```bash
git add src/priya_forecast/refit_taylor.py tests/test_refit_taylor.py
git commit -m "Stage 6: AdditiveTaylorModel log-space combine branch"
```

---

## Task 6: `combine.py` + `forecast.py` — thread `log_space`

**Files:** Modify `src/priya_forecast/single_z/combine.py`, `src/priya_forecast/single_z/forecast.py`, `tests/test_single_z_combine.py`, `tests/test_single_z_forecast.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_single_z_combine.py`:
```python
def test_build_combined_model_additive_log_space():
    from priya_forecast.models.gp_model import MockGPModel
    from priya_forecast.refit_taylor import AdditiveTaylorModel
    from priya_forecast.single_z.combine import build_combined_model

    model = build_combined_model(
        combine_mode="additive", gp=MockGPModel(), fid=_fid(),
        refits=_none_refits(), k_grid=np.linspace(0.001, 0.04, 12), z=3.6,
        log_space=True,
    )
    assert isinstance(model, AdditiveTaylorModel)
    assert model.log_space is True
```

Append to `tests/test_single_z_forecast.py`:
```python
def test_per_param_local_norm_log_space():
    """per_param_local_norm log_space normalizes log(flux)."""
    rng = np.random.default_rng(0)
    k = np.linspace(0.001, 0.04, 10)
    flux = rng.random((50, 10)) + 1.0
    norm = per_param_local_norm(
        flux_lf_z=flux, k_grid=k, param_min=0.8, param_max=1.05,
        log_space=True,
    )
    np.testing.assert_allclose(norm.mean_flux, np.log(flux).mean(axis=0))
    assert np.all(norm.std_flux > 0)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `combine.py`** — `build_combined_model` adds `log_space: bool = False` (last keyword param) and passes `log_space=log_space` into the `AdditiveTaylorModel(...)` constructor in the `additive` branch.

- [ ] **Step 4: Implement `forecast.py`**
  - `per_param_local_norm` — add `log_space: bool = False`. When `True`: reject non-positive flux (`if np.any(flux_lf_z <= 0): raise ValueError("log_space ... requires positive flux")`), then compute `mean_flux`/`std_flux` from `np.log(flux_lf_z)` instead of `flux_lf_z`. Keep the `np.where(std > 0, std, 1.0)` floor.
  - `build_refit_from_pareto` — add `log_space: bool = False`. Pass `log_space=log_space` to the `per_param_local_norm(...)` call, and add `log_space=log_space` to the `Refit1DResult(...)` constructor.
  - `run_three_fisher` — read `log_space = (cfg.target_space == "log")` and pass `log_space=log_space` to BOTH `build_combined_model(...)` calls (`perfect_model` and `pysr_model`). It does NOT call `build_refit_from_pareto`.

- [ ] **Step 5: Run, expect PASS** — `PYTHONPATH=src pytest tests/test_single_z_combine.py tests/test_single_z_forecast.py -v`.

- [ ] **Step 6: Commit**
```bash
git add src/priya_forecast/single_z/combine.py src/priya_forecast/single_z/forecast.py tests/test_single_z_combine.py tests/test_single_z_forecast.py
git commit -m "Stage 6: thread log_space through combine.py and forecast.py"
```

---

## Task 7: `pipeline.py` + `refit.py` — wire `target_space`

**Files:** Modify `src/priya_forecast/single_z/pipeline.py`, `src/priya_forecast/single_z/refit.py`, `tests/test_single_z_pipeline.py`.

- [ ] **Step 1: Implement `single_z/refit.py`** — in `refit_one_param_single_z`, add `log_space=(cfg.target_space == "log")` to the `refit_1d_for_param(...)` call.

- [ ] **Step 2: Implement `pipeline.py`** — in `run_forecast_only`, the loop that calls `_fc.build_refit_from_pareto(...)` — add `log_space=(cfg.target_space == "log")` to that call. `run_refit_and_forecast` needs no change (its refits come from `refit_one_param_single_z`, fixed in Step 1; `run_three_fisher` reads `cfg.target_space` itself).

- [ ] **Step 3: Write the gated end-to-end test** — append to `tests/test_single_z_pipeline.py`:

```python
@pytest.mark.skipif(
    not (RUN_SLOW_REFIT and LYAEMU_AVAILABLE and GP_BASEDIR.exists()),
    reason="gated on RUN_SLOW_REFIT=1 + lyaemu + data/kodiaq_gp/ (runs PySR)",
)
def test_refit_and_forecast_log_space_end_to_end(tmp_path: Path):
    """refit_and_forecast with target_space=log runs end to end."""
    import numpy as np
    from priya_forecast.single_z.pipeline import run

    cfg = PipelineConfig(
        mode="refit_and_forecast", redshift=3.6,
        output_dir=str(tmp_path / "out"),
        gp=GPConfig(basedir=str(GP_BASEDIR)),
        parameters=["ns", "Ap"],
        k_range=KRange(min=0.001, max=0.04),
        data=DataConfig(source="kodiaq"),
        target_space="log",
    )
    result = run(cfg)
    for label in ("GP", "perfect_1D", "PySR"):
        s = result["sigmas"][label]
        assert s.shape == (2,)
        assert np.all(np.isfinite(s)) and np.all(s > 0)
```

- [ ] **Step 4: Run** — `PYTHONPATH=src pytest tests/test_single_z_pipeline.py -q` (new gated test SKIPs; no regression).

- [ ] **Step 5: Commit**
```bash
git add src/priya_forecast/single_z/pipeline.py src/priya_forecast/single_z/refit.py tests/test_single_z_pipeline.py
git commit -m "Stage 6: wire target_space through pipeline + refit"
```

---

## Task 8: σ_perfect_1D ≡ σ_GP in log-space + verification

**Files:** Modify `tests/test_single_z_forecast.py`; verification only otherwise.

- [ ] **Step 1: Write the test** — append to `tests/test_single_z_forecast.py`:

```python
def test_run_three_fisher_log_space_perfect_equals_gp():
    """In log-space too, perfect_1D σ ≈ GP σ (shared covariance, exact 1D)."""
    from priya_forecast.models.gp_model import MockGPModel
    from priya_forecast.parameters import PARAM_NAMES, fiducial_vector
    from priya_forecast.single_z.config import (
        PipelineConfig, DataConfig, FisherConfig,
    )
    from priya_forecast.single_z.forecast import run_three_fisher

    gp = MockGPModel()
    cfg = PipelineConfig(
        mode="forecast_only", redshift=3.6, parameters=["ns", "Ap"],
        combine="additive", target_space="log",
        data=DataConfig(source="eboss_dr14"),
        fisher=FisherConfig(step_frac=0.05, rel_tol=0.05),
    )
    results = run_three_fisher(
        cfg=cfg, gp=gp, fid=np.asarray(fiducial_vector(), dtype=float),
        refits={n: None for n in PARAM_NAMES},
    )
    # exact analytically; finite adaptive stencil → allclose, not equal
    np.testing.assert_allclose(
        results["perfect_1D"].sigma, results["GP"].sigma, rtol=1e-3,
    )
```

- [ ] **Step 2: Run, expect PASS** — `PYTHONPATH=src pytest tests/test_single_z_forecast.py -k log_space -v`. If `run_three_fisher` does not yet read `cfg.target_space` (Task 6 Step 4 covers it), fix that. If the assertion fails beyond `rtol=1e-3`, do not loosen it blindly — diagnose: the log-space GP-slice combine should reproduce the GP's first derivative; a real gap means the log branch is wrong. Report findings.

- [ ] **Step 3: Full verification** — `PYTHONPATH=src pytest tests/ -q`. All pure tests pass, gated tests SKIP, no regression from the linear default.

- [ ] **Step 4: Commit**
```bash
git add tests/test_single_z_forecast.py
git commit -m "Stage 6: verify perfect_1D == GP holds in log-space"
```

---

## Done criteria

- `target_space: linear | log` config flag works end to end.
- `log` path: SR trains on normalized `log(P_F)`, the combine is additive-in-log, `predict` still returns raw `P_F`, Fisher is unchanged.
- σ_perfect_1D ≈ σ_GP holds in log-space (the comparison stays valid).
- All linear-path tests stay green (default unchanged); no regression.

Carryover: Stage 7 = multi-z Fisher `F = Σ_z F(z)`; Stage 8 = Sobolev derivative loss. After Stage 6 lands, re-run the z=3.6 refit_and_forecast with `target_space: log` and compare σ_PySR/σ_GP against the linear baseline to measure log(P)'s impact.
