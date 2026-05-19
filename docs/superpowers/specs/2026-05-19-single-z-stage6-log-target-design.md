# Single-z Stage 6 — log(P) SR target + log-space combine

**Date:** 2026-05-19
**Branch:** `single_z_forecast_clean`
**Status:** design approved, pending spec review

## 1. Background and motivation

The Stage 1–5 single-z pipeline is complete and runs end to end, but the
z=3.6 result exposed that the per-parameter PySR surrogate is
**value-accurate yet derivative-unfaithful**: σ_PySR/σ_GP came out
0.09×–27×, including physically impossible sub-1 ratios. A literature
review (2026-05-18) diagnosed this as the standard symbolic-regression
emulator pathology — a value loss does not constrain the gradient — and
gave a prioritized fix list. **#1 was: fit `log(P)` instead of `P`.**

`∂P/∂θ = P · ∂(log P)/∂θ`. Fitting `log(P)`:
- makes the SR loss target *fractional* error uniformly in (k, z) — which
  is the quantity the Fisher matrix is sensitive to;
- guarantees positivity;
- collapses the ~1-dex dynamic range so the genetic search is not
  dominated by a few high-P points;
- turns the multiplicative P1D additive in log-space → simpler, smoother
  SR trees, which have better-behaved derivatives.

Stage 6 implements the `log(P)` target. Multi-z Fisher (`F = Σ_z F(z)`)
and the Sobolev derivative loss are deferred to later stages so the
isolated impact of `log(P)` can be measured first.

## 2. Scope and data flow

Stage 6 changes the **space the SR equation lives in**, end to end, behind
a `target_space: linear | log` config flag. The linear path is kept — the
scientific point is to *compare* linear vs log, so both must run; `log`
becomes the `example.yaml` default.

- **Training** (`refit_1d_pysr`): when `log_space`, the per-param
  `NormalizationSpec` is computed from `log(flux_lf_z)` and the SR target
  is `Y = (log P_F − mean_logk)/std_logk`. The regenerated
  `data/single_z_1pvar/` HDF5s are **unchanged** — they store raw `P_F`
  (the Stage 1 decision holds); `log` is applied when the training matrix
  is built.
- **`Refit1DResult`**: gains a `log_space` flag and a `predict_log`
  method. `predict` still returns raw `P_F`.
- **Combine** (`AdditiveTaylorModel`): a log-space branch — additive in
  log = multiplicative in P.
- **Fisher / `run_three_fisher` / likelihoods**: **unchanged** — the
  combined model still exposes `predict → raw P_F`. `∂P/∂θ = P ·
  ∂(logP)/∂θ` falls out by the chain rule.

**Property preserved:** σ_perfect_1D ≡ σ_GP still holds. The log-space
combine of exact GP log-slices reproduces the GP's *first* derivative
`∂P/∂θ` at fid exactly (`∂P/∂θ = P_GP(fid)·∂logP_GP/∂θ = ∂P_GP/∂θ`), so
the σ_GP / σ_PySR comparison stays valid.

## 3. Training side — `refit_1d_pysr`

Thread an optional `log_space: bool = False` through `refit_1d_for_param`,
`_build_training_matrix`, and `compute_local_normalization`. The default
`False` keeps every existing caller — including the multi-z refits —
byte-identical.

- `compute_local_normalization(..., log_space=False)`: when `True`,
  compute `mean_k` / `std_k` from `log(flux_lf_z)` instead of
  `flux_lf_z`. The returned `NormalizationSpec` then round-trips to
  `log P_F`.
- `_build_training_matrix(..., log_space=False)`: when `True`, normalize
  `log(flux)` rather than `flux` for both LF and HF (`flux_*_norm =
  (log flux_* − mean_logk)/std_logk`). The `(θ_norm, k_norm, resolution)`
  X-matrix is unchanged.
- `refit_1d_for_param(..., log_space=False)`: passes the flag down and
  stores it on the returned `Refit1DResult`.

`single_z/refit.py:refit_one_param_single_z` passes
`log_space=(cfg.target_space == "log")`.

## 4. `Refit1DResult` + the combine

### 4.1 `Refit1DResult`

New field `log_space: bool = False`. New method:

```
predict_log(theta_phys, k, resolution=HF) -> log P_F
```

Both `predict` and `predict_log` work regardless of training space; only
the cheap path differs:
- `log_space=True`: the bundled `NormalizationSpec` denormalizes to
  `log P_F`. `predict_log` returns the denormalized value directly;
  `predict` returns `exp(predict_log)`.
- `log_space=False`: the norm denormalizes to `P_F`. `predict` returns it
  directly; `predict_log` returns `log(predict)`.

`predict`'s external contract (returns raw `P_F`) is unchanged.

### 4.2 `AdditiveTaylorModel`

New field `log_space: bool = False`. When `True` (with the existing
`local_anchored` mode):

```
log P(θ, k) = log P_GP(fid, k)
            + Σ_i [ predict_log_i(θ_i) − predict_log_i(fid_i) ]
P_F(θ, k)   = exp( log P(θ, k) )
```

`__post_init__` caches `log P_GP(fid, k_grid)` and each per-param
`predict_log_i(fid_i)`. Un-refit parameters fall back to a GP 1D-slice
taken in log space: `log P_GP(θ_i-slice) − log P_GP(fid)`. At θ=fid all
deviations are zero → `exp(log P_GP(fid)) = P_GP(fid)` — anchor identity
holds.

`combine.build_combined_model` threads a `log_space` argument into the
`additive` branch.

## 5. Forecast wiring + config

- **Config** (`single_z/config.py`): new field
  `PipelineConfig.target_space: str = "linear"`, validated against
  `VALID_TARGET_SPACES = ("linear", "log")`. `load_config` reads the
  top-level `target_space` key. `configs/single_z/example.yaml` set to
  `target_space: log`.
- **`forecast.py`**:
  - `per_param_local_norm(..., log_space=False)` — when `True`, compute
    `mean/std` from `log(flux_lf_z)`.
  - `build_refit_from_pareto(..., log_space=False)` — passes the flag to
    `per_param_local_norm` and sets `Refit1DResult.log_space`.
  - `run_three_fisher` reads `cfg.target_space`, threads `log_space` into
    `build_refit_from_pareto` and `build_combined_model`.
- **`pipeline.py`**: `run_forecast_only` / `run_refit_and_forecast` are
  unchanged except that the `log_space` flag now flows from
  `cfg.target_space` through the calls above.

## 6. Error handling

- `log(P_F)` requires `P_F > 0`. The regenerated 1pvar data and the GP
  anchor are strictly positive (verified in Stage 1). If a non-positive
  flux is encountered when `log_space`, raise a clear `ValueError` naming
  the parameter rather than emitting `nan`/`-inf` silently.
- `target_space` outside `{linear, log}` → `ValueError` from
  `PipelineConfig.validate`.

## 7. Testing

- `compute_local_normalization` / `per_param_local_norm`: log vs linear —
  a log-space spec's `mean_flux` equals `mean(log flux)`.
- `Refit1DResult`: `predict` and `predict_log` round-trip consistently in
  both spaces (`predict == exp(predict_log)`).
- `AdditiveTaylorModel` log-space: anchor identity at θ=fid
  (`predict(fid) == P_GP(fid)`); σ_perfect_1D ≡ σ_GP with all-None refits.
- Config: `target_space` default `linear`, `log` accepted, bad value
  rejected.
- A gated end-to-end `refit_and_forecast` run with `target_space: log`.
- All existing linear-path tests stay green (default `target_space`
  unchanged).

## 8. Out of scope (later stages)

- Multi-z Fisher `F = Σ_z F(z)` — Stage 7.
- Sobolev derivative-matching loss + derivative-validation gate — Stage 8.
- Auditing the additive-Taylor combine for dropped cross-terms.
