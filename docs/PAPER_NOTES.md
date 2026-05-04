# Paper notes: tricks and modifications beyond the original student pipeline

Read this end-to-end before writing the methods section. Each modification
is documented with **what was changed**, **why it was needed** (the
observation that motivated it), and **the file/memory note** where the
detail lives.

The "original design" reference is the student's
`/home/mfho/student_projects/priya_pysr/pysr_mf_given.py` +
`/home/mfho/student_projects/InferenceLyaData/mf_*.py` pipeline.

Cross-reference: `~/.claude/projects/-home-mfho-lya1d-priya-forecast/memory/MEMORY.md`.

---

## 1. Emulator and k-grid

### 1a. Switched to KODIAQ-SQUAD + XQ-100 production emulator
- **Original**: priya / InferenceLyaData emulator with k_max ≈ 0.02 s/km
  (eBOSS-like).
- **Modified**: kodiaq_2_2_4_6-48-48/ on Greatlakes (cluster path
  `/nfs/turbo/umor-yueyingn/mfho/birdgroup/lya_xq100/kodiaq_2_2_4_6-48-48/`).
  Same 9 cosmo+thermal params; multi-fidelity emulator (LF + HF/hires).
- **Why**: production paper arXiv:2509.18271 uses this emulator. Also,
  the priya emulator gave singular GP Fisher at single z=3.6 (mean-flux
  pair dtau0/tau0 degenerate) regardless of cov.
- **Code**: `src/priya_forecast/models/gp_model.py::GPModel(basedir=...,
  fidelity="hf"|"lf", kf=...)`. Both LF and HF supported via the
  `fidelity` flag (LF passes `HRbasedir=None` to upstream `GPWrap`).

### 1b. k-grid 0.005 → 0.064 s/km (production range)
- **Original**: eBOSS DR14 k-grid (k ≤ 0.02 s/km).
- **Modified**: `np.linspace(0.005, 0.064, n_k)`. Production papers use
  this range (`docs/SLURM_COMMANDS.md` in `~/lya_emulator_full/`).
- **Why**: covers all three physical k-regimes (cosmic-variance floor at
  k<0.005, peculiar-velocity dip near k=0.02, LF resolution loss at
  k>0.04). See `memory/p1d_physics_regimes.md`.

---

## 2. PySR configuration changes

### 2a. Drop sin/cos from unary operators
- **Original**: `unary_operators=["sin","cos","exp","log","square","sqrt","inv"]`.
- **Modified**: `["exp","log","square","sqrt","inv"]` — no trig.
- **Why**: oscillatory derivatives wreck Fisher conditioning. With
  trig-heavy equations, σ_hybrid ratios came out 10³ to 10⁶× the GP
  because per-param gradients were dominated by trig high-frequency
  terms instead of the smooth k-shape the GP encodes.
- **Memory**: `memory/feedback_pysr_operators.md`.

### 2b. Constrain and lightly penalize `^`
- **Original**: `^` allowed with no constraints.
- **Modified**: `constraints={"^": (-1, 1)}` (arbitrary base, simple
  exponent only) + `complexity_of_operators={"^": 3}` (mild penalty).
- **Why**: untamed `^` produces `(complex)^(complex)` patterns that fit
  the mean but blow up at prior boundaries. Observed: Ap HF max rel-err
  was 19% before this change, 8% after.

### 2c. niter 20 → 50, multithreaded
- **Original**: niter=20, `parallelism="serial"`, `deterministic=True`.
- **Modified**: niter=50, `parallelism="multithreading", procs=4,
  deterministic=False`.
- **Why**: 4-5× wall-time speedup, mild fit-quality improvement.
  Reproducibility is sacrificed (results stable up to genetic-algorithm
  noise; `random_state=42` keeps the best-eq mostly stable).
- **Memory**: `memory/feedback_pysr_speed.md`. Full breakdown:
  `docs/PYSR_PERFORMANCE.md`.

### 2d. Reproducibility note
For paper-final fits we should re-run with `deterministic=True,
parallelism="serial"` to lock in bit-reproducible equations. See
`docs/PYSR_PERFORMANCE.md` "Reproducibility footnote".

---

## 3. Training data generation

### 3a. Inline 1pvar (no HDF5 dependency)
- **Original**: `1pvar/{lf,hf}_<param>_npoints50_datacorrFalse.hdf5`
  pre-saved files at fixed k-grid.
- **Modified**: `_generate_1pvar_inline(gp_lf, gp_hf, ...)` calls
  `gp.predict()` directly. No HDF5 file dependency.
- **Why**: lets us use ANY emulator (kodiaq vs priya) and ANY k-grid
  (production 0.005-0.064 vs eBOSS 0.001-0.02). Also enables multi-z.
- **Code**: `src/priya_forecast/refit_1d_pysr.py::_generate_1pvar_inline`.

### 3b. Sobol over (θ, z) for multi-z
- **Original**: linspace per z (n sims at each of 9 z bins).
- **Modified**: 2D Sobol over `(θ_param, z)`, snapped to discrete z grid.
- **Why**: linspace-per-z would just repeat θ values across z bins,
  giving PySR no joint information about how θ-effect varies with z.
  Sobol scatters in 2D so each (θ, z) draw is unique → PySR can learn
  z-evolution.
- **Code**: `_generate_1pvar_multiz_inline`.

### 3c. Undo `k·P/π` transform on 1pvar load
- **Original** (legacy HDF5 path): the student's 1pvar HDF5 stores `k·P_F/π`
  (the `sample_1P_predictions` path applies that transform, see
  `lyaemu.priya_explorer.py:193`).
- **Modified**: undo on load (`P_F = stored × π / k`).
- **Why**: massive unit mismatch — flux_norm at the prior fid_phys gave
  89% rel-err before this fix, 0.92% after.

---

## 4. Normalization architecture: Option B local-std + at-fid anchor

### 4a. Option B (per-param 1D-local std) for single-z
- **Original**: ONE multi-D global `(mean_k, std_k)` from a Sobol of LF
  emulator over the full 11D prior cube (the student's "fallback path").
- **Modified**: per-param `(mean_k, std_k)` computed from the LF 1pvar
  slice for that param.
- **Why**: weakly-coupled params (hub, bhfeedback, etc.) have flux_norm
  signal ≈ 0.025 in multi-D-global units (vs ~1 for the dominant
  param tau0). PySR's MSE ≈ signal² → can't distinguish "x0 contributes"
  from "x0 doesn't matter". Switching to per-param std gives every refit
  ~1σ signal.
- **Effect**: hub σ ratio went 3.26× → 1.32× (with priors); bhfeedback
  → 1.00×.
- **Code**: `compute_local_normalization`.

### 4b. Per-(z, k) at-fid anchor for multi-z
- **Original**: Option B's per-param 1D-local std, naively extended to
  multi-z = ONE (mean, std) per param across all (z, sims).
- **Modified (intermediate)**: per-(z, k) `(mean, std)` from per-z Sobol.
  Helped flux_norm amplitude balance (was 0.087 at z=2.6 vs 0.278 at
  z=4.2 with single-norm).
- **Modified (final)**: at-fid anchor — `mean_per_z_k = P_GP_LF(fid, k, z)`
  (NOT empirical mean across the Sobol sims). Train target:
  `(P_F − P_GP_LF(fid, k, z)) / std_per_z(k)`.
- **Why**: with empirical-mean anchor, **6 of 11 multi-z equations
  dropped x0** (the parameter dependence) → multi-z Fisher singular.
  At-fid anchor forces target = 0 at θ=fid_phys for every (z, k), so
  the eq MUST have x0 dependence to express any non-zero deviation.
- **Memory**: `memory/at_fid_anchor_for_multiz.md`.
- **Code**: `compute_local_normalization_multiz` requires `gp_lf, fid`.

### 4c. Local-anchored combine (replaces student's combine)
- **Original** (student `mf_dtau0_ap_ns_hf.py`):
  ```
  P_norm = Σ_i [eq_i(θ_i_norm, k_norm, 0.8) − eq_i(0.5_norm, k_norm, 0.8)]
         + (1/n) Σ_i eq_i(0.5_norm, k_norm, 0.8)
  P_F   = P_norm × std_k_global + mean_k_global
  ```
  Used `fid_norm=0.5` hardcoded approximation.
- **Modified** (`mode="local_anchored"`):
  ```
  P_F(θ, k, z) = P_GP_HF(fid, k, z)
              + Σ_i [r_i.predict(θ_i, k, z) − r_i.predict(fid_i_phys, k, z)]
  ```
  Each per-param contribution is in physical P_F units (via per-param
  norm round-trip).
- **Why**:
  - The student's `fid_norm=0.5` is wrong — `fid_phys` ≠ prior midpoint
    (e.g., `ns` fid_norm = 0.732, not 0.5). Using 0.5 produces a 15%
    mismatch at fid.
  - The local-anchored form is **exact at fid** by construction (every
    deviation cancels at θ=fid_phys for every i).
  - Each per-param contribution scales by ITS OWN std_per_z, so weak
    params don't drown.
- **Code**: `src/priya_forecast/refit_taylor.py::AdditiveTaylorModel(mode="local_anchored")`
  and `MultiZAdditiveTaylorModel`.

---

## 5. Fisher pipeline modifications

### 5a. Synthetic diagonal cov for forecast comparison
- **Original**: eBOSS DR14 covariance via `priya_forecast.data::load_eboss`.
- **Modified**: Synthetic `σ_k = 5% · P_F(fid, k)` diagonal on the kodiaq
  k-grid.
- **Why**: kodiaq k-grid (0.005-0.064) doesn't span the eBOSS k-grid
  (0.001-0.02); using synthetic cov lets us compare GP vs hybrid σ ratios
  cleanly during pipeline development. **For paper-final, switch to real
  KSData(conservative=True) covariance** (Karacayli et al. 2021,
  KODIAQ-SQUAD); see § 7.

### 5b. dtau0 fixed at fid (for single-z)
- **Original**: dtau0 floated in the Fisher.
- **Modified**: `--fix-params dtau0` removes it from the varying set
  (held at upstream `best_par[0] = -0.009`).
- **Why**: at single z, the (dtau0, tau0) mean-flux pair is degenerate
  by construction. Fixing dtau0 breaks the degeneracy. Production paper
  convention.
- **Note**: not needed for multi-z (each z provides independent leverage
  on the slope). Multi-z runs vary all 11.

### 5c. Production Gaussian priors
- **Original**: no priors.
- **Modified**: priors from `~/lya_emulator_full/lyaemu/likelihood.py`:
  - hub σ = 0.015 (cosmic-variance prior)
  - omegamh2 σ = 0.001 (Planck 2018, arXiv:1807.06209)
  - bhfeedback σ = 0.005 (centered at 0.05)
- **Effect**: hub/Ω/bh σ ratios all → ~1× (prior-dominated for both GP
  and hybrid). Removes 3 of the 4 outliers from the unprio'd run.
- **Code**: `priors_sigma={...}` arg to `fisher_matrix`.

### 5d. Multi-z Fisher aggregation
- **New**: `combine_fisher_phys_arrays(F_per_z, ..., priors_sigma=...)` and
  `compute_fisher_F_phys(...)` (returns `F_phys = Y^T Y` without inverting).
- **Why**: per-z F can be singular (e.g., at z=2.6 herei has zero
  effect). Need to sum F_phys across z first, add priors, then invert
  once. `fisher_matrix(...)` per z would fail on the singular per-z
  matrix.
- **Caveat**: assumes z-bins independent in covariance. For paper-final
  with KSData (cross-z block), refactor to a single stacked
  `MultiZGaussianLikelihood` instead.

---

## 6. Per-D resolution correction output

- **Original** (student `mf_*.py`): the resolution correction is
  implicit in the eq's x2 dependence; not extracted as a deliverable.
- **Modified**: `Δ_i(k) = eq_i(fid_phys, k, x2=0.8) − eq_i(fid_phys, k, x2=0.4)`
  per parameter, exported as JSON + a 11-panel grid figure.
  - **Initial bug**: I was evaluating at θ_norm=0.5 (legacy student
    hack); fixed to evaluate at fid_phys.
- **Why**: this is one of the four required forecast deliverables (per
  the user's spec). It tells you, parameter by parameter, how much the
  LF→HF lift contributes to P_F at fid.
- **Code**: `_resolution_correction_per_dim` and
  `_write_resolution_correction_summary` in
  `scripts/refit_all_11_params.py`.

---

## 6.5 Pareto-search-for-x0 (post-fit selection)

- **Original** (PySR default): `model_selection="best"` picks the
  Pareto-front entry with the lowest training loss.
- **Modified** (`refit_one_param.py`): scan the Pareto front for
  equations containing `x0` (the parameter input), and pick the
  lowest-loss x0-using entry. Fall back to global best only if no
  Pareto entry contains x0.
- **Why**: PySR's MSE loss is dominated by k-, z-, and resolution-
  variance for weakly-coupled params. The genetic algorithm often
  finds equations that fit the per-(k, z, res) target shape without
  using x0. The Pareto front USUALLY contains an x0-using equation at
  slightly higher complexity / loss; selecting it preserves x0
  dependence at the cost of ≤10% extra training MSE.
- **Effect**: rescues 6/11 multi-z params that drop x0 in the
  best-loss selection; no PySR retry needed.

## 6.6 Future improvement: dimension-balanced loss

**Idea (user-suggested, multi-task ML inspired)**: PySR's `elementwise_loss
(prediction, target, weight)` and `model.fit(X, y, weights=...)` let
us re-weight per-row loss contributions. For a target whose variance
is dominated by k/z/res and barely depends on θ (weak-coupling
params), reweighting rows by `1/var_kzr` boosts θ-driven residuals.

- Reference: Kendall et al. 2018, "Multi-Task Learning Using
  Uncertainty to Weigh Losses" — minimize Σ_t (1/σ_t²)·loss_t +
  log σ_t² where σ_t is per-task uncertainty.
- Even simpler version: per-(k, z) bin, compute the empirical θ-spread
  of the target. Use that spread as a per-row weight in PySR's loss.
- This is a Julia-side change to the `elementwise_loss` definition;
  more involved than the Pareto-search fix and not landed for this
  paper (Pareto-search alone is sufficient for our σ-ratio targets).



- **Real KODIAQ-SQUAD covariance** (`KSData(conservative=True)`):
  cross-(z, k) 182×182 matrix from Karacayli et al. 2021. Replace
  synthetic cov.
- **Kim per-z mean-flux prior** on tau0 (Kim σ ≈ 0.304 fractional → 0.331
  absolute at tau0_fid=1.090). Production paper applies this; we
  haven't yet (data already constrains tau0 at single-z; multi-z
  constrains it further).
- **Residual-PySR** on the IGM thermal sub-block {herei, heref, alphaq,
  hireionz} if multi-z + at-fid anchor still leaves them with σ ratios
  > 2× (Comment 4 from the user's original ask, code already exists in
  `refit_residual.py` / `run_residual_pysr.py`).
- **Reproducibility re-run**: switch to `deterministic=True,
  parallelism="serial"` for the final equations.
- **Equation table**: per-param paper-grade equation listing with
  Pareto complexity, training loss, and rel-err, formatted for inclusion
  as a paper table.

---

## Pipeline summary (for the methods section)

For each of the 11 PRIYA cosmological + IGM parameters, we train a 1D
PySR equation with 4 input features `(θ_norm, k_norm, resolution,
z_norm)` and per-(z, k) at-fid-anchored target
`(P_F − P_GP_LF(fid, k, z)) / std_LF(z, k)`. Training data is generated
inline by 2D Sobol-scattering 225 `(θ, z)` points across each
parameter's prior and z ∈ [2.6, 4.2], then evaluating the LF
(single-fidelity) and HF (multi-fidelity) PRIYA emulators on a 32-bin
linspace k-grid in [0.005, 0.064] s/km. PySR settings: `niter=50`,
`maxsize=20`, binary `+ − * / ^` with `^` constrained to scalar
exponents, unary `exp/log/square/sqrt/inv` (no trig — see § 2a),
multithreaded with `procs=4`, random_state=42.

The 11 per-param equations are combined via the local-anchored Taylor:

    P_F_combine(θ, k, z) = P_GP_HF(fid, k, z)
                         + Σ_i [r_i.predict(θ_i, k, z) − r_i.predict(fid_i_phys, k, z)]

This is exact at fid by construction. Each `r_i.predict()` is in
physical P_F units via its per-z normalization round-trip.

11D Fisher forecasts use a synthetic 5%-of-P_F diagonal covariance for
development; final paper numbers use the KODIAQ-SQUAD covariance from
Karacayli et al. 2021 (`KSData(conservative=True)`). Production
Gaussian priors on hub (σ=0.015), Ω_m h² (σ=0.001), and BH feedback
(σ=0.005) are applied as in `lya_emu_xq100/lyaemu/likelihood.py`.
