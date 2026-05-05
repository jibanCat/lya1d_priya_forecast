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

## Related work (for the references section)

Two recent papers explicitly relevant to our preprocessing and
emulator-design choices.

### Cabayol-Garcia et al. 2023 — arXiv:2305.19064

> *"A neural network emulator for the Lyman-α forest 1D flux power spectrum"*

- **Key parametrization**: emulate P1D as a function of the linear-power
  amplitude/slope `(Δ²_p, n_p)` at a pivot scale + IGM nuisance params
  (F̄, σ_T, γ, k_F), **not** the raw cosmological parameters. Cosmology
  dependence is compressed into 2 numbers.
- **Target preprocessing**: fit **`log₁₀ P1D` as a polynomial in
  `log₁₀ k_∥`** (4th–6th order). The NN outputs polynomial coefficients,
  not per-k flux values. Loss is in `log₁₀ P1D` space.
- **Input normalization**: inputs min-max scaled then shifted to
  `[-0.5, 0.5]`; output P1D divided by the median of training P1D
  before log.
- **Multi-z**: a *single* network across redshifts; z enters via
  `(Δ²_p(z), n_p(z))`, no per-z heads.
- **Polynomial vs. PCA**: their Appendix C tests both — polynomial
  coefficients in log-log space win on accuracy and robustness.

**Implication for our pipeline**: our flux_norm target
`(P_F − P_GP_LF(fid, k, z)) / std_per_z(k)` is in linear P_F space.
Cabayol-Garcia's `log₁₀ P_F` vs `log₁₀ k` representation might give
PySR cleaner equations — `(k/k_pivot)^(n_s − 1)` becomes
`(n_s − 1) · log(k/k_pivot)` in log-log, a polynomial PySR finds
trivially. **Worth testing as a future experiment**; for this paper
we stick with linear-P_F to keep continuity with the student's
pipeline.

### Yang, Bird, Ho, Qezlou 2025 — arXiv:2507.07184 (GokuNEmu)

> *"Design and optimization of neural networks for multifidelity cosmological emulation"*

- **Architecture**: FCNN + SiLU + AdamW + Bayesian HPO. Replaces
  GP-based MF-Box because GP cost scales cubically and degrades in
  10D parameter space.
- **The critique of "2-step" multi-fidelity NNs**: the original
  design concatenates LF output with input (dim = `d_in + d_out`) for
  the LF→HF correction; **this blows up when `d_out ≫ d_in`**. They
  propose a **modified 2-step** where the second NN learns the
  ratio `r = y_H / y_L` as a function of x only (dim = `d_in`), and
  HF = `y_L · r`.
- **Per-z PCA**: output compression by PCA — but **local (per-z) PCA
  outperforms global PCA**, because nonlinear redshift evolution gets
  absorbed into z-specific eigenbases. One NN; per-z structure lives in
  the PCA basis.
- **Multiplicative ratio is in *linear* (not log) space** for matter
  power spectrum (their setting).

**Implication for our pipeline**: the user's explicit preference is
**not** a two-separate-emulator design. Our current architecture
satisfies that — we use **one** MF emulator (`GPModel(fidelity="lf")`
and `GPModel(fidelity="hf")` are different views of the same upstream
`lyaemu.GPWrap`), and **one** PySR equation per parameter (or per
sub-block) that takes the resolution as a feature `r ∈ {0.4, 0.8}`.
The LF→HF lift is encoded in the equation's `r` dependence — single
unified flow, no second NN/equation.

The multiplicative ratio idea (`HF = LF · ratio(x)`) is interesting:
our `resolution_correction.md` already exports `R_i(k) = P_F^HF / P_F^LF`,
which is *implicitly* the same ratio. We could in principle re-derive
the per-param equations to fit `log(P_F^HF / P_F^LF)` directly rather
than the additive deviation; for this paper we report the additive
form (option B local-anchored) and the multiplicative ratio side-by-side.

The per-z PCA finding *parallels* our per-z normalization
(`compute_local_normalization_multiz`): we already have per-z `(mean,
std)` arrays absorbing the smooth z-evolution, leaving the PySR fit
to focus on residual shape. We're doing it without the PCA basis,
but the spirit is identical.

---

## Design decisions (with reasoning)

The user has explicit preferences on these; they're locked in for this
paper.

### D1. `dtau0` fixed at **0**, NOT at the upstream `best_par` value (-0.009)

`-0.009` is the upstream `best_par[0]` — an earlier MCMC-fit slope
of the mean-flux evolution against eBOSS DR14 P1D data. **0** is the
slope from Kim et al.'s observational mean-flux measurement
(an EXTERNAL anchor).

**Why 0**: our forecast uses eBOSS-equivalent P1D statistics (KSData
covariance from KODIAQ-SQUAD). Using `dtau0 = -0.009` would re-use
the same data statistics that informed it (circular — would
double-count the eBOSS-DR14 mean-flux information). Kim's
observational slope is from independent quasar absorption observations
and avoids this circularity.

Implementation: scripts default to `--fix-dtau0-to-zero` (use
`--no-fix-dtau0-to-zero` to override).

### D2. Multi-D PySR cross-coupled subset = `{ns, Ap, herei, heref, alphaq, hireionz}`

Mixes cosmology (`ns`, `Ap`) and IGM thermal (`herei`, `heref`,
`alphaq`, `hireionz`). Why mix?

- The **headline coupling-matrix finding** (Phase 5 of the original
  spec) shows `herei × alphaq` is the only positive cross-coupling in
  the 11-param prior cube — within the IGM thermal block.
- Cosmology params (`ns`, `Ap`) interact weakly with IGM thermal in
  P_F shape, but the per-1D + additive-Taylor combine treats them
  as fully separable. If there's any subtle cosmology × thermal
  cross-coupling (e.g., the P_F amplitude shifts with σ₈ are partly
  degenerate with mean-flux scaling), the multi-D fit captures it.
- Outside the subset (`tau0`, `hub`, `omegamh2`, `bhfeedback`) we use
  GP-slice fallback. They're either prior-dominated (hub/Ω/bh) or
  weakly coupled to anything else (tau0 is approximately a global
  multiplicative scale). So separating them as 1D-via-GP is safe.

### D3. Custom loss: functional ANOVA main-effect penalty

PySR's standard MSE is **dimension-blind at the batch level**: it can
fit a target via patterns in (k, z, resolution) without using x₀
(theta), as long as the per-θ residual variance is small. For weakly
-coupled params (`omegamh2`, `bhfeedback`, `hireionz`), this exactly
happens — the genetic search drops x₀ entirely. We saw this ≥ 6/11
times in the multi-z runs.

**Functional ANOVA decomposition**. Any function `f(x₀, ..., xₙ)` can
be written as a sum of orthogonal effects:

    f(x) = f₀ + Σᵢ fᵢ(xᵢ) + Σᵢ<ⱼ fᵢⱼ(xᵢ, xⱼ) + ...

For the residual `r(x) = pred − target`:

- `r₀ = mean(r)` over the whole batch.
- `rᵢ(xᵢ) = E[r(X) | X_i=x_i] − r₀` is the **main effect** in dim i:
  how much of the residual is *systematically* driven by xᵢ alone
  (marginalized over the other dimensions).
- If the equation drops xᵢ, `rᵢ` becomes large because r varies
  systematically with that input.

We estimate the main-effect L² norm from a Sobol batch by binning xᵢ
into `n_bins` quantile bins and summing:

    ‖rᵢ‖² ≈ Σ_b (P(b) · (mean(r | X_i ∈ bin b) − r₀)²)

The dim-balanced loss:

    L = MSE  +  α · Σ_d ‖r_d‖²

where d ranges over all input features. α controls the weight of the
per-dimension main-effect penalty (default α=5).

**Why ANOVA over correlation² (the simpler proxy I had earlier)**: a
correlation² catches only **linear** residual-vs-feature dependence.
ANOVA main effects catch **any** dependence — quadratic, sigmoidal,
piecewise. For weakly-coupled params, the residual may depend on x₀
nonlinearly (e.g. via an interaction with z that, when marginalized,
shows a non-monotone main effect). Correlation² misses these.

Implementation in `src/priya_forecast/dim_balanced_loss.py` exposes
both `dim_balanced_loss_corr` (legacy correlation² ref) and
`dim_balanced_loss_anova` (recommended). `JULIA_LOSS_FUNCTION` uses
the ANOVA form by default. Unit tests cover both. ~half-day of work
implemented in this session.

### D4. Multi-D Pareto pick: **most-θ-used** + sanity guard

The script picks the lowest-loss Pareto entry that
  (a) uses the maximum number of subset θ-features, and
  (b) doesn't contain a literal constant with `|c| > 100`
      (rejects the `(x0 - 3.4e11)/(x3 - 0.23)` failure mode where the
      eq technically uses x0 but is effectively constant in θ via a
      huge offset).

Interpretability matters more than minimum loss for the paper —
every-θ-used is the criterion.

### D5. Resolution correction: **HF/LF multiplicative ratio** (headline)

Paper figure: ratio `R_i(k, z) = P_F^HF / P_F^LF`. Per-panel y-axis
range tuned to each parameter (NOT sharey), and split into two grids:

- **Cosmology + mean-flux grid**: `dtau0, tau0, ns, Ap, hub, omegamh2`.
- **IGM thermal / astro grid**: `herei, heref, alphaq, hireionz, bhfeedback`.

Outputs at `resolution_correction_grid_{cosmo,astro}.{png,pdf}`.

Additive form (`Δ = HF − LF`) also exported in
`resolution_correction.json` for reference.

### D5.5. Multi-D PySR over 6 cross-coupled features — **abandoned** (post-mortem)

We attempted a single PySR equation over the cross-coupled subset
`{ns, Ap, herei, heref, alphaq, hireionz}` plus `(k, resolution, z)`,
trained on the at-fid-anchored normalized flux residual (D2). Two
configurations were tried, both abandoned:

| Run | procs | niter | complexity | flux_norm loss | Outcome |
|---|---|---|---|---|---|
| login-node smoke | 4 | 50 | 24 | 0.554 | `Ap`/`herei` σ NaN; `heref` σ ratio = 5×10⁶× |
| SLURM, stencil-safe filter | 15 | 100 | 25 | 0.585 | `ns`/`herei`/`heref` σ NaN; `Ap`, `alphaq`, `hireionz` ratios 10⁶–10¹⁹× |

**Result**: both Fisher matrices have eigenvalue spreads of 26 orders
of magnitude — one near-zero positive, one large negative (numerical
artifact of inverting a near-rank-deficient matrix). Diagonal
entries of `cov_hybrid` for the rank-deficient block are negative, so
σ = √diag is NaN.

**Diagnosis** — the discovered equations look like

    eq ≈ exp(θ_herei + θ_heref / (θ_Ap · c)) + θ_ns · k + …  (login)
    eq ≈ exp((…) + (k · −5.19) + …) + (θ_ns · 2.55 − r) + k  (SLURM)

In both, the IGM thermal triple `{Ap, herei, heref}` enters through a
single `exp(...)` group whose argument is **a single linear (or affine)
combination** of those three features. To first order in θ,

    ∂(eq) / ∂θ_Ap  ∝ ∂(eq) / ∂θ_herei  ∝ ∂(eq) / ∂θ_heref

so the three gradient vectors w.r.t. `flux_norm` are collinear. The
Fisher block over those three is **rank 1**, not rank 3. The SLURM run
with niter=100 collapsed *more* dimensions onto the shared exp/affine
group — its larger Pareto budget was spent on a *more* compact form,
which is what PySR rewards.

**This is not a transient bug — it's structural.** PySR's Pareto front
rewards *low loss per complexity*; for high-D inputs, the cheapest way
to lower loss is to fold features into a shared `exp(·)`, `(·)^p`, or
sigmoid group. Such groups have rank-1 first-order behavior in θ. So
"single PySR eq over k cross-coupled features" tends to produce
**rank-deficient** Fisher blocks — and the more nominally "expressive"
the eq, the worse the rank.

**What works instead** (per-1D + additive Taylor, our Option C
headline): each θᵢ has its own 1D PySR eq, so each gradient direction
is structurally distinct from the others by construction. Rank is full
by design, regardless of what compact functional form each 1D eq picks.
This is the operative reason Option C delivers a well-conditioned
multi-z Fisher and the multi-D run does not.

**Implication for the paper**: report the multi-D failure as a negative
result in the methods discussion. The headline forecast is per-1D +
additive-Taylor combine. Cross-coupling correction is left as future
work, with the next-step proposal being a **per-pair (or small-block)
PySR cross-coupling residual on top of Option C** — bounded complexity
per pair, structurally rank-additive, much cheaper to fit.

Outputs from the abandoned runs are kept under
`results/refit_multid_z2.6-4.2{,_login}/` for reproducibility; the
forecast scorecard does **not** use them.

### D6. Paper figure budget: 6 main figures + 2 appendix grids

**Main figures**:

| # | Figure | Purpose |
|---|---|---|
| 1 | Resolution correction grid (cosmo + astro, side-by-side) | per-D HF/LF ratio at θ=fid, split into cosmology+mean-flux and IGM thermal blocks. Output: `resolution_correction_grid_{cosmo,astro}.pdf` |
| 2 | Resolution correction param-variation grid (cosmo + astro) | R(k; θ) at 5 quantiles (q=0.1, 0.3, 0.5, 0.7, 0.9) of each parameter's prior; shows where the resolution correction is ≈ θ-flat vs θ-dependent. Empirical observation: only `tau0` shows clear θ-dependence; cosmology/astro params overlap. Output: `resolution_correction_param_variation_{cosmo,astro}.pdf` |
| 3 | Hold-out validation grid (cosmo + astro) | mean & max `\|pred − truth\| / \|truth\|` vs k for both LF and HF on a fresh n=50 Sobol sweep (seed ≠ training). Demonstrates pipeline accuracy on unseen θ. 1% reference dashed line. Output: `holdout_validation_{cosmo,astro}.pdf` |
| 4 | Multi-z Fisher corner plot | 10D Fisher overlay: GP (black) vs PySR hybrid (red). Output: `corner.pdf` |
| 5 | σ-ratio bar chart | per-param hybrid σ / GP σ on the multi-z + KSData scorecard, color-coded by treatment (multi-D PySR / per-1D / GP-slice / fixed). Shows where the pipeline matches GP and where it doesn't |
| 6 | Ablation table | single-z vs multi-z; per-1D + Taylor vs multi-D + cross-coupled; synthetic vs KSData covariance; without vs with residual-PySR (if needed). Numbers from `scorecard.md` runs |

**Appendix material**:
- Per-param equation table (LaTeX, generated from `per_param_summary.md`)
  with full PySR equations + complexity + flux_norm loss + LF/HF rel-err.
- Per-D resolution correction symbolic expressions (from
  `resolution_correction_equations.md`).

---

## PySR hyperparameter budget — measured cost and rationale

For paper Methods + reproducibility. Wall time scaling matters because
it determines what equation-search depth is feasible per parameter.
All numbers are on Greatlakes login node (~16 cores, mamba py3.11
base, multithreading via `parallelism="multithreading"` + `procs`),
unless noted.

### Per-1D refit (single param's PySR, multi-z 4-input fit)

| niter | maxsize | procs | n_total Sobol | rows in fit | wall/param | rationale |
|---|---|---|---|---|---|---|
| 20 | 20 | 1 (serial, deterministic) | 50 | 6,400 | ~100 s | reproduces student `pysr_mf_given.py` exactly; baseline |
| 20 | 20 | 4 | 225 | 14,400 | ~30 s | multithreaded — no science change, ~3-4× faster |
| 50 | 20 | 4 | 225 | 14,400 | ~50–80 s | **default** for production. Closes "x0 dropped" failures: 8 of 11 weak-coupling params recover x0 dependence |
| 100 | 20 | 4 | 225 | 14,400 | ~3-5 min | retry-with-different-seed for the residual 3 weak params (omegamh2, hireionz, bhfeedback). Even at niter=100 these can drop x0 unless dim-balanced ANOVA loss is enabled |

**Why niter=50 is the default**: at niter=20 we observed 6/11 multi-z
equations dropping `x0` (the parameter dependence). The genetic
algorithm needed more time to find x0-using Pareto entries. niter=50
recovered 8/11; niter=100 didn't significantly improve over 50 except
for the genuinely-weakly-coupled params (which the **dim-balanced
ANOVA loss** is the architectural fix for, see § D3).

### Multi-D cross-coupled fit (single 9-input PySR over the subset)

Subset = {ns, Ap, herei, heref, alphaq, hireionz}; inputs are 6 θs +
k + resolution + z. Larger search space than per-1D.

| niter | maxsize | procs | n_total | rows | wall (login) | wall (SLURM 16-cpu) | rationale |
|---|---|---|---|---|---|---|---|
| 50 | 20 | 4 | 128 | 8,192 | ~10 min | ~3–4 min | **smoke test** — fast iteration; equation may be pathological (insufficient genetic search) |
| 100 | 25 | 4 | 256 | 16,384 | ~25-35 min | ~10 min | login-node production; multi-D needs more iter than per-1D because 9D search is harder |
| 100 | 25 | 15 | 256 | 16,384 | n/a | ~6–8 min | **SLURM production** — `procs=15` on a 16-CPU node ≈ 2× login-node speedup |
| 200 | 30 | 15 | 512 | 32,768 | n/a | ~20-30 min | reserved if niter=100 doesn't converge to a stable equation |

**Why budget went up vs per-1D**: per-1D PySR has 3 inputs (θ, k, res),
multi-D has 9 (6 θs + k + res + z). The genetic search has to explore
~3× more "feature combinations" per equation. Our observed per-fit
loss at the same niter is 5-10× higher in multi-D vs per-1D, and the
Pareto front is sparser (fewer x0..x5-using entries). Doubling niter
(20 → 50) helped per-1D; we set multi-D default to niter=100 for the
same fractional improvement.

**Sobol n_total**: 256 (= 25 × ~10 random θ × 9 z-bins after
snapping) gives ~2× the points of a full grid scan over the 6D θ
prior. Doubling to 512 doesn't materially improve fit accuracy in our
tests; 256 is the sweet spot for ~10 min wall.

### Cost ledger for the full forecast pipeline

This is what one **complete forecast iteration** costs (data gen +
all per-1D refits + multi-D fit + Fisher + paper deliverables):

| Step | Cost | Notes |
|---|---|---|
| Phase 1: precompute_payloads.py (11 × 1pvar + per-z norm) | ~3.5 min | one HF + LF emulator load each (60 s + 60 s); 11 × ~20 s payload gen |
| Phase 2: 11 per-1D refits (SLURM array, parallel) | ~3 min wall | 11 × ~50 s, run as `--array=0-10` |
| Phase 3a: Multi-z aggregate Fisher (synthetic 5%-of-P_F cov) | ~1 min | per-z Fisher (9 z-bins) + summation + corner |
| Phase 3b: Multi-z aggregate Fisher (KSData covariance) | ~1 min | single Fisher call with KSData full cov; same scale |
| Phase 4: Multi-D fit (cross-coupled subset) | ~6–8 min (SLURM 16 cpu) | one PySR call over 9-input space |
| Phase 5: Multi-D Fisher + scorecard | ~30 s | uses cached multi-D refit + GP-slice for outside-subset |
| **Total wall (one forecast iteration)** | **~15 min** with SLURM parallelism | vs ~1 hour serial-on-login |

For paper-final, re-run with `deterministic=True, parallelism="serial"`
to lock in bit-reproducible equations: ~4-5× slower → ~1 hour total.
Worth doing once at the end.

---

## D7. Phase 2 (planned): per-pair PySR cross-coupling on top of Phase 1

Phase 1 (the headline result of this PR) is per-1D PySR + additive-Taylor
combine, structurally rank-correct but cannot capture cross-coupling
between parameters. Multi-D PySR over the joint subset failed (§ D5.5)
because PySR's Pareto rewards rank-1 shared-`exp(·)` groups.

**Phase 2 design** (full plan in `docs/PAIR_FIT_PLAN.md`): keep Phase 1
unchanged; add one small PySR equation per parameter pair, fit on the
*residual* after subtracting Phase 1's prediction. Each pair adds one new
gradient direction by construction (a "pure 2-way ANOVA interaction"
term that vanishes whenever either θᵢ=fidᵢ or θⱼ=fidⱼ). Fisher rank stays
full; if a pair's signal is weak, its `Ĝ_ij` fits to ≈ 0 and Fisher is
unchanged — graceful degradation.

**Pair selection from synthetic-data MCMC** (real-data MCMC compresses
correlations because posteriors hit prior boundaries). Cached at
`results/simdat_ind15_truth.npz`; top |ρ_simdata| ≥ 0.2:

| pair | ρ_simdata | tier |
|---|---|---|
| **tau0 × ns** | **−0.92** | must-have |
| Ap × alphaq | +0.68 | should-have |
| tau0 × Ap | −0.66 | should-have |
| ns × Ap | +0.55 | maybe |
| tau0 × alphaq | −0.55 | maybe |
| ns × alphaq | +0.43 | maybe |
| heref × alphaq | +0.29 | maybe |
| **herei × alphaq** | **−0.22** | must-have (Phase 5 IGM coupling headline) |

`dtau0` pairs skipped (dtau0 fixed at 0, § D1). Phase 2 starts with the
two must-have pairs; escalates to should-have only if the off-fid corner
remains discrepant from σ_MCMC_simdat.

**Validation strategy** is GP-Fisher vs PySR-Fisher head-to-head at the
synthetic-target θ_target_simdat (Data Index 15 of the closure suite),
with σ_MCMC_simdat as the truth overlay. We do **not** validate against
the real-data MCMC chain because its posteriors hit prior boundaries and
the resulting σ are driven by the prior not the likelihood.

**Phase 1 closure to σ_MCMC_simdat** (motivates pair selection — these
will be re-pulled after PR #1's BLOCKER #1 fix; current numbers are from
`results/refit_optionC_z2.6-4.2_ksdata/scorecard.md`):

| param | σ_PySR / σ_MCMC | flag |
|---|---|---|
| ns | 1.0× | ✓ closed |
| tau0, hub, bhfeedback | 1.4–1.5× | OK |
| herei | 3.4× | needs pair |
| **heref** | **14×** | biggest miss; possibly needs per-1D refit at niter=200 first |
| alphaq, omegamh2 | 0.6× | overconfident — possibly fitting GP interpolation noise |
| hireionz | broken eq → 1e12× | BLOCKER #1 fix routes to GP-slice |

The `heref × 14×` and overconfidence on `alphaq, omegamh2` are the
strongest motivations for adding pair coupling. `tau0 × ns` is also
needed: Phase 1 cannot capture the dominant ρ = −0.92 cosmology
degeneracy regardless of how good the per-1D fits are.

**Cost**: ~1 h SLURM per pair (5-D PySR fit on Sobol residuals,
embarrassingly parallel via SLURM array). 2 must-have pairs ≈ 1 h wall;
4 must+should ≈ 1 h wall. Full Phase 2 = ~2 days end-to-end including
validation plots.

**Fallback (Option β)**: if the residual fits emulator noise (signal too
weak), use PySR's `TemplateExpressionSpec`
(https://github.com/MilesCranmer/PySR/discussions/787) — fixes the outer
form (additive per-1D + per-pair) and lets PySR jointly fit all
sub-expressions. ~12 h SLURM, untested API, used only if Option α fails.

**Future-future work (not Phase 2)**: subclass
`lyaemu.likelihood.CobayaLikelihoodClass` to swap the GP for the PySR
hybrid → full MCMC with the symbolic emulator. Tests whether PySR is
faithful for *nonlinear* sampling, not just Cramer-Rao at fid. Reserved
for follow-up paper or appendix.

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
