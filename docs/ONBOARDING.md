# priya-forecast onboarding (math-first, up to Phase 1.5)

This guide is for who wants to **understand
the math** behind the forecast, not the pipeline plumbing. After reading,
you should be able to reproduce the Phase 1.5 σ-table given a
trained PySR equation set.

The reading map at the bottom points you at the **theory-bearing source
files** (the ones whose top-of-file docstrings spell out the formulas).
Skip the scripts in `scripts/` for now — they're orchestration around the
math in `src/priya_forecast/`.

---

## 1. What the forecast computes

We have a PRIYA Lyman-α flux power spectrum `P_F(θ; k, z)` parameterized
by 11 cosmology + IGM-thermal parameters θ. We have a measurement
covariance C (KSData, Karaçaylı+ 2021) on a fixed `(k, z)` grid. The
question is: **how tightly do those data constrain each θ_i?**

For a Gaussian likelihood with parameter-independent covariance,
the **Fisher information matrix** is

```
              ⎛ ∂m ⎞ᵀ      ⎛ ∂m ⎞
  F_{ij}  =   ⎜ ── ⎟   C⁻¹ ⎜ ── ⎟
              ⎝ ∂θᵢ⎠       ⎝ ∂θⱼ⎠
```

where `m(θ)` is the model prediction `P_F(θ; k, z)` **stacked over the
full `(k, z)` grid into one long vector**, and `C` is the KSData
covariance over that same stacked layout. The **marginalized 1σ error**
on θ_i is `σ_i = sqrt((F⁻¹)_{ii})`.

> Quick mental model: `F` is the local curvature of `−log L` at `θ=fid`.
> Diagonal entries `F_{ii}` measure how sharply the log-likelihood
> bends in the θ_i direction (large = data is informative about θ_i);
> off-diagonals encode parameter degeneracies. Inverting `F` and taking
> the i-th diagonal gives the **marginal** 1σ — i.e., after profiling
> out the other 10 parameters' uncertainty, including their
> correlations with θ_i. This is the Cramér-Rao / Gaussian-posterior
> approximation; it equals the MCMC marginal width whenever the
> posterior is locally Gaussian at fid.

> Important: do not split the sum into a per-z block sum. The KSData
> covariance is **not block-diagonal in redshift** (`ksdata_likelihood.py:17`
> calls this out explicitly), so cross-z entries of `C⁻¹` matter. Use one
> full stacked dot product, not `Σ_z (∂m_z)ᵀ C_z⁻¹ (∂m_z)`.

Both Fisher and the KSData likelihood are computed in **physical
`P_F(k, z)` space**, the same space the covariance lives in. The PySR
training target (§ 3b) is normalized flux, but predictions are
de-normalized back to physical `P_F` before any derivative is taken —
the Fisher derivative must always be in the data's units, not in the
surrogate's training units.

The whole project is "what should we use for `m(θ)`?":
- **GP emulator** = upstream PRIYA emulator (the ground truth, expensive,
  opaque).
- **Phase 1.5 hybrid** = per-1D PySR equations stitched together by an
  **"additive Taylor combine"**, with a **GP-slice fallback** for
  parameters whose PySR fit fails a quality gate.

> The codebase calls this combine "additive Taylor" (the class is
> literally `MultiZAdditiveTaylorModel`) — but mathematically it is
> **not** a literal first-order Taylor expansion. Each per-axis
> equation `eq_i(θ_i, ...)` is itself a nonlinear function of θ_i;
> what's truncated is *interaction order between parameters*, not
> Taylor order in any one parameter. The correct technical name is
> **additive main-effect model** (or "first-order functional ANOVA"):
> we model `P_F(θ)` as the sum of per-parameter univariate responses,
> dropping all pair, triple, ... interaction terms. We keep the
> historical "Taylor" name for code-grep continuity but explain the
> math in § 2.

Phase 1.5's job is to make σ_PySR ≈ σ_GP, parameter by parameter.

> Theory file: `src/priya_forecast/fisher.py`. Read its top docstring for
> the 5-point-stencil derivative formula and the adaptive step-halving
> rule.

---

## 2. The forward model: additive Taylor combine of per-1D PySR

The Phase 1.5 prediction at any θ is

```
                                   ┌                              ┐
  P̂(θ; k, z)  =  P_GP(fid; k, z) + Σ ⎢ eq_i(θ_i, k, z) − eq_i(fid_i, k, z) ⎥
                                  i └                              ┘
```

(at HF resolution `r = 0.8`; the LF case is handled symmetrically). Each
`eq_i` is a single PySR equation in **four normalized inputs**:

```
  eq_i ( x0 = θ_i_norm,    x1 = k_norm,    x2 = resolution,    x3 = z_norm )
```

with all four mapped to `[0, 1]`. **Notation note**: every `eq_i(θ_i, k, z)`
in this and the following formulas means the **de-normalized** equation
output (round-trip from § 3b applied), so the sum is in physical
`P_F(k, z)` units — the same space as `P_GP` and the KSData covariance.
PySR itself outputs `flux_norm`; we always de-normalize before
combining.

The combine has three properties worth remembering:

1. **Anchored at fid.** Every bracket is `eq_i(fid_i) − eq_i(fid_i) = 0`,
   so `P̂(fid) ≡ P_GP(fid)`. The forecast is exactly the GP at the
   fiducial point by construction.
2. **Axis-wise exact w.r.t. the chosen 1D surrogate.** If only `θ_i` is
   perturbed, every `j ≠ i` term vanishes and the prediction reduces to
   `P_GP(fid) + eq_i(θ_i) − eq_i(fid_i)`. Note this is "exact" with
   respect to the per-1D PySR equation we picked for θ_i — *not* with
   respect to the GP. Along-axis accuracy relative to the GP is set
   by the quality of that 1D fit (the 5%-rel-err quality gate, § 4) or
   is exactly the GP if that param is GP-sliced.
3. **Approximate off-axis (cross terms dropped).** The combine is **not**
   a literal first-order Taylor expansion — each `eq_i(θ_i)` is a
   nonlinear function of θ_i. What is dropped is **cross-parameter
   interaction**: pair, triple, and higher functional-ANOVA terms. The
   correct framing is "additive main-effect / first-order functional
   ANOVA model". Missing cross terms can make Fisher constraints either
   too tight or too loose depending on the geometry of the omitted
   interactions; the sign is not a priori. (Phase 2 adds pair
   cross-terms; Phase 1.5 does not.)

> Theory file: `src/priya_forecast/refit_taylor.py`. The
> `MultiZAdditiveTaylorModel.predict` method (line 206) is the combine
> in code; the top-of-file docstring is the formal recipe.

### The "GP-slice fallback"

If a parameter's PySR fit fails the quality gate (training-set mean
LF or HF rel-err ≥ 5%, or the equation drops `x0` and contributes no
gradient — see § 4 below for the full criteria), we route that
parameter through a **GP slice** instead of a PySR equation:

```
  P̂_phase1.5(θ; k, z)  =  P_GP(fid)
       +  Σ          [ eq_i(θ_i)       − eq_i(fid_i)    ]    ← PySR-routed
         i ∈ pysr
       +  Σ          [ P_GP(θ_i, others=fid) − P_GP(fid) ]    ← GP-sliced
         i ∈ slice
```

A GP-sliced parameter uses the GP response along its own axis, so the
**data-space derivative vector** `∂P̂/∂θ_i` matches the GP's by
construction (at the same stencil step and covariance). Therefore the
**diagonal Fisher contribution** also matches:

```
  F_{ii}  =  (∂P̂/∂θ_i)ᵀ  C⁻¹  (∂P̂/∂θ_i)    ≡   F_{ii}^{GP}
```

This is the "fixed-others" or unmarginalized 1σ. The **marginalized
error** `σ_i = sqrt((F⁻¹)_{ii})`, however, depends on the full inverse,
which mixes in every column `∂P̂/∂θ_j` for `j ≠ i` — including the
PySR-routed ones. So `σ_i` can drift slightly from the all-GP value
through cross-correlations. This is why the σ-table in § 6 shows
GP-sliced params at e.g. 0.99× / 1.01× / 1.02× rather than literal
1.00×.

The cost of GP-slicing is interpretability — a sliced parameter has no
symbolic equation. In Phase 1.5 the typical slice list is
`{omegamh2, heref, hireionz, bhfeedback}` — parameters whose PySR fit
either drops `x0` (eq doesn't depend on θ_i, so its gradient is zero)
or fails the 5%-rel-err quality gate. Phase 2 shrinks this list by
extending the smart operator/loss policy to all 11 params (`heref`
and `bhfeedback` get real PySR equations in Phase 2).

---

## 3. Per-1D PySR fit (the key training step)

This is the only PySR call in the whole forecast. For each of the 11
parameters, repeat:

### 3a. Build the training set

1. Sample over the product `(θ_i, z)` **2D Sobol-style**, with all other
   10 parameters held at fid. (Both θ_i and z must vary because the eq
   carries a `z_norm` feature; PySR needs joint coverage to discover
   z-dependence.)
2. Query the GP at every `(θ_i, k_grid, z, resolution=0.4)` (LF) and
   `(θ_i, k_grid, z, resolution=0.8)` (HF). The LF/HF distinction lets
   PySR *learn* the resolution correction inside the equation (via
   the `x2 = resolution` feature) instead of being told it.
3. Stack: `X = [(θ_i_norm, k_norm, 0.4, z_norm)_LF;
                (θ_i_norm, k_norm, 0.8, z_norm)_HF]`.

### 3b. Normalize the flux per `(z, k)` — the "at-fid anchor"

Within each (z, k) bin, normalize the training flux to

```
  flux_norm  =  ( P_F − mean_{z,k} ) / std_{z,k}
```

with **two distinct sources** for mean and std (`refit_1d_pysr.py:707-714`):

- `mean_{z,k}` = `P_GP^LF(fid; k, z)` — the **LF** GP prediction at the
  fiducial parameter vector. This is the **"at-fid anchor"**. Two
  consequences, depending on which resolution row you look at:
  - For an **LF training row** at `θ = fid` (resolution `x2 = 0.4`),
    `P_F = mean_{z,k}` exactly, so the normalized target is zero.
    Any nonzero target value at `θ ≠ fid` therefore *must* be
    explained by the eq's dependence on `x0 = θ_i_norm`.
  - For an **HF training row** at `θ = fid` (resolution `x2 = 0.8`),
    the target is `P_F^HF(fid) − P_F^LF(fid)`, which is generally
    nonzero. PySR learns this LF→HF gap through the `x2` feature —
    that's how the resolution correction enters the equation.

  So the LF anchor strips the gross k/z flux shape out of the symbolic
  target, while leaving the LF→HF correction visible. Without the
  anchor, PySR's genetic search latches onto k/z/resolution-only
  patterns and drops `x0` (observed empirically: 6/11 parameters lost
  x0 dependence under naive empirical-Sobol-mean anchoring).
- `std_{z,k}` = empirical std across the Sobol training sample at that z.
  Just a per-(z,k) scale so the target is O(1).

At inference time we round-trip via `flux_phys = flux_norm · std_{z,k} +
mean_{z,k}`. Both `mean_{z,k}` and `std_{z,k}` are arrays of shape
`(n_z, n_k)`, NOT scalars. This normalization choice is non-obvious and
consequential — see `docs/PAPER_NOTES.md § 4` for the full design log.

> Subtle point worth internalizing: the at-fid anchor `mean_{z,k}` is
> **purely a training-time device**. In the additive combine of § 2,
> `eq_i(θ_i) − eq_i(fid_i)` causes the per-(z,k) mean to **cancel
> exactly** (it's the same constant on both sides), so the prediction
> `P̂(θ)` is independent of how `mean_{z,k}` was chosen. The anchor's
> only purpose is to push PySR's genetic search toward x0-using
> equations during training.

### 3c. PySR config — Phase 1.5's "smart kwargs" (option B)

Two policies running together, both essential:

**(i) Operator policy "option B"** — production drops oscillatory
operators and explicit unary inversion / sqrt operators that wreck
Fisher conditioning:

```python
binary_operators  = ["+", "-", "*", "/", "^"]
unary_operators   = ["exp", "log", "square"]      # NO sin, cos, sqrt, inv
constraints       = {"^": (-1, 0)}   # base unconstrained; exponent cannot contain nested operators
complexity_of_operators = {"^": 3}    # mild parsimony penalty per `^`
```

A note on PySR's `constraints` semantics: the tuple `(left, right)` bounds
the **subtree complexity** of the two operands, not their numeric
values. `-1` means "no bound" (the left operand / base may be any
expression); `0` means "the operand cannot contain nested operators" —
in practice, a leaf such as a constant or a single input variable. So
`(-1, 0)` reads as *"base arbitrary, exponent must be leaf-like (e.g.
the literal `2`, or `x0`, or a fitted constant)"*. This forbids
`(complex)^(complex)` patterns without forbidding `x^2` or `x^c` for
fitted `c`. (The exact boundary between "leaf" and "near-leaf" can drift
between PySR versions — check the docs for your installed version if
you need a hard guarantee.)

**Why these operators**:
- Trig oscillates: derivatives flip sign repeatedly across the prior
  box, so the 5-point stencil sees noise and Fisher conditioning
  collapses.
- Explicit `inv` (`1/x`) has a pole at `x=0`; `sqrt(x)` has a branch
  point and ill-defined derivatives there. Both can be inside the prior
  box for normalized features.
- Unconstrained `^` is the worst: PySR's genetic search readily finds
  `(complex)^(complex)` patterns that fit fid well but blow up by orders
  of magnitude at the prior boundaries.
- **Note that binary `/` is still allowed.** It can in principle
  produce the same kind of singularity as `inv`, but the
  finite-prediction filter and 5%-rel-err quality gate (§ 4) screen
  out divisions that misbehave on the prior box. The empirical finding
  was that the explicit *unary* `inv` was disproportionately tempting
  to the symbolic search; binary `/` paired with sensible operands has
  been fine in practice.

The dropped operators have been tried in ablations and found to break
σ-conditioning; see `docs/PAPER_NOTES.md § 2` for the log.

**(ii) Dimension-balanced ANOVA loss** (more principled, marginal in
practice):

```
  L(prediction, target, X)  =  MSE  +  α · Σ_d  corr²( residual, X_d )
                                       d
```

`corr` is Pearson correlation across the batch. The intent: weakly-coupled
θ (`omegamh2`, `hireionz`, `bhfeedback`) give per-θ residual variance
that's small relative to the (k, z, r)-driven variance, so plain-MSE
PySR is tempted to drop `x0` (parsimony beats the tiny MSE win of
using it). The penalty explicitly punishes any leftover
`corr(residual, x0)`, pushing the search toward x0-using equations.

> **Honest assessment of the loss's impact.** It's a more principled
> objective than plain MSE — it directly targets the failure mode
> ("x0 dropped because residual-x0 correlation didn't lower MSE
> enough"). But in practice it doesn't move the headline σ-table much.
> The parameter the loss actually unlocks is `bhfeedback` (the AGN
> feedback term), which is **already heavily prior-bound**
> (`σ_prior = 0.005` is tight relative to its data information). So
> whether `bhfeedback` routes through PySR or through GP-slice, its σ
> is dominated by the prior either way — the σ-ratio sits at 1.00–1.02×
> regardless. `omegamh2` and `hireionz` are similarly priored or
> weakly-constrained. So we use the ANOVA loss because it's the
> theoretically right thing to do (and avoids the embarrassment of
> ablation reviewers asking why we didn't), but don't expect it to
> swing the headline σ on the cosmologically interesting parameters.

> Theory files: `src/priya_forecast/refit_1d_pysr.py` (training recipe,
> 999 lines, top docstring lists steps 1–6 verbatim);
> `src/priya_forecast/dim_balanced_loss.py` (ANOVA penalty math);
> `src/priya_forecast/models/normalization.py` (per-(z,k) round-trip).

---

## 4. Picking the equation from PySR's Pareto front

PySR returns a Pareto front: ~20 equations of increasing complexity,
each the best of its complexity. We never just take "lowest loss" — we
filter, then rank.

**Filter** (`is_fisher_stencil_safe` in `pareto_filters.py` for the
finite-prediction check; the gate itself runs in
`scripts/multi_z_aggregate.py:111-129` after the per-1D refits land):
- Drop eqs whose prediction is non-finite at any prior-box extreme
  (i.e., evaluating at the corners of the θ_i prior, swept across the
  full `(k, z)` grid). This catches eqs that blow up at boundaries.
- Drop eqs that don't reference `x0` (i.e. don't depend on θ_i) — those
  have `∂eq/∂θ_i ≡ 0` along the parameter axis and contribute nothing
  to that param's Fisher entry.
- Drop eqs whose **training-set mean** LF or HF flux rel-err exceeds
  **5%** (`REL_ERR_THRESHOLD = 0.05` in `multi_z_aggregate.py:111`).
  The gate is on the *training* rel-err, not a separately held-out
  Sobol set; the held-out validation is the multi-D Sobol hold-out
  reported in § 6, which evaluates the *whole hybrid model*, not a
  single per-1D fit.

**Rank** survivors by:
1. Most-θ-used (eqs that reference all 4 of `x0..x3` are preferred over
   ones that drop `x2` or `x3`).
2. Among ties, simplest (lowest complexity).

If no eq survives → the parameter is routed to GP-slice fallback (§ 2).

> Theory file: `src/priya_forecast/pareto_filters.py` (158 lines).

---

## 5. Computing Fisher

With `P̂(θ; k, z)` defined by § 2, Fisher is

```
  F_{ij}  =  ( ∂P̂/∂θ_i )ᵀ  C⁻¹  ( ∂P̂/∂θ_j )
```

where `∂P̂/∂θ_i` and `∂P̂/∂θ_j` are gradient vectors over the **full
stacked `(k, z)` grid** and `C` is the full KSData covariance over the
same layout (not block-diagonal in z — see § 1).

**Gaussian priors**. An independent Gaussian prior on `θ_i` with std
`σ_{prior,i}` adds an information term to the diagonal of `F`:

```
  F_{ii}  ←  F_{ii}  +  1 / σ²_{prior,i}
```

(This is the well-known result that Gaussian priors compose additively
with the data Fisher when both are Gaussian — the inverse-variances
sum.) Phase 1.5 adds these for `hub` (σ=0.015), `omegamh2` (σ=0.001),
`bhfeedback` (σ=0.005), and `tau0` (σ=0.331); see
`docs/PAPER_NOTES.md § 5c` for where each value comes from (Kim+
mean-flux constraints + the prior-width convention used elsewhere in
PRIYA).

The derivatives are **5-point centered stencils**:

```
                  −P̂(θ + 2h ê_i) + 8 P̂(θ + h ê_i) − 8 P̂(θ − h ê_i) + P̂(θ − 2h ê_i)
  ∂P̂/∂θ_i ≈ ─────────────────────────────────────────────────────────────────────────
                                              12 h_i
```

**Adaptive step halving**: `h_i` starts at `step_frac · (prior_hi − prior_lo)`
and halves until the relative change in the diagonal `F_{ii}` drops
below `rel_tol`. Each parameter halves independently because curvature
scales differ wildly (e.g. omegamh2 vs hireionz).

The output bundle: `F` (matrix), `cov = F⁻¹`, `σ = sqrt(diag(cov))`,
`corr` (correlation), `steps` (converged h_i — useful diagnostic).

> Theory file: `src/priya_forecast/fisher.py` (417 lines, top docstring
> is the math reference).

---

## 6. Phase 1.5 production scope and headline numbers

**Configuration**:
- Emulator: `kodiaq_2_2_4_6-48-48` (KODIAQ-SQUAD + XQ-100 multi-fidelity).
- z-grid: 9 bins `[2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2]`.
- k-grid: `linspace(0.005, 0.064, 32)` s/km. (Covers cosmic-variance
  floor < 0.005, peculiar-velocity dip near 0.02, LF resolution loss
  > 0.04.)
- Covariance: KSData (Karaçaylı+ 2021), conservative=True, k ≤ 0.064.
- Gaussian priors on `hub σ=0.015, omegamh2 σ=0.001, bhfeedback σ=0.005,
  tau0 σ=0.331` (Kim mean-flux × prior width).
- `dtau0 = 0` fixed (Kim USE_TAU0_ONLY convention).
- **Smart kwargs** (option B operators + ANOVA loss) for the **IGM
  thermal block only**: herei, heref, alphaq, hireionz, bhfeedback.
  Cosmology block (tau0, ns, Ap, hub, omegamh2) uses default kwargs.
- **No pair coupling.** That's Phase 2.

**Phase 1.5 headline** (11-θ joint Sobol hold-out at z=3.6, n=64):
```
  mean rel-err  =  3.27 %
  p99  rel-err  = 12.08 %
  max  rel-err  = 23.93 %
```

**Phase 1.5 σ-table** (selected, σ_PySR / σ_GP):

| param      | ratio   | route      |
|------------|---------|------------|
| ns         | 1.40×   | PySR       |
| Ap         | 0.79×   | PySR       |
| tau0       | 1.26×   | PySR       |
| hub        | 1.26×   | PySR       |
| omegamh2   | 0.99×   | GP-slice   |
| herei      | 5.90×   | PySR       |
| heref      | 0.88×   | GP-slice (gated) |
| alphaq     | 1.30×   | PySR       |
| hireionz   | 1.01×   | GP-slice   |
| bhfeedback | 1.00×   | GP-slice (gated) |

The IGM thermal block ratios > 1× are partly **intrinsic** to those
parameters: their posteriors are non-Gaussian, so the Fisher
approximation (which assumes a Gaussian posterior at the fiducial
point) does not match the MCMC marginal width — even when the model is
the GP itself. The mismatch can go in either direction depending on
curvature, priors, and parameter-space boundaries; for these params it
manifests as σ_GP / σ_MCMC = 1.6–4.5× at θ_target_simdat. See
`docs/PAPER_NOTES.md § D8.5` for the full decomposition.

**Where to read the results yourself**:
- `results/refit_optionC_z2.6-4.2_phase1_5_ksdata/scorecard.md` —
  per-parameter σ-table + route + x0-usage flag.
- `results/refit_optionC_z2.6-4.2_phase1_5_ksdata/per_param_summary.md`
  — the actual chosen PySR equations, one per parameter.
- `results/refit_optionC_z2.6-4.2_phase1_5_ksdata/corner.pdf` —
  Fisher corner plot.
- `results/holdout_multid_phase1_5/holdout_multid.md` — the 11-θ
  Sobol hold-out table the headline numbers come from.
- `results/closure_at_simdat_ind15_phase1_5_ksdata/scorecard.md` —
  off-fid Fisher closure at θ_target_simdat (σ_PySR / σ_MCMC).

---

## 7. Reading map (theory files, in math order)

Read in this order — each file's top docstring states the math it
implements; the code is below that. **Skip the `scripts/` directory
entirely** until you understand all of these.

| # | File | What math it carries |
|---|------|---------------------|
| 1 | `src/priya_forecast/parameters.py` | The 11 PRIYA params: names, fid, prior bounds. (One-page contract.) |
| 2 | `src/priya_forecast/models/base.py` | `P1DModel` ABC: `predict(θ, k, z) → P_F`. The whole forecast plugs in here. |
| 3 | `src/priya_forecast/models/normalization.py` | The per-(z,k) flux normalization round-trip. Read first because everything else imports it. |
| 4 | `src/priya_forecast/refit_1d_pysr.py` | Per-1D PySR training: Sobol sample, normalize, stack LF+HF, hand to PySR, save best Pareto eq. § 3 of this doc. |
| 5 | `src/priya_forecast/dim_balanced_loss.py` | ANOVA main-effect penalty: `L = MSE + α·Σ corr²(res, X_d)`. § 3c(ii). |
| 6 | `src/priya_forecast/pareto_filters.py` | Pareto-pick rules (`is_fisher_stencil_safe`, x0 used, max-rel-err gate). § 4. |
| 7 | `src/priya_forecast/models/pysr_model.py` | Wraps a saved equation back into a `P1DModel`. Closes the loop from training back to forecasting. |
| 8 | `src/priya_forecast/refit_taylor.py` | The 1D → multi-D additive Taylor combine. `MultiZAdditiveTaylorModel.predict` (line 206) is the formula in code. § 2. |
| 9 | `src/priya_forecast/likelihood.py` | Gaussian log-L `−½ (d−m)ᵀ C⁻¹ (d−m)`. |
| 10 | `src/priya_forecast/ksdata_likelihood.py` | KSData covariance loader. |
| 11 | `src/priya_forecast/fisher.py` | The Fisher computation: 5-point stencil + adaptive step halving + diagonal-convergence rule. § 5. |

After all 11, you can read any script in `scripts/` and immediately
recognize it as orchestration around these primitives.

---

## 8. Running things (one-time setup, brief)

```bash
git clone <this repo> && cd priya-forecast
pip install -e ".[forecast,pysr,gp,dev]"
# Point PYTHONPATH at the upstream lya_emulator clone + this repo's src/.
# On Greatlakes the upstream repo is already at
#   /home/mfho/student_projects/lya_emulator_full
# On a fresh machine, clone https://github.com/sbird/lya_emulator first.
export PYTHONPATH=/path/to/lya_emulator_full:$PWD/src
```

To reproduce a Phase 1.5 result, the canonical pipeline is:

1. **Train**: `scripts/refit_all_11_params.py` (calls
   `refit_1d_pysr.py` per parameter; produces 4-feature multi-z
   `Refit1DResult` objects with `variables = (θ_i, k, resolution,
   z_norm)`).
2. **Aggregate**: `scripts/multi_z_aggregate.py` (loads the per-1D
   refits, applies the 5%-rel-err quality gate from § 4, builds
   `MultiZAdditiveTaylorModel`, computes Fisher).
3. **Validate**: `scripts/holdout_multid.py` (multi-D Sobol hold-out,
   the headline rel-err in § 6).
4. **Off-fid closure**: `scripts/closure_at_simdat_target.py`
   (σ_PySR vs σ_MCMC at θ_target_simdat).

> **Do NOT use `scripts/train_and_forecast.py` to score Phase 1.5
> equations.** That script is the simpler **single-z student-recipe**
> reward loop (eBOSS DR14 covariance, 2- or 3-feature equations
> trained at one z); it cannot consume the multi-z 4-feature
> production refits because its YAML schema and hard-coded `fix:`
> block don't allow a `z_norm` feature. See `README.md` for what the
> student loop actually scores.

**Do not start by reading any of these scripts** — read the theory
files in § 7 first, then the scripts will be self-explanatory.

---

## 9. Where to ask for help

- Math / methodology questions → check `docs/PAPER_NOTES.md` first
  (§ 1–6 = Phase 1 design, § D1–D9 = production design decisions, with
  reasoning for every choice). Then ask the team.
- Code / bug reports → file in this repo's GitHub issues.
- Upstream GP emulator → `lya_emulator_full` repo (sbird).
- PySR training pipeline (the student's original recipe) →
  `priya_pysr` repo.

For Phase 2 (per-pair PySR cross-coupling) and Phase 3 (Ap σ-ratio
remediation), come back here once Phase 1.5 makes sense — those build on
top of everything above.
