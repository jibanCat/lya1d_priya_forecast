# Phase 2: per-pair PySR cross-coupling fits — plan

> Forward-looking plan for the next iteration of the PRIYA Lyα P1D
> symbolic-emulator pipeline. Written 2026-05-04, after PR #1 (Phase 1)
> closes. Phase 1 = per-1D + additive Taylor + multi-z + KSData covariance.
> Phase 2 = adds per-pair PySR cross-coupling on top.

## Motivation

Phase 1's per-1D + additive-Taylor combine is structurally rank-correct: each
θᵢ has its own 1D PySR equation, so its θ-gradient direction is independent
from every other parameter's by construction. Fisher stays full-rank.

**It cannot capture cross-coupling.** Multi-D PySR (a single equation over the
6-θ cross-coupled subset) was attempted in Phase 1 and abandoned — see
`PAPER_NOTES.md § D5.5` for the rank-deficiency post-mortem. PySR's Pareto
front rewards compact eqs that fold features into a shared `exp(·)` group;
those have rank-1 first-order θ-behavior; Fisher block becomes rank-deficient.

Phase 2 sidesteps this: instead of one joint multi-D eq, fit **one small PySR
eq per parameter pair**, on the residual after subtracting Phase 1's per-1D
prediction. Each pair adds *one* new gradient direction in the (θᵢ, θⱼ) plane,
orthogonal to every per-1D direction. Rank stays full **by construction**.

## Math: rank-additive ANOVA decomposition

Phase 1 prediction (recap):

    P̂_phase1(θ; k, z, r) = P_GP^HF(fid; k, z, r)
                         + Σᵢ [P̂ᵢ(θᵢ; k, z, r) − P̂ᵢ(fidᵢ; k, z, r)]

Phase 2 prediction adds a "pure interaction" term per fitted pair:

    P̂(θ) = P̂_phase1(θ)
          + Σ_{(i,j) ∈ pairs}
              [Ĝ_ij(θᵢ, θⱼ)  − Ĝ_ij(θᵢ, fidⱼ)
                              − Ĝ_ij(fidᵢ, θⱼ)
                              + Ĝ_ij(fidᵢ, fidⱼ)]    (k, z, r suppressed)

The bracketed quadrant cross-difference is the standard functional ANOVA
**pure 2-way interaction** term. Properties:

- **At fid**: every bracketed term is 0; P̂ collapses to P_GP^HF(fid). Hybrid
  exact at fid, by construction.
- **At one off-fid θᵢ (others at fid)**: only the per-1D main effect for i
  contributes; pair terms are 0. Phase 1 result recovered exactly.
- **Each pair adds an independent gradient direction** in the (θᵢ, θⱼ) plane.
  Fisher rank ≥ rank(Phase 1) + |pairs|.
- **Graceful degradation**: if pair (i, j) signal is small, PySR fits Ĝ_ij ≈ 0
  and the cross-difference is 0 to machine precision. Adding a redundant pair
  costs compute but cannot harm Fisher.

## Pair selection: simdata-MCMC posterior correlations

We pick pairs from posterior correlations on synthetic data (clean, no
boundary effects), not from the real-data MCMC chain (boundaries compress ρ).

Source: `chains/simdat/s-simdat-ind15-48-z2.6-4.2-loo-nodatacorr-noemuerror-optimiseGP-discardkbins-0.005-0.064/simdat-48-z2.6-4.2.1.txt`
(49 720 weighted rows; cached at `results/simdat_ind15_truth.npz`).

Top |ρ| ≥ 0.2 on the 11 forecast params:

| pair             | ρ_simdata | ρ_realdata | tier |
|---|---|---|---|
| tau0 × ns        | **−0.92** | −0.70      | must |
| Ap × alphaq      | **+0.68** | +0.07      | should |
| tau0 × Ap        | **−0.66** | +0.28      | should |
| ns × Ap          | +0.55     | +0.03      | maybe |
| tau0 × alphaq    | −0.55     | −0.28      | maybe |
| dtau0 × ns       | −0.51     | −0.15      | skip (dtau0 fixed) |
| ns × alphaq      | +0.43     | +0.13      | maybe |
| dtau0 × tau0     | +0.36     | −0.31      | skip |
| dtau0 × Ap       | −0.34     | −0.28      | skip |
| heref × alphaq   | +0.29     | +0.19      | maybe |
| Ap × omegamh2    | −0.25     | −0.01      | maybe |
| dtau0 × alphaq   | −0.24     | +0.12      | skip |
| **herei × alphaq** | **−0.22** | −0.26    | must |

**Selection tiers**:

- **Must-have (start here)**: `tau0×ns`, `herei×alphaq`. Strongest cosmology +
  IGM-thermal coupling pair respectively. Phase 5 coupling-matrix headline
  (herei × alphaq) is in this tier.
- **Should-have (add if must-have insufficient)**: `Ap×alphaq`, `tau0×Ap`.
  Together with `tau0×ns` they span the (tau0, ns, Ap, alphaq) entangled
  block that simdata reveals.
- **Maybe (only if corner still off)**: `ns×Ap`, `tau0×alphaq`, `ns×alphaq`,
  `heref×alphaq`. Likely redundant with the should-have set.

`dtau0` pairs are skipped: dtau0 is fixed at 0 in the Kim mean-flux
convention (PAPER_NOTES § D1).

## Phase 1 closure to σ_MCMC_simdat (motivates pair selection)

Per-1D + Taylor at θ=fid, KSData covariance, post-PR-#1 BLOCKER fix
(four refits gated to GP-slice: dtau0, omegamh2, hireionz, bhfeedback).
Numbers from `results/refit_optionC_z2.6-4.2_ksdata/scorecard.md`:

| param | σ_MCMC | σ_GP_Fisher | σ_PySR_Fisher | PySR/MCMC | route | flag |
|---|---|---|---|---|---|---|
| tau0 | 0.027 | 0.029 | 0.041 | 1.5× | PySR | OK |
| ns | 0.058 | 0.044 | 0.058 | 1.0× | PySR | ✓ closed |
| herei | 0.147 | 0.12  | 0.52  | 3.5× | PySR | needs pair |
| heref | 0.162 | 0.39  | 2.35  | 14×  | PySR | **biggest miss** |
| alphaq | 0.395 | 0.39 | 0.32  | 0.8× | PySR | OK (was 0.6× pre-fix) |
| hub | 0.011 | 0.012 | 0.015 | 1.4× | PySR | OK |
| omegamh2 | 0.0017 | 0.001 | 0.001 | 0.6× | GP-slice | overconfident (priored, GP-slice gated) |
| hireionz | 0.43 | 1.56 | 1.61 | 3.7× | GP-slice | per-1D eq broken; routed via GP-slice (gated) |
| bhfeedback | 0.0035 | 0.0049 | 0.005 | 1.4× | GP-slice | OK (gated) |
| Ap | (recompute units) | 0.174 | 0.134 | 0.77× | PySR | mildly overconstrained |

Patterns:
- **`heref` is 14× too loose** at fid — single largest per-1D failure.
  Likely a per-1D problem (curvature at fid), not a pair-coupling problem.
  Phase 1.5 (below) addresses this before adding pair fits.
- **`herei` 3.5× too loose** — second-biggest miss. Same diagnosis: likely
  per-1D fid-curvature mismatch. Phase 1.5 again.
- **`alphaq` 0.8× tighter than σ_MCMC** — was 0.6× pre-BLOCKER fix; the
  gate fix lifted it. Marginal overconfidence remaining; possibly fixed by
  ANOVA loss in Phase 1.5.
- **`omegamh2` 0.6×** — production prior σ=0.001 dominates; Fisher numeric
  floor. Expected, not a bug.

## Phase 1.5: heref/alphaq smart per-1D refits (BEFORE pair fits)

Both `heref` (14×) and `herei` (3.5×) miss σ_MCMC by enough that pair
coupling alone is unlikely to recover them — these look like per-1D
fid-curvature problems, not interaction problems. User direction
(2026-05-04): **don't naively bump niter** since these params are weakly
sensitive to P_F at fid (more genetic-search time can't find signal that
isn't there). Architectural fixes only.

Three smart-refit changes specifically for `heref`, `herei`, `alphaq`:

1. **Wire `JULIA_LOSS_FUNCTION_ANOVA` into the per-1D path**. Currently
   `src/priya_forecast/refit_1d_pysr.py:70` uses
   `elementwise_loss="loss(prediction, target) = (prediction - target)^2"`
   (plain MSE). The ANOVA loss already exists in
   `src/priya_forecast/dim_balanced_loss.py` (verified 2026-05-04: only
   wired into the abandoned multi-D path at `refit_multi_d.py:339`). One-line
   change: replace `elementwise_loss=...` with
   `loss_function=JULIA_LOSS_FUNCTION_ANOVA` in `DEFAULT_PYSR_KWARGS`.
   Effect: penalizes batch-level main effects on dropped features → forces
   PySR to use θ even when k+z+r alone could lower MSE.

2. **Restrict unary operators** for the IGM-thermal smart refits. Drop
   `inv` and `sqrt` (the two operators producing sharpest fid-curvature),
   keep `exp`, `log`, `square`. Smoother eqs → better-behaved Fisher
   stencil derivatives. Configurable via per-param override of
   `DEFAULT_PYSR_KWARGS["unary_operators"]`.

3. **Keep niter at 50, maxsize at 20**. No naive bump — the architectural
   fix (loss + operators) is the lever, not search depth.

Refit just `heref`, `herei`, `alphaq` (the three flagged in the table
above). Cosmology + mean-flux + bhfeedback are already at 1.0–1.5×; don't
disturb.

**Expected outcome** (hypothesis): heref drops from 14× to ≤ 5×; herei
drops from 3.5× to ≤ 2×; alphaq stays around 0.8× or improves to closer
to 1.0×. If alphaq becomes more conservative, that's a win — overconfident
σ is a worse paper bug than under-confidence.

**Compute**: ~3 min wall (3 params × 50 s SLURM each, parallel).

**Decision gate**: re-run aggregator after Phase 1.5; if heref still > 5×
or herei still > 2×, that's structural cross-coupling, proceed to Option α
pair fits (`herei × alphaq`, `heref × alphaq` if needed). If they close to
≤ 2× without pair fits, Phase 2 only needs the cosmology pair `tau0 × ns`.

## Implementation: Option α — sequential per-pair on residuals

Per pair (i, j):

1. **Generate Sobol training samples** in (θᵢ, θⱼ) plane, all other θ at fid:
   - N_samples = 256 (or 128 if compute-tight) per (k, z) bin.
   - Sobol seed=42; first scrambled.
   - Inputs: 5 = (θᵢ_norm, θⱼ_norm, k_norm, r ∈ {LF, HF}, z_norm).

2. **Compute residual target** at each sample:

       residual = P_GP(θᵢ, θⱼ, others=fid; k, z, r)
                − P̂_phase1(θᵢ, θⱼ, others=fid; k, z, r)

   where P̂_phase1 is the cached per-1D + Taylor combine. Both LF and HF
   stacks computed for the resolution feature.

3. **Normalize** the residual using the same per-(z, k) at-fid-anchored
   convention as per-1D (`(target − mean_per_(k, z)) / std_per_(k, z)` with
   mean_per_(k, z) = `P_GP_LF(fid, k, z)` cancelling). See PAPER_NOTES § D2/D4.

4. **PySR fit `Ĝ_ij`** with 5 inputs.
   - Operators: same as per-1D (no trig, see § 2a). Binary `+ − * /`,
     constrained `^`, unary `exp log square sqrt inv`.
   - Settings: niter=50, maxsize=20, multithreading procs=8, random_state=42.
   - **Pareto pick**: prefer eqs that use BOTH `x0` AND `x1` (the two θ
     features); fall back to single-θ eq if no both-θ candidate. Same
     `is_eq_well_behaved` + `is_fisher_stencil_safe` +
     `has_pathological_constant` filters as per-1D.

5. **Cache as `Refit2DPairResult`** with `predict()`, `cross_difference()`
   methods. Integrate into a new `MultiZPairCoupledModel` (or extend
   `MultiZAdditiveTaylorModel`) by adding the bracketed cross-difference at
   `predict()` time.

**Compute budget**: ~1 h SLURM per pair (5D fit, niter=50, procs=8). Embarrassingly
parallel via SLURM array. 2 must-have pairs ≈ 1 h wall. 4 must+should ≈ 1 h wall.

## Implementation: Option β (fallback) — TemplateExpressionSpec joint fit

Use only if Option α fits emulator interpolation noise (residual signal too weak).

PySR ≥ 1.x supports `TemplateExpressionSpec`
(https://github.com/MilesCranmer/PySR/discussions/787). Fix the outer form:

    P̂_norm(θ, k, z, r) = Σᵢ fᵢ(θᵢ_norm, k_norm, r, z_norm)
                       + Σ_{(i,j)∈pairs} g_ij(θᵢ_norm, θⱼ_norm, k_norm, r, z_norm)

PySR fits all `fᵢ` and `g_ij` simultaneously over 11+ features, allocating
expression budget to each. Each `f` is internally 4-D (low-D, where PySR
shines); each `g` is 5-D.

**Compute**: ~12 h SLURM (joint search over many sub-expressions).

**Risk**: untested on this cluster's PySR version. Verify `pysr.TemplateExpressionSpec`
is importable before committing.

## Validation strategy: GP-Fisher vs PySR-Fisher at θ_target_simdat

Three benchmarks at the same physical scenario:

1. **σ_MCMC_simdat** — full nonlinear Bayesian σ from
   `simdat-48-z2.6-4.2.1.txt` chain at θ_target_simdat. Truth.
2. **σ_GP_Fisher** at θ_target with KSData covariance — Gaussian
   linearization of (1). Tests whether Gaussianity holds at θ_target.
3. **σ_PySR_Fisher** at θ_target with KSData covariance — what we produce.
   Tests symbolic-emulator faithfulness.

(2) ↔ (3) is the head-to-head closure test. (1) ↔ (2) is a Gaussianity
sanity check (not our pipeline's fault if it fails). (1) ↔ (3) is the
bottom-line "does the paper figure work?".

Two test points (cheap to do both):
- **At θ = fid**: PySR hybrid ≡ GP at fid by construction. Diagnostic
  only — exercises gradient closure.
- **At θ = θ_target_simdat (with dtau0 → 0)**: real off-fid test. Phase 1's
  Taylor extrapolation is non-trivial here. Σ_MCMC available for reference.

θ_target_simdat (cached at `results/simdat_ind15_truth.npz`):
```
dtau0=0.0114  tau0=1.003  ns=0.902   Ap=1.887e-9  herei=3.773
heref=2.910   alphaq=2.272 hub=0.699 omegamh2=0.142
hireionz=7.946 bhfeedback=0.0502
```

For closure: force `dtau0 → 0` to match Kim mean-flux convention.

**Paper figure (planned)**: per-param σ-ratio bar chart at fid vs at
θ_target_simdat, with σ_MCMC overlay as horizontal lines. Color-coded by
treatment (per-1D / per-pair / GP-slice). Reveals which params close at fid
but not off-fid → motivates pair selection.

## Cobaya MCMC with PySR (future-future work, not Phase 2)

The Cobaya likelihood class `lyaemu.likelihood.CobayaLikelihoodClass` could
be subclassed to swap the GP for the PySR hybrid. Result: full MCMC sampling
with PySR. Test: does it converge to the same posterior as GP-MCMC at
θ_target_simdat?

If yes → PySR is good for nonlinear sampling, not just Cramer-Rao Fisher.
Strong selling point. Requires:
- PySR hybrid `predict()` + analytic `gradient()` to drive Cobaya.
- Compatible Cobaya likelihood interface.
- ~1 day implementation, ~12 h compute for one MCMC run.

Not for this paper. Reserved for follow-up paper or appendix.

## Risks and mitigations

| risk | likelihood | mitigation |
|---|---|---|
| Residual fits emulator noise (Option α) | medium | Validate σ at θ_target_simdat ↔ σ_MCMC; if pair improves Fisher beyond noise level, ship; else fall back to Option β |
| `heref` 14× failure cannot be fixed by pairs | medium | Re-fit per-1D for heref alone with niter=200 first; if still 14× off, investigate prior-vs-emulator mismatch on heref |
| TemplateExpressionSpec not on cluster | low | `pip install pysr --upgrade` in the user mamba env; smoke-test before committing |
| Pair fits add complexity user finds hard to read | low | Per-pair eqs printed in `per_param_pair_summary.md` with prettified θ names; Sympy simplification |
| Off-fid Taylor extrapolation fails badly | high | This is what we're testing in step (3) below. If σ_PySR at θ_target diverges from σ_GP, that's diagnostic — investigate before adding pairs |

## Timeline (after PR #1 merges)

| Step | Wall time | Description |
|---|---|---|
| 0. Off-fid closure plot at θ_target_simdat | ~1 h | Compute σ_GP, σ_PySR at θ_target with Phase-1-only hybrid (dtau0→0); overlay σ_MCMC_simdat. Diagnostic: which params lose σ-closure off-fid? |
| 1. **Phase 1.5**: smart refit of heref + herei + alphaq | ~30 min | Wire ANOVA loss into per-1D `DEFAULT_PYSR_KWARGS`; restrict operators to {exp, log, square}; refit just these 3 params; re-aggregate scorecard |
| 2. Decision gate | ~5 min | If heref ≤ 5× and herei ≤ 2× after Phase 1.5 → may skip pair fits for IGM block; if not, proceed to step 4 |
| 3. Scaffold `Refit2DPairResult` + `MultiZPairCoupledModel` + tests | ~half day | New module; reuse per-1D normalization spec |
| 4. Sobol-pair payload generator + per-pair PySR driver + SLURM | ~1 h | Mirror `precompute_payloads.py` + `refit_one_param.py` |
| 5. Run must-have pairs (2× SLURM, parallel) | ~1 h | `tau0×ns` (always), plus `herei×alphaq` (if Phase 1.5 didn't close herei) |
| 6. Aggregate + Fisher + scorecard | ~30 min | Updated multi-z aggregator that calls pair model |
| 7. Re-validate at θ_target_simdat | ~30 min | Same closure plot as step 0; decide whether to add should-have |
| 8. (If needed) should-have pairs | ~1 h | `Ap×alphaq`, `tau0×Ap` |
| 9. Update paper figures + PAPER_NOTES | ~half day | Final scorecard + corner |
| **Total** | **~2 days** | If Phase 1.5 closes IGM block + must-have pairs suffice |

## Open questions for the user (resolved 2026-05-04)

1. ~~**σ_alphaq overconfidence**~~: addressed in Phase 1.5 (ANOVA loss).
2. ~~**σ_heref 14× off**~~: addressed in Phase 1.5 (operator restriction +
   ANOVA loss) before adding pair fits. User direction: don't naively bump
   niter; weakly-sensitive params need architectural fixes.
3. ~~**dtau0 at θ_target_simdat**~~: confirmed dtau0=0 (Kim convention),
   bias to truth is <0.2σ_MCMC[dtau0].
4. ~~**Cobaya MCMC validation**~~: confirmed follow-up paper / appendix,
   not in this paper's scope.

## References

- `docs/PAPER_NOTES.md` — Phase 1 design decisions D1–D6, multi-D post-mortem D5.5.
- `LOCAL_PAPER_HANDOFF.md` — local paper-writing replay guide.
- `results/simdat_ind15_truth.npz` — cached θ_target_simdat + MCMC σ + ρ.
- PySR docs: https://astroautomata.com/PySR/
- TemplateExpressionSpec: https://github.com/MilesCranmer/PySR/discussions/787
- SAGE additive-GP analog: arXiv:2410.00931
