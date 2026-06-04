# Stage 7 — multi-z joint Fisher vs. Stage 6 single-z (z=3.6)

**Date:** 2026-06-03
**Run:** `forecast_only`, KODIAQ-SQUAD, z ∈ [2.6, 4.2] (joint), `target_space: log`,
additive combine. Refits: 11-param multi-z PySR array (yueyingn0), 8/11 with
usable Fisher-safe equations (3 GP-slice fallbacks — see caveats).

## Headline — multi-z lifts the IGM-thermal rank-deficiency

The single-z all-11-param Fisher was rank-deficient: the IGM-thermal parameters
were essentially unconstrained (σ_GP of tens to hundreds). Combining Fisher
information across the 9 KODIAQ z-bins (`F = Σ_z F(z)`, plus cross-z covariance)
collapses those degeneracies:

| param | single-z z=3.6 σ_GP | multi-z σ_GP | tighter by |
|---|---|---|---|
| herei | 26.68 | **0.355** | ~75× |
| heref | 94.35 | **1.07** | ~88× |
| alphaq | 235.3 | **1.43** | ~165× |
| hireionz | 86.35 | **4.34** | ~20× |
| dtau0 | 1.19 | **0.366** | ~3× |

This is the expected physical payoff — PRIYA's IGM-thermal parameters need
redshift leverage to be constrained, which a single z-bin cannot provide.

## σ_perfect_1D ≡ σ_GP (anchor identity holds in multi-z log-space)

Every parameter: σ_perfect_1D == σ_GP to 4 significant figures (forecast table)
and rtol 1e-3 in the gated test, in **both linear and log space**. The additive
Taylor combine reproduces the GP's first derivatives exactly; Fisher is
first-order, so the "3-σ ladder" collapses to σ_GP vs σ_PySR, as at single-z.

## Approach A vs. legacy per-z-sum (cross-z covariance)

`KSDataLikelihood`'s covariance is **not block-diagonal in z** (its own
docstring). The gated A-vs-B diagnostic on real KODIAQ:

| param | σ_A(joint) / σ_B(per-z-sum) |
|---|---|
| ns | 0.971 |
| Ap | 0.952 |
| tau0 | 0.963 |

Approach A (one z-spanning likelihood + `fisher_matrix`) gives ~3–5% **tighter**
σ than the legacy per-z-sum (`combine_fisher_phys_arrays`,
`scripts/multi_z_aggregate.py`). The legacy path was therefore biasing σ ~few-%
loose for cosmological parameters by ignoring cross-z covariance; **Approach A is
the correct production path.** (Effect may be larger for IGM-thermal params; the
diagnostic sampled ns/Ap/tau0.)

## σ_PySR vs σ_GP — Fisher's-Mirage status

Mirage persists, as expected (Stage 6 log-target attenuates but does not
eliminate it; Stage 8 Sobolev loss is the planned fix). σ_PySR/σ_GP:

| param | σ_PySR/σ_GP | | param | σ_PySR/σ_GP |
|---|---|---|---|---|
| ns* | 0.63 | | heref | 1.54 |
| Ap | 0.40 | | alphaq | 1.79 |
| hub | 2.08 | | hireionz | 2.34 |
| omegamh2 | 0.18 | | bhfeedback* | 0.38 |
| herei | 2.51 | | dtau0* | 0.56 |
| | | | tau0 | 0.54 |

- Mean |log10(σ_PySR/σ_GP)| ≈ **0.35** (vs Stage 6 single-z 0.366) — comparable;
  multi-z did not worsen the Mirage but did not fix it.
- Sub-1 (unphysically tight) ratios: 6/11. >1 ratios: 5/11.
- `*` = GP-slice fallback param (no real PySR equation); its σ_PySR differs from
  σ_GP only through marginalization against the other params' equations.

## Caveats / follow-ups

1. **3 GP-slice fallbacks (ns, bhfeedback, dtau0):** their multi-z 4-input PySR
   equations had **no Fisher-safe, x0-dependent member** on the Pareto front
   (all 11–15 rows rejected). These params exhausted the seed-retry loop
   (dtau0 ~59 min, ns >1 h wall) and still produced only marginal equations →
   GP-slice fallback. The forecast is valid (σ_GP/σ_perfect unaffected), but the
   σ_PySR story for these three is GP-slice, not symbolic.
2. **Refit efficiency:** the long retry loops come from `max_retries=4` × slow
   multi-z PySR fits. Consider lowering `max_retries` or loosening the
   Fisher-safe gate for multi-z. The literature levers in
   `docs/SR_EMULATOR_LITERATURE_NOTES.md` (ratio-response target, `aq` operator,
   derivative-validation gate) target exactly this Fisher-faithfulness problem
   and should improve the usable-equation yield — fold into Stage 8.
3. **Verification point #1 cleared:** the per-z k-grid uniformity guard
   (`shared_k_and_z_grid`) passed — KODIAQ uses a common kf grid across z-blocks,
   so Approach A's joint stacking is valid.

## Artifacts

`results/multi_z_stage7/`: `corner.png`, `forecast_table.txt`, `scorecard.md`,
`fisher_{GP,perfect_1D,PySR}.npz`, `refit/z2.6-4.2/{pareto,norm}_*.{csv,npz}` (11).
