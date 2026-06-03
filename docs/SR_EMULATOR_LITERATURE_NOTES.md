# Symbolic-emulator literature — recipes & gap analysis for our P1D SR pipeline

**Date:** 2026-06-03
**Purpose:** Distill the cosmology symbolic-regression-emulator literature into a
gap analysis against our Lyman-α P1D PySR + Fisher-forecast pipeline, and
identify the design points we are missing. Feeds Stage 8 (derivative loss).

**Provenance:** Manual synthesis from three directly-fetched sources plus domain
knowledge. (An automated deep-research harness run failed — arxiv PDF extraction
broke its agents — so this is not from that run.)

## Sources

- **syren-baryon** — Kammerer, Bartlett, Kronberger, Desmond, Ferreira 2025,
  [arXiv:2506.08783](https://arxiv.org/abs/2506.08783). Analytic/symbolic
  emulators for the baryonic suppression of the matter power spectrum (CAMELS:
  Astrid, IllustrisTNG, SIMBA, Swift-EAGLE + baryonification). The anchor paper.
- **syren (linear P(k))** — Bartlett et al.,
  [arXiv:2311.15865](https://arxiv.org/html/2311.15865). The foundational recipe.
- **Symbolic Emulators for Cosmology (review)** —
  [arXiv:2510.18749](https://arxiv.org/html/2510.18749v1) (Phil. Trans. R. Soc. A).
- **Fisher's Mirage** — [arXiv:2406.06067](https://arxiv.org/abs/2406.06067).
  Our known diagnosis of derivative-unfaithful surrogates biasing Fisher σ.

## The syren recipe (consistent across the family)

| Element | syren | us |
|---|---|---|
| Target | log of a **ratio to a physical baseline**: `P = P_EH · F`, regress `log F`; MSE → fractional error + positivity | `log P` (Stage 6): fractional ✓, but **no ratio-to-reference** |
| Target scaling | `log F × 100` so target is O(1) | per-(z,k) mean/std normalize |
| Physical limits | enforce **`log F → 0` as `k → 0`** analytically (post-hoc) | only the at-fiducial anchor in the combine |
| Engine / operators | **Operon** GP; `+ − × log pow cos` + **analytic quotient `aq(x,y)=x/√(1+y²)`** (bounded, pole-free) | PySR; trig dropped; raw `/` |
| Selection | Pareto + ε-dominance; **reject train/val loss gap**; manual interpretability pick | Fisher-safety filter → best_loss |
| Accuracy | 0.2–0.6% RMS fractional error on P(k) | — |
| Derivatives | analytic → gradient-based NUTS (60× faster); **derivative accuracy never validated** | Fisher needs accurate `∂P/∂θ` — our core problem |
| UQ | ships **uncertainty expressions** (fitting + stochastic) | none |
| Architecture | **one joint expression over all params** | **per-parameter 1D refits + additive Taylor combine** |

## Prioritized recommendations

### Well-supported (multi-source) — high value

1. **Fit the ratio-to-reference, not raw log P.** Make each per-parameter SR
   target `log[P(θ)/P(θ_fid)]` — the fractional *response* to that parameter.
   Its derivative is `∂logP/∂θ`, exactly the Fisher quantity, so this attacks
   Fisher's-Mirage at the target level and composes with the anchor + Stage 8.
   **Biggest lever.** (syren fits `log[P/P_ref]` universally.)

2. **Adopt `aq(x,y)=x/√(1+y²)`; drop raw division.** Raw `/` makes poles and
   spurious curvature near zeros — a mechanical cause of derivative-unfaithful
   equations. `aq` is bounded/smooth → well-behaved derivatives. Low-effort
   (PySR custom binary operator), high-leverage for the Mirage.

3. **Train/validation-gap rejection gate** in equation selection (syren rejects
   Pareto members whose train and val losses diverge). Cheap overfitting guard
   that correlates with Mirage equations.

### Principled, single-source / needs adaptation — medium

4. **Enforce physical asymptotic limits** (syren's `log F → 0` as `k → 0`).
   Our analogue: response → 0 as `θ → θ_fid` (have it via anchor) + optional
   sign/monotonicity constraints on `∂P/∂θ`. Needs a custom-loss penalty in PySR.

5. **Emulator-error expressions** propagated into the Fisher covariance rather
   than treating σ_PySR as exact — a more honest forecast.

### Flag — contradicts current design (tradeoff, not a directive)

6. **syren fits ONE joint multivariate expression; we fit per-parameter 1D
   refits + an additive combine that drops cross-terms.** We have a documented
   **herei×alphaq coupling** (`memory/headline_findings.md`) that an additive
   combine cannot represent; joint SR captures interactions. Tradeoff: joint SR
   is far harder to search and less interpretable. **Suggested experiment:** a
   joint 2-parameter refit on the herei–alphaq pair to measure how much the
   additive combine leaves on the table.

## What is genuinely novel about our approach

The literature does **not** solve Fisher's-Mirage. syren et al. validate *value*
accuracy and use derivatives downstream for sampling, but **none validate
derivative accuracy** and none use a derivative-matching loss. Therefore:

- **Stage 8 Sobolev loss** `‖∂_θ logP_SR − ∂_θ logP_GP‖²` is a genuine extension,
  not a reinvention. Sobolev training is established in PINN/ML but, per these
  sources, unused in SR cosmology emulators. Pursue it.
- A **derivative-validation selection gate** (reject on
  `median|∂logP_SR/∂logP_GP − 1|`, not value RMSE) is the natural complement and
  also novel here.

## Net

Highest-ROI: **#1 (ratio-response target) + #2 (`aq` operator) + Stage 8 (Sobolev
loss)** — they attack the Mirage at the target, the operator, and the loss
respectively. **#6 (joint SR for known couplings)** is the deeper architectural
question worth a scoped experiment.
