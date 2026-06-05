# Stage 9 — Sobolev derivative loss: outcome

**Date:** 2026-06-04
**Branch:** `stage9-sobolev-loss`
**What shipped:** a custom Julia Sobolev `loss_function` (finite-diff tree
derivative in-loss + per-fidelity GP target gradient via the `weights` channel),
wired into the single-z refit. **λ = 5.** 11/11 refits + forecast on KODIAQ z=3.6.

## Headline: ns recovered — GP-slice set back to {hub, bhfeedback}

The whole arc was about Fisher's-Mirage (value-accurate, derivative-unfaithful
equations). The three methods, by per-parameter gradient faithfulness
(`median_k |∂eq/∂θ ÷ ∂logP_GP/∂θ − 1|` at fid; gate threshold 0.25):

| param | value-loss (Stage 8) | ratio-response spike | **Sobolev (λ=5)** |
|---|---|---|---|
| **ns** | 0.69 ✗ (all rejected) | 0.07 ✓ | **0.134 ✓** |
| **hub** | 0.92 ✗ | 0.92 ✗ | **0.93 ✗** (even λ=15) |

Production forecast GP-slice set:
- **Stage 8** (value-loss + gate): `{ns, hub, bhfeedback}` — ns rejected = failure.
- **Stage 9** (Sobolev + gate): **`{hub, bhfeedback}`** — ns recovered.

**This meets the acceptance criterion** ("only hub + bhfeedback may GP-slice,
no more"). ns — the headline science parameter — now has a faithful symbolic
equation where the value loss could produce none.

## The method hierarchy (the scientific result)

- **aq operator + gate (Stage 8):** a *filter* — rejects unfaithful equations
  but can't generate faithful ones. For ns the search never produced one →
  GP-slice.
- **Ratio-response target (lever #1):** *indirect generative* — reshapes the
  target so a good value-fit tends to imply a good gradient. Fixed 8–9/11
  including ns, but not hub; production wiring was fragile (multi-fidelity
  anchor) and was retreated.
- **Sobolev loss (Stage 9):** *direct generative* — penalizes the derivative
  error during the search. Recovers ns cleanly. λ=1 too weak (0.43); λ=5 works
  (0.13).

## hub is genuinely hard — independent of method

hub stays ~0.92–0.93 under value-loss, ratio-response, AND Sobolev (even λ=15).
No method makes ∂P/∂hub faithfully fittable with this operator set / single-z
training. This is a real finding, not an implementation gap — hub (and
bhfeedback, priored out) remain GP-slice fallbacks, which the pipeline handles
correctly (σ_GP/σ_perfect unaffected). A hub-specific investigation (is its
single-z gradient small/degenerate, or a k-shape the operators can't express?)
is the natural follow-up.

## On the σ metric

`forecast_table.txt` σ_PySR/σ_GP is recorded but **remains single-z-confounded**
(rank-deficient Fisher; dtau0 passes the gradient gate yet shows a wild σ ratio —
Stage 8 §). The meaningful Stage 9 result is the **per-parameter gradient
faithfulness** and the **narrowed GP-slice set** above. The clean σ comparison
is the well-conditioned **multi-z** forecast — the next step (mirror the Sobolev
refit + gate to multi-z), deferred.

## Cost note

The in-loss finite-diff doubles tree evaluations per loss call; the 11-param
λ=5 array still finished in ~14 min wall on yueyingn0 (%3) — acceptable.
