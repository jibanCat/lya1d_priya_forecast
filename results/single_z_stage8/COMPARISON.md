# Stage 8 (cheap levers) — outcome & findings

**Date:** 2026-06-04
**Branch:** `stage8-sobolev-derivative-loss`
**What shipped:** the `aq` operator (pole-free division) + a derivative-validation
gate, wired into single-z equation selection. 381 tests pass.

## Headline: the cheap levers are a *filter*, not a *generator*

The aq operator + derivative gate work correctly — but they can only **reject**
derivative-unfaithful equations, not **make** the search produce faithful ones.
On the single-z z=3.6 production run, the gate (correctly) rejected ns, hub, and
bhfeedback because **no equation in their Pareto fronts had a faithful gradient**
(all 6 Fisher-safe ns equations had 69–97% gradient error — textbook Fisher's
Mirage: value-accurate, derivative-wrong). Rejecting ns — a headline science
parameter — to GP-slice is a failure, not a success.

## The single-z σ metric is confounded

| metric (single-z z=3.6) | Stage 6 | Stage 8 |
|---|---|---|
| mean \|log10(σ_PySR/σ_GP)\| | 0.366 | 0.677 (worse) |

This metric is **not trustworthy at single-z**: the all-11-param Fisher is
rank-deficient (σ_GP 26–235 for IGM-thermal params), so marginalized σ_PySR/σ_GP
is hypersensitive. Smoking gun: **dtau0 passed the gradient gate yet shows
σ_PySR/σ_GP = 23×** — a faithful-at-fid gradient producing a 23× marginalized
blow-up is pure ill-conditioning, not equation quality. The levers must be
evaluated on the well-conditioned **multi-z** forecast, not single-z.

## Lever #1 (ratio-response target) — validated by spike, NOT shipped

A spike fitting the **at-fid-anchored** target `log[P(θ)/P_fid]` (HF-only)
**fixed the gradient faithfulness**: ns → 0.07 gradient error (vs 0.69 with the
value target), and **8–9 / 11 params** produced a gate-passing equation:

```
dtau0 0.004  tau0 0.005  Ap 0.071  herei 0.060  heref 0.096
alphaq 0.121  omegamh2 0.123  hireionz 0.144  ns 0.07(niter40) / 0.26(niter30)
hub 0.92 FAIL   bhfeedback 0.55 FAIL (priored out)
```

So the *fractional-response* target pushes the search toward gradient-faithful
equations — the **generative** complement to the filter. But:
- The **focused production implementation diverged** from the spike: it anchored
  the multi-fidelity (LF+HF) target on the **LF** GP at fid, so HF rows became
  `log P_HF − log P_LF(fid)` (resolution offset baked in) → loss 1.999 vs the
  spike's 1e-5, and ns failed the gate again. Retreated cleanly; the plumbing is
  **stashed** (`git stash` "stage8 lever#1 log_ratio …") for a careful redo.
- **hub stays unfaithful** even with ratio-response (0.92) — only a direct
  derivative loss is likely to fix it.

## Conclusion → Stage 9 = Sobolev derivative loss

The cheap levers (filter) + ratio-response (indirect generative) are necessary
but insufficient for the hardest key params. The direct fix is a **Sobolev
training loss** that penalizes `[∂_θ logP_SR − ∂_θ logP_GP]²` during the PySR
search — optimizing the Fisher quantity itself. Architecture is unchanged
(per-1D refits, inputs θ/k/z/resolution, additive combine); only the PySR loss +
a target-gradient training column change. First task: a Julia
`eval_grad_tree_array` feasibility spike (the latent risk).

**Reusable from Stage 8:** the `derivative_gate` (GP/equation finite-diff
gradients + the faithfulness predicate) validates any Sobolev-trained equation;
the `aq` operator stays (pole-free is strictly better for derivatives).
