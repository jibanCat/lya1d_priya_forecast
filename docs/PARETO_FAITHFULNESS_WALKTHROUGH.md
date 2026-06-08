# Per-parameter Pareto-faithfulness — walkthrough

**Status:** living document. Phase 1 (gray, value-loss layout) landed 2026-06-08;
the color (derivative-faithfulness) overlay and per-parameter verdicts land in
Phase 2 after the cluster gradient eval. Spec:
`docs/superpowers/specs/2026-06-08-pareto-faithfulness-diagnostic-design.md`.

## What this figure is, and why the paper turns on it

The paper is now a **diagnostic / failure-modes** contribution: *where, why, and
how badly per-parameter 1D symbolic regression (PySR) fails as a Fisher emulator
for the Lyman-α P1D, and what a Sobolev derivative-matching loss does and does
not fix.* This figure is the central instrument. It is modeled on the syren
Pareto-front plots (arXiv:2506.08783, Fig. A1) but adds the quantity the syren
family never reports: **derivative faithfulness** — whether an equation that
matches the GP's *values* also matches its *slopes* ∂P/∂θ, which is the only
thing a Fisher forecast actually uses.

**This redirect deliberately drops the σ_PySR/σ_GP forecast claim.** The 4-agent
review (2026-06-05) showed that σ_perfect_1D ≡ σ_GP is forced by construction
(the additive combine is anchored at P_GP(fid)), so the forecast "ladder" tests a
Jacobian, not an emulator, and the GP-slice fallback silently reports GP-derived
σ in the σ_PySR column. The derivative-faithfulness diagnostic below is what the
GP-as-oracle setup *can* legitimately support, and it is plotted with **no
GP-slice fallback** — a parameter with no faithful equation simply shows up
all-red.

## How to read the figure

![Per-parameter Pareto-faithfulness](../results/single_z_stage_pareto_diag/pareto_faithfulness.png)

- **One panel per parameter** (11 PRIYA parameters), single-z **z = 3.6**, on the
  real KODIAQ-SQUAD covariance.
- **x = complexity** (PySR equation node count); **y = value loss** (log scale).
  Each marker is one Pareto-optimal equation at that complexity.
- **Series** (marker shape): `value@20` = circles (value-loss target, maxsize 20,
  stage6_log); `Sobolev@20` = squares (Sobolev derivative loss λ=5, stage9). A
  third series `value@budget` (certified maxsize≈35) is added on the ns panel in
  Phase 3 to test the budget confound.
- **Marker color = `grad_err`** = `median_k |∂eq/∂θ ÷ ∂P_GP/∂θ − 1|` at the
  fiducial point over non-negligible k-bins — the *same metric the production
  derivative gate uses*. The colorbar is thresholded at the **0.25 gate**: green
  ≤ 0.25 (derivative-faithful) → red ≫ 0.25 (the "Fisher's Mirage": right value,
  wrong slope). `grad_err` is clipped at 1 for color only; the sidecar keeps the
  raw value.
- **Phase 1 caveat:** in the current figure **all markers are gray** — the
  gradient sidecars do not exist yet, so color is pending the cluster eval. The
  *value-loss geometry* (circles vs squares) is already meaningful and is read
  below.

## The mechanism the figure makes visible

PySR minimizes **value** mean-squared error. An equation can match P(θ) to high
accuracy yet have the wrong **slope** ∂P/∂θ at the fiducial point — and Fisher
sees only the slope. So *value-accurate ⇏ derivative-accurate*
(arXiv:2406.06067, "Fisher's Mirage"). In the finished figure this shows up as a
disagreement between a marker's **height** (value loss, low = good) and its
**color** (derivative error, green = good): a low, red marker is the Mirage. The
Sobolev loss adds `λ·‖∂_θ eq − ∂_θ logP_GP‖²` to the objective, trading a little
value loss (squares sit *above* circles) to pull the slope onto the GP's — i.e.
trading height for color.

## Per-parameter reading

> Phase-1 entries note only what the *value-loss* geometry already shows. The
> `grad_err`/color verdict (and the final category) is filled in Phase 2.

### dtau0
_Color pending (Phase 2)._ Value front: smooth, deep descent — looks easy on value.

### tau0
_Color pending (Phase 2)._ Value front descends cleanly; Sobolev tracks close.

### ns
_Color pending (Phase 2)._ Sobolev squares plateau visibly **above** value
circles → the derivative constraint costs value loss here. This is the headline
Mirage-cure case (Stage 9: grad_err 0.69 → 0.13 under Sobolev); the budget
control (Phase 3) tests whether deeper complexity alone, without Sobolev, ever
reaches the gate.

### Ap
_Color pending (Phase 2)._ Value front descends cleanly; expected easy.

### herei
_Color pending (Phase 2)._ Large value-loss gap between Sobolev and value fronts.
One half of the real **herei × alphaq** coupling (+0.45) that a per-param-1D +
additive combine cannot represent — expect this among the worst on color.

### heref
_Color pending (Phase 2)._ Similar to herei; Sobolev front sits well above value.

### alphaq
_Color pending (Phase 2)._ The other half of the herei × alphaq coupling — expect
poor faithfulness for the same structural reason.

### hub
_Color pending (Phase 2)._ **The key resister.** Two candidate causes to check on
the colored panel: (a) **under-search** — at what complexity does the parameter
feature `x0` first enter the equation? (review notes ~complexity 6, signal buried
under a resolution offset); (b) **wrong basis** — hub acts like a k-rescaling /
Alcock-Paczynski-like distortion, a coordinate transform of k that a per-param
native-k 1D ansatz cannot express. If hub stays red even under Sobolev at *all*
complexities, that is the basis argument, not merely starved search.

### omegamh2
_Color pending (Phase 2)._ Value front descends cleanly; expected tractable.

### hireionz
_Color pending (Phase 2)._ Deep value descent; an IGM-thermal parameter (needs
multi-z to be well-conditioned, but value-fit looks fine at z=3.6).

### bhfeedback
_Color pending (Phase 2)._ **The second resister.** Mechanism: weak / near-
degenerate gradient — bhfeedback is effectively priored out, so ∂P/∂bhfeedback is
tiny and `grad_err` is ill-conditioned. Expect the equation cannot lock onto the
signal regardless of complexity or loss.

## Failure-mode taxonomy

| parameter | category | mechanism | what Sobolev does |
|-----------|----------|-----------|-------------------|
| dtau0 | TBD (Phase 2) | TBD | TBD |
| tau0 | TBD (Phase 2) | TBD | TBD |
| ns | TBD (Phase 2) | pivot/tilt; value-accurate but wrong slope (Mirage) | TBD (expect: cured) |
| Ap | TBD (Phase 2) | TBD | TBD |
| herei | TBD (Phase 2) | herei×alphaq coupling unrepresentable by additive 1D | TBD |
| heref | TBD (Phase 2) | TBD | TBD |
| alphaq | TBD (Phase 2) | herei×alphaq coupling unrepresentable by additive 1D | TBD |
| hub | TBD (Phase 2) | under-search and/or k-rescaling (AP-like) basis | TBD (expect: resists) |
| omegamh2 | TBD (Phase 2) | TBD | TBD |
| hireionz | TBD (Phase 2) | TBD | TBD |
| bhfeedback | TBD (Phase 2) | weak/degenerate gradient (priored out) | TBD (expect: resists) |

**Categories** (to be assigned in Phase 2): `easy` (value drops, color goes green
at low complexity), `mirage-cured-by-Sobolev` (value-front red, Sobolev-front
green), `resistant` (red under every series at every complexity).
