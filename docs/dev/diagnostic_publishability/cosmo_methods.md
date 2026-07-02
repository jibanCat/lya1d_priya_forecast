# Referee report — publishability as a DIAGNOSTIC / methods paper

**Referee lens:** ML-for-cosmology + emulator-methods referee (JCAP / OJA / MNRAS /
NeurIPS-ML4PS). I am judging this **specifically against the bar for a diagnostic /
failure-modes / methods contribution** — *when and why do per-parameter symbolic
equations fail to be derivative-faithful enough for a Fisher forecast, and what does a
Sobolev derivative-matching loss fix* — **not** against the bar for a sims-validated
emulator that claims σ_PySR can replace σ_GP. The authors have **retracted** that
forecast claim (correctly — σ_perfect_1D ≡ σ_GP is a forced Jacobian identity), and I
hold them only to what the GP-as-oracle setup can honestly support.

**Materials read:** `README.md`, `HANDOFF.md`,
`docs/PARETO_FAITHFULNESS_WALKTHROUGH.md` (source of truth, incl. Scope &
reproducibility), the four figures in `results/single_z_stage_pareto_diag/`
(`pareto_faithfulness`, `faithfulness_scorecard`, `ns_budget_panel`,
`crossz_faithfulness` — all opened), `results/h_basis_test/h_basis.json`, the
`grad_faith_*.csv` sidecars (z=3.6 value/Sobolev, budget@35, z=2.6/4.2),
`src/priya_forecast/derivative_gate.py`, the prior general-referee verdict
`docs/lya_referee_verdict.md`, and the plain-language draft `PAPER_NARRATIVE.md`.

I independently re-derived the load-bearing numbers from the CSVs (see §A).

---

## 1. Verdict

**PUBLISHABLE AS A DIAGNOSTIC / METHODS PAPER — yes-with-minimal-additions.**

The core question — *is a value-accurate symbolic surrogate derivative-accurate, which
is the only thing a Fisher forecast consumes?* — is a genuine, novel methods
contribution. The syren family (arXiv:2506.08783) and CosmoPower validate emulators on
**value** RMSE essentially exclusively; neither reports whether the surrogate's
**slope** ∂P/∂θ matches the reference. Putting any such surrogate inside a Fisher
matrix silently assumes derivative faithfulness, and this paper is the first I know of
to (a) define an operating metric for it on the Lyα P1D, (b) demonstrate the
value≠slope decoupling literally (the "Fisher's Mirage"), and (c) offer a
**derivative-matching training objective (Sobolev) + a derivative-validation gate** as
the cure, with a clean budget control proving the cure is a *better objective*, not
*more search*. That last triplet — gate + Sobolev loss + budget control — is the real
extension over the syren/CosmoPower validation practice, and it is the part a
methods-skeptical referee will find most convincing.

**The single most important framing point:** a diagnostic paper is judged on whether its
diagnostic claims are *honest, reproducible, and correctly scoped* — NOT on whether the
surrogate is good enough to deploy. By that bar this work largely clears it already. The
internal honesty here is well above the field norm (the σ-retraction, the GP-as-oracle
caveat, the "multi-z is a branch-name artifact" note, the explicit disclosure that
Sobolev *worsens* the best-loss pick for heref/alphaq). I verified the load-bearing
numbers against the CSVs and they are accurate to the digit (§A).

**Venue.** In order of fit:
- **OJA (Open Journal of Astrophysics)** — best home. It is exactly the venue for an
  honest, narrow, reproducible methods/diagnostic note; the emulator-free figure
  reproducer and committed sidecars suit its open-science posture. The single-z /
  single-seed scope is acceptable there *if* framed as a diagnostic.
- **JCAP** — also appropriate, slightly higher bar; would want the seed band and the
  covariance-grounded gate (both cheap, §3) to avoid a "single demonstration" objection
  from a forecast-minded referee.
- **NeurIPS-ML4PS / a methods letter (RASTI / ML4PS workshop)** — the cleanest
  *immediate* home if the authors want a short, high-impact statement of the
  Mirage + Sobolev-gate idea without broadening physics scope. The budget control alone
  is a workshop-worthy result.
- **MNRAS** — possible but the weakest fit: an MNRAS referee is most likely to demand
  the sims anchor and full multi-z forecast, i.e. push it back toward the emulator-paper
  bar the authors are deliberately *not* claiming.

**Recommendation: OJA (or a ML4PS letter for the short form), with the two
must-haves in §3.** This is a real contribution and the budget control alone justifies
publication of a focused version.

---

## 2. Is the methods contribution genuine and novel? (the core judgment)

**Yes, on three counts, and I want to be specific because this is the crux.**

1. **Derivative faithfulness as a reported quantity is novel for SR/analytic emulators.**
   The syren-style Pareto plots (value loss vs complexity) are the direct prior art and
   they never color by, or report, slope error. The paper's central figure is modeled on
   them and adds exactly the missing axis (color = `grad_err`). This is not a
   solution-looking-for-a-problem: any forecast built on an SR emulator implicitly bets
   on derivative faithfulness, and nobody checks it.

2. **The Sobolev-loss + derivative-gate is a real extension, not a relabeling.** Sobolev
   training (penalizing ‖∂eq − ∂ref‖²) is known in the function-approximation and
   PINN literature, but applying it to *symbolic-regression emulator distillation* and
   tying it to a *Fisher-consistent validation gate* (the gate scores linear-P_F slopes;
   the loss trains normalized-log-P slopes; the `∂logP/∂θ = (∂_θP)/P` bridge connects
   them) is a genuine methods step. I checked `derivative_gate.py`: both the candidate
   and GP gradients are central differences of raw `predict` (linear P_F), so the gate is
   honestly a linear-slope ratio — the linear/log discipline is implemented correctly,
   not just claimed.

3. **The budget control is the strongest single result and it is clean.** I reproduced it
   from `decider_budget_z3.6/.../grad_faith_ns.csv`: the *entire* complexity-13→35 front
   has minimum grad_err 0.319 (best-loss at complexity 35 is grad_err 0.319 / value_mse
   3.82e-4 / FAIL), while Sobolev@18 reaches 0.193 / 4.74e-4 / PASS. The paired framing —
   deeper search reaches *lower* value error yet never crosses the gate; the cure is the
   *objective* — is exactly the kind of control that converts "PySR can't" from a
   could-be-under-search complaint into a supported claim. **This is the result I would
   build the paper around.** It is the cleanest demonstration that value-fit and
   derivative-fit are independent axes, and it pre-empts the obvious referee objection
   that the Mirage is just a starved search.

**What is NOT novel, and the paper already concedes this:** the additive per-parameter
distillation itself (prior/student work), and the bare observation that a function and
its derivative are independently constrained (elementary; the "Fisher's Mirage"
branding is borrowed from arXiv:2406.06067 and should be softened — keep the
demonstration, not the marketing). The contribution is the *empirical taxonomy of which
P1D parameters fail and why, plus the Sobolev-gate cure*, not the math of Sobolev
losses.

---

## 3. Minimal evidence bar for a diagnostic paper — MUST-HAVE vs NICE-TO-HAVE

The prior general referee (`docs/lya_referee_verdict.md`) listed four "required before
acceptance" asks: **(1) sims-as-truth anchor, (2) across-seed spread, (3) off-fiducial
validation, (4) covariance-grounded gate.** Those are the right asks **for an emulator
paper**. For a **diagnostic paper** the calculus is different, because the object under
test is *the symbolic equation's fidelity to the emulator*, not the emulator's fidelity
to the universe. I rule on each explicitly.

### MUST-HAVE (blocks publication of the diagnostic)

**M1 — Across-seed spread (prior ask #2). MUST-HAVE.**
This is the one prior-referee ask that is a genuine blocker *for a diagnostic paper*,
because the diagnostic's deliverable IS a per-parameter classification, and PySR is
stochastic. Right now the headline taxonomy and every near-gate verdict are **one
seed-42 draw**. The most exposed claims are precisely the load-bearing ones:
- The Ap "a faithful equation exists" claim rests on **exactly one** outlier — I
  confirmed `grad_faith_Ap.csv`: every candidate fails (grad_err 0.26–0.91) except a
  single complexity-4 row at 0.108. That is one lucky equation, not a robust property of
  the front, and "selection-sensitive" as a *category* lives or dies on whether it
  survives reseeding.
- ns Sobolev = 0.193 clears the *chosen* 0.25 gate by 23%; heref 0.206, omegamh2 0.198,
  hireionz@z2.6 0.324 (already a fail) are all within a seed-jitter of the line.
- **Required:** re-run the z=3.6 value and Sobolev fronts for ≥5 seeds, report
  grad_err mean±spread per parameter, and **reclassify any parameter whose category flips
  across seeds.** A taxonomy that is not seed-stable is not a taxonomy — it is a single
  observation. This is the minimum that makes the four-box claim a *finding* rather than
  an anecdote. Effort: low (re-run an existing pipeline 5× per parameter; embarrassingly
  parallel; no new code).

**M2 — Covariance-grounded gate (prior ask #4). MUST-HAVE (in its cheap form).**
The 0.25 gate is currently *asserted* ("a 25%-per-bin slope error is sub-dominant to the
KODIAQ-SQUAD covariance") but never demonstrated. For a diagnostic whose entire output
is pass/fail against that gate, an undefended threshold is the single easiest point for a
referee to reject on — every verdict in the paper inherits the arbitrariness of 0.25.
- **Required:** propagate a controlled per-bin slope error through the **single-z** Fisher
  matrix with the real KODIAQ-SQUAD covariance and show σ_θ bias vs grad_err — i.e. turn
  0.25 from an operating point into a *derived* tolerance ("a grad_err of X inflates σ_θ
  by Y%"). This is a single-z, single-script calculation; the Fisher machinery already
  exists in the repo. Effort: low. It is a must-have because it is *cheap* and it
  converts the paper's central instrument from asserted to grounded. (I am NOT requiring
  the full multi-z forecast — only that the gate be calibrated against the covariance the
  paper already uses.)

### NICE-TO-HAVE (strengthens; does NOT block a diagnostic paper)

**N1 — Sims-as-truth anchor (prior ask #1). NICE-TO-HAVE for a diagnostic, not a
blocker.** This is where I most diverge from the prior referee. For an *emulator* paper
the sims anchor is mandatory. For a *diagnostic* paper whose stated, honestly-disclosed
object is "faithfulness *to the emulator*," the GP-as-oracle is a legitimate scope choice
— the paper measures whether the SR equation reproduces the GP's slope, and that is a
well-posed, useful question on its own (it tells a syren-style practitioner whether they
can trust their symbolic fit's derivatives against the emulator they distilled from). The
GP-slope-noise confound is real but it is a **caveat to disclose, not a gap that
invalidates the diagnostic** — *provided* the paper does not claim the verdicts are
about the simulations. One anchor for one or two parameters (ns + a mean-flux param)
would materially strengthen the "resistant" and z=4.2 stories and I recommend it, but its
absence does not block a correctly-scoped diagnostic. **Make it explicit in the abstract
and conclusion that a green verdict certifies faithfulness to the GP, not to PRIYA's
sims.** If the authors keep any language implying physical truth (e.g. "faithfulness
*tracks the physics*" in the cross-z story — see N3), then the anchor escalates toward
must-have *for that specific claim*.

**N2 — Off-fiducial validation (prior ask #3). NICE-TO-HAVE for a diagnostic.** The gate
is evaluated only at the fiducial point. For a *forecast* (a non-local inference) this
matters; for a *diagnostic of derivative faithfulness*, the fiducial slope is the natural
and defensible evaluation point (it is what a Fisher matrix linearizes around). Scoring at
±1σ along a couple of axes would be a good robustness check and the k^θ-pathology argument
predicts trouble near prior edges — so it is worth doing — but a fiducial-only diagnostic
is internally complete. Not a blocker.

**N3 — Disambiguate the z=4.2 He II degradation. NICE-TO-HAVE, but tied to a framing
fix that IS required.** The cross-z "faithfulness tracks the physics" story conflates (i)
a genuinely small physical slope, (ii) the GP being least accurate at z=4.2 (~2% vs ~1%),
and (iii) SR failure. Separating these (report the GP's own z=4.2 slope magnitude + its
emulator error bar alongside grad_err) is nice-to-have. But the **framing fix is
required (zero-cost):** state the cross-z result as "*the gate becomes unscoreable where
the GP slope is small — whether from physics or emulator error*," with physics offered as
the likely-but-unproven driver. As written, "tracks the physics" overclaims and a Lyα
referee will catch it.

**N4 — herei×alphaq coupling, shown not just stated. NICE-TO-HAVE.** Building the 2-param
sub-Fisher and comparing ρ against the GP would convert the headline limitation (the
diagnostic is diagonal-only and blind to the +0.45 He II coupling) from a prose caveat
into a quantified scope statement. Strongly strengthens the paper for a Lyα audience but
is not required for the diagnostic to stand — *provided* the paper states plainly that a
green taxonomy does **not** imply a faithful joint forecast (it currently does, in prose).

**N5 — Harden the "resistant" category (prior ask #7). Partly a framing fix (required),
partly nice-to-have.** The required, zero-cost part: the hub verdict rests on a **one-row
front** (I confirmed `grad_faith_hub.csv` has exactly one Fisher-safe candidate,
complexity 20). A positive mechanistic claim ("weak/under-determined") cannot be drawn
from one point — restate as "the feature barely enters under value-search and Sobolev
cannot rescue a ~1% signal," and keep the h basis test as what it is: a clean refutation
of the AP guess (corr ≈ −0.25, ~6% variance — I verified `h_basis.json`), NOT a positive
establishment of the mechanism. Also: "resistant" is not a single category (an
unscoreable target like bhfeedback vs a possible expressivity wall) and it is
redshift-dependent (bhfeedback *improves* to 0.37 at z=4.2). Present the taxonomy
explicitly as **per-redshift / "observed categories at z=3.6"**, not as a stable
four-box object. These are framing fixes, required, and cost nothing. Widening the hub
search until the feature enters at lower complexity is the nice-to-have part.

### Summary table

| Prior referee ask | Emulator-paper bar | **Diagnostic-paper bar (my ruling)** |
|---|---|---|
| Sims-as-truth anchor | required | **NICE-TO-HAVE** (caveat suffices if scope is honest; escalates to must-have only for "tracks the physics") |
| Across-seed spread | required | **MUST-HAVE** (the deliverable is a classification; one seed ≠ a taxonomy) |
| Off-fiducial validation | required | **NICE-TO-HAVE** (fiducial slope is the natural eval point for a linearized diagnostic) |
| Covariance-grounded gate | required | **MUST-HAVE, cheap form** (single-z σ_θ-vs-grad_err calibration; the gate is the instrument) |

Plus two **zero-cost required framing fixes**: (F1) demote the taxonomy to per-redshift /
z=3.6-observed; (F2) soften "tracks the physics" and the hub mechanistic claim to what
one-point/emulator-limited evidence supports.

---

## 4. The single highest-value next experiment

**The across-seed spread on the z=3.6 value + Sobolev fronts (M1) — run it first.**

Rationale: it is the cheapest experiment (re-run an existing, parallelizable pipeline
~5× per parameter; no new code, no sims, no GP retraining) and it simultaneously (a)
converts the four-box taxonomy from one observation into a *finding*, (b) directly tests
the two most fragile claims — the single-equation Ap "selection-sensitive" verdict and
the ns Sobolev pass at 0.193 vs 0.25 — and (c) is the one prior-referee ask that a
diagnostic paper genuinely *cannot* skip. If, additionally, the seeds are run at the
budget-control setting for ns, the same job hardens the headline result (the budget
control) with a seed band at no extra design cost. Everything else (sims anchor,
off-fiducial, coupling sub-Fisher) is either deferrable to a follow-up or reducible to a
zero-cost framing change.

A close second, if a second job-slot is free, is **M2 (covariance-grounded gate)** —
also cheap, single-z, reuses the repo's Fisher code, and removes the easiest rejection
hook (the asserted 0.25 threshold). I would run M1 and M2 together; they are the entire
must-have list and both are low-effort.

---

## A. Numbers I independently verified (from the CSVs / JSON / code)

- **Budget control (ns), `decider_budget_z3.6/.../grad_faith_ns.csv`:** best-loss
  complexity 35 → grad_err **0.31892**, value_mse **3.82e-4**, gate **FAIL**; minimum
  grad_err over the entire complexity 13→35 front = **0.319** (never crosses 0.25).
  Matches the claim exactly.
- **Sobolev ns (from walkthrough/figure):** 0.193 / 4.74e-4 / PASS at complexity 18;
  4.74/3.82 = 1.24 → "~24% worse on value, not equal." The paired framing is honest.
- **Ap, `single_z_stage6_log/.../grad_faith_Ap.csv`:** entire front FAILS (grad_err
  0.262–0.909) **except one** row — complexity 4, grad_err **0.1075**, PASS. The
  "faithful equation exists" claim rests on a single low-complexity outlier. (→ M1.)
- **hub, `single_z_stage6_log/.../grad_faith_hub.csv`:** **exactly one** Fisher-safe row
  (complexity 20, grad_err 1.000, FAIL). A one-point front. (→ N5 framing fix.)
- **h basis test, `h_basis.json`:** corr(∂P/∂h, ∂P/∂lnk) ≈ −0.21/−0.25/−0.26 at
  z=2.6/3.6/4.2; AP template R² ≈ 6%. Refutes the AP guess; does not positively
  establish "weak/under-determined."
- **Gate code, `derivative_gate.py`:** candidate and GP gradients are both central
  differences of raw `predict` (linear P_F), median over bins with |target| ≥ 1e-3·max,
  all-masked → False ("can't adjudicate"). Implementation matches the documented metric.

---

## B. Bottom line

A genuine, novel, honestly-scoped methods/diagnostic contribution. The
value≠slope decoupling, the Sobolev-loss + Fisher-consistent gate, and especially the
budget control are publishable as they stand in a focused form. For a **diagnostic
paper**, only **two of the prior referee's four asks are true blockers** — across-seed
spread and a covariance-grounded gate — and **both are low-effort, single-z, no-new-code
calculations.** The sims anchor and off-fiducial checks are nice-to-haves *because the
paper correctly scopes itself to faithfulness-to-the-GP*, provided it (i) says so plainly
in the abstract, (ii) demotes the taxonomy to per-redshift, and (iii) softens the
"tracks the physics" and hub mechanistic claims. Add the seed band + gate calibration,
apply the zero-cost framing fixes, and this is a clean OJA / JCAP / ML4PS paper.
