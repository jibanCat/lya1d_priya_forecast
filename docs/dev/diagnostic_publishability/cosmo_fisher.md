# Referee report — publishability **as a diagnostic / methods paper**

**Referee lens:** Fisher-forecast / inference cosmology (P1D emulator faithfulness).
**Date:** 2026-06-09
**Question posed:** Is this work publishable *specifically* as a DIAGNOSTIC /
failure-modes / methods contribution — **not** as a claim that PySR replaces the GP
emulator? The σ_PySR/σ_GP forecast claim was deliberately **retracted** (forced
Jacobian identity, σ_perfect_1D ≡ σ_GP) and reframed: *when and why do per-parameter
symbolic equations fail to be derivative-faithful enough for a Fisher forecast, and
what does a Sobolev derivative-matching loss fix?* I judge it against the bar for an
**honest, narrow diagnostic**, not against a sims-validated emulator.

**Materials read:** `README.md`, `HANDOFF.md`,
`docs/PARETO_FAITHFULNESS_WALKTHROUGH.md` (source of truth, incl. Scope &
reproducibility), the four figures in `results/single_z_stage_pareto_diag/`
(`pareto_faithfulness`, `faithfulness_scorecard`, `ns_budget_panel`,
`crossz_faithfulness` — opened), `results/h_basis_test/h_basis.json`, the
`grad_faith_*.csv` sidecars (z=3.6 value/Sobolev, the budget@35 ns run, the z=2.6/4.2
retrains), the operator config (`configs/diagnostic.yaml`), `src/priya_forecast/fisher.py`,
the plain-language draft `PAPER_NARRATIVE.md`, and the prior general-referee verdict
`docs/lya_referee_verdict.md`.

---

## 1. Verdict

**Publishable as a diagnostic/methods paper: YES, with minimal additions
(`yes-with-minimal-additions`).**

This is a genuine, novel, and *useful* methodological contribution: the SR-emulator
literature (the syren family, arXiv:2506.08783) reports value RMSE almost
exclusively, and "does a value-accurate symbolic surrogate also reproduce the GP's
**slope** ∂P_F/∂θ — the only thing a Fisher matrix consumes?" is exactly the question
a forecaster needs answered before putting an SR emulator inside a Fisher matrix. The
answer here is non-trivial and behaviour-changing (see §3). The execution is unusually
honest: the σ-ratio retraction, the GP-as-oracle caveat, the single-seed/single-z
disclaimers, the "multi-z is a branch-name artifact" note, and the explicit admission
that Sobolev *worsens* the best-loss pick for heref/alphaq are all volunteered, and I
verified each against the CSVs and code. The budget control (ns) is clean and is the
single strongest result. The operator set is derivative-safe (`configs/diagnostic.yaml`:
binary `+−*/`, unary `log,exp` — no trig), which closes the prior referee's §73 worry.

**The crucial point for this verdict:** the prior general referee asked for
sims-as-truth, across-seed, off-fiducial, and a covariance-grounded gate "as required
before acceptance." That referee was (correctly) judging it as a *forecast/emulator*
paper. **For a diagnostic paper, that bar is wrong.** Three of those four asks are
NICE-TO-HAVE, not MUST-HAVE, *provided the scope is honestly stated* — which here it
already is. The one ask that survives as a near-must-have is **across-seed**, and only
because the headline currently leans on single equations that may not survive a seed
shuffle. See §4 for the ask-by-ask ruling.

### Venue / form

- **Best fit: a methods/diagnostics journal paper** — **OJA (Open Journal of
  Astrophysics)** or **JCAP**, explicitly titled and framed as a *failure-modes
  diagnostic* ("Derivative-faithfulness of symbolic-regression emulators for the Lyα
  P1D: a Fisher-forecast diagnostic"). OJA is the natural home given the methods-note
  character and the strong reproducibility story (emulator-free figure reproducer +
  committed sidecars).
- **Also viable: a NeurIPS/ICML "ML for the Physical Sciences" workshop paper** if the
  authors want a fast, low-bar venue — the "value-fit ⇏ derivative-fit, and here is a
  Sobolev cure with a budget control that rules out under-search" story is a clean
  workshop result *as-is*, single-seed and all.
- **Not** a full-length flagship cosmology result (MNRAS/PRD headline) — for that you
  would need the sims anchor and the joint-Fisher coupling result, which are out of
  scope for the diagnostic.

The single firm requirement for any of these: the **title and abstract must lead with
"diagnostic / faithfulness *to the emulator*"** and must NOT phrase anything as a
σ-forecast or an emulator-replacement. The repo already does this internally; the paper
`.tex` must match.

---

## 2. Does the Fisher-faithfulness framing matter for forecasting practice?

**Yes — decisively, and this is the core reason to publish.** A forecaster *would*
change behaviour:

1. **Selection rule.** The "selection-sensitive" category (Ap, herei, omegamh2) shows
   that picking the value-optimal PySR equation gives the Mirage, while a
   derivative-gated pick (or Sobolev training) recovers a faithful one *on the same
   front*. A practitioner reading this stops selecting symbolic emulators by value RMSE
   and adds a derivative gate — that is a concrete, actionable change.
2. **Objective.** The ns "generative Mirage" + budget control proves that for some
   parameters **no** value-trained equation at any searched complexity is faithful, and
   only a Sobolev (derivative-matching) objective fixes it. That tells a forecaster:
   for tilt-like responses, train the derivative, do not just search harder. This is the
   load-bearing methodological claim and it is well supported (budget@35: 20 candidates,
   complexity 13→35, **none** cross the 0.25 gate; Sobolev@18 reaches 0.193 — verified
   in `decider_budget_z3.6/.../grad_faith_ns.csv`).
3. **Operator hygiene.** The `k^θ` observation (∂_θ k^θ = k^θ ln k flips sign at k=1,
   explodes at small k — fine for value, poison for Fisher) is a genuinely citable
   "do not do this in SR-for-forecasting" rule. It should be promoted to a short methods
   paragraph (prior referee §72 — I concur).

So the framing is not a curiosity; it is a checklist a forecaster would adopt. That is
the bar for a methods paper and it is met.

---

## 3. Is the 0.25 gate defensible for a diagnostic?

**Yes, if *motivated* — it does NOT need to be covariance-derived for a diagnostic.**
The gate is an operating point, and the paper already (honestly) calls it "a chosen
operating point" rather than a derived constant. For a diagnostic whose job is to
*sort* parameters into categories, an order-of-magnitude-motivated threshold is
defensible: a 25%-per-bin slope error is plausibly sub-dominant to KODIAQ-SQUAD's
per-bin error, and the **taxonomy is robust to the exact threshold** for almost every
parameter (dtau0/tau0 at ~0.003–0.01 and hub/bhfeedback at ~0.9–1.7 are nowhere near
0.25; only ns, heref, omegamh2, hireionz-at-z2.6 sit close enough that the verdict
could flip). So the gate value is *not* load-bearing for the headline taxonomy — only
for the handful of near-gate parameters.

**Recommendation (cheap, high-value):** rather than leave 0.25 asserted, do the
one-line covariance propagation (§5) to *report what a given grad_err costs in σ_θ* —
this converts "chosen" into "motivated" without making the gate a hard derived
constant. The infrastructure is already in-repo (`src/priya_forecast/fisher.py`
already builds F, inverts to cov, and extracts σ against the KODIAQ-SQUAD covariance),
so this is hours, not weeks. I rank this NICE-TO-HAVE-bordering-MUST (see §4/§5) — it
is the cheapest single thing that upgrades the paper's rigor.

---

## 4. The prior referee's four asks — must-have vs nice-to-have *for a diagnostic*

The prior verdict (`docs/lya_referee_verdict.md`) listed sims-as-truth, across-seed,
off-fiducial, and a covariance gate as "required before acceptance." Re-ruled for a
**diagnostic** paper:

| Prior referee ask | For an *emulator/forecast* paper | **For a DIAGNOSTIC paper** | Why |
|---|---|---|---|
| **Sims-as-truth anchor** | MUST | **NICE-TO-HAVE** | The diagnostic is honestly framed as faithfulness *to the emulator*. "Does the SR equation reproduce the GP's slope?" is a well-posed, complete question that needs no simulation. Sims-as-truth would answer a *different* (and bigger) question — "is the GP slope itself right?" — which the paper explicitly does not claim. Requiring it conflates the two. It is a strong *future-work* / robustness addition, not a gate on this contribution. Staging paired HF PRIYA runs is also genuinely expensive. |
| **Across-seed spread** | MUST | **MUST-HAVE (the one that survives)** | PySR is stochastic and the headline currently leans on *single* equations. The Ap "selection-sensitive" claim rests entirely on **one** complexity-4 row (the only `gate_pass=True` in `grad_faith_Ap.csv`; every other candidate is 0.26–0.91). The hub "resistant" verdict rests on a **one-row** front (`grad_faith_hub.csv` is literally a single complexity-20 candidate). The ns Sobolev pass (0.193) clears the chosen gate by only 23%. Without a seed band a reader cannot tell which categories are real and which are one lucky/unlucky draw. This is the one ask that genuinely *blocks* the diagnostic, because the taxonomy IS the deliverable and its near-gate boxes are seed-fragile. |
| **Off-fiducial validation** | MUST | **NICE-TO-HAVE** | The gate is evaluated at the fiducial, and "is the slope faithful at the linearisation point?" is exactly the quantity a *single-z Fisher matrix at fiducial* uses — so the diagnostic is internally consistent. A non-local check (±1σ along each axis) strengthens the practical reach and would test the k^θ-pathology prediction, but for a diagnostic that explicitly scores the fiducial Fisher slice it is an enhancement, not a prerequisite. |
| **Covariance-grounded gate** | MUST | **NICE-TO-HAVE (but cheap → do it)** | A diagnostic may use a *motivated* operating point (§3); it need not derive the gate from the covariance. But because the infrastructure already exists (`fisher.py` + KODIAQ-SQUAD covariance in-repo), the cost/benefit is so favorable that I recommend doing the σ_θ(grad_err) propagation anyway. Reclassified as a strongly-encouraged NICE-TO-HAVE. |

**Net:** of the prior referee's four "required" items, **only across-seed is a
must-have for the diagnostic.** The other three are legitimately downgraded *because
the paper's scope is narrower and honestly stated* — and that honesty is the whole
justification for the downgrade. If the abstract ever drifts back toward an
emulator/forecast claim, three of these snap back to must-have.

---

## 5. Minimal set of additional work

### MUST-HAVE (blocks publication of the diagnostic)

1. **Across-seed spread (≥5 PySR seeds) on the z=3.6 value and Sobolev fronts; report
   grad_err mean±spread per parameter, and reclassify any parameter whose box flips.**
   This is the only true blocker. It directly de-risks the two most fragile claims —
   the Ap single-equation "selection-sensitive" verdict and the hub one-row "resistant"
   verdict — and puts an honest error bar on the marquee ns Sobolev pass (0.193 vs 0.25).
   Effort: moderate — it is re-running the existing refit pipeline with 5 seeds and
   re-scoring; no new code, the sidecar machinery already exists. This is days, not
   weeks. *If a category survives the seed band, the paper is solid; if Ap or hub flips,
   the honest fix is to restate those verdicts ("the feature barely enters and the
   front is unstable across seeds"), which is itself a publishable, honest result.*

2. **Demote "taxonomy" to "observed categories, per-redshift" OR explicitly label it
   per-z; carry the heref/alphaq asterisk in the table itself.** This is a framing /
   writing fix, near-zero effort, but it is a must because the cross-z figure already
   shows the four boxes are *not* redshift-stable (bhfeedback improves to 0.37 at z=4.2;
   the He II block blows up). Presenting a "taxonomy" as a stable object while your own
   figure refutes it is the kind of internal inconsistency a referee will reject on.
   Reframe as "at z=3.6 we observe four behaviours; §cross-z shows they are
   redshift-dependent, and that dependence tracks where each parameter informs the P1D."

3. **State plainly that a green taxonomy does NOT certify a faithful *joint* forecast**
   (the per-parameter construction certifies only the diagonal of the Fisher matrix;
   the herei×alphaq ρ≈+0.45 coupling is invisible by construction). The narrative
   handles this in prose; it must be an explicit *limitation*, not a footnote, because
   the paper's selling point is "is this safe for a forecast?" Writing-only, near-zero
   effort, but a must for honesty.

### NICE-TO-HAVE (strengthens, not required)

4. **Covariance-grounded σ_θ(grad_err) curve** — propagate a controlled per-bin slope
   error through the in-repo single-z Fisher (`fisher.py`) with the KODIAQ-SQUAD
   covariance and plot the σ_θ inflation vs grad_err. Cheap (infrastructure exists),
   converts the 0.25 gate from "asserted" to "motivated," and tells the reader what a
   given grad_err *costs*. Strongly encouraged precisely because it is so cheap.

5. **Off-fiducial grad_err** at ±1σ along a few axes (tests the k^θ pathology
   prediction and the practical reach of the slope). Moderate effort (needs GP). A good
   robustness section, not a gate.

6. **The herei×alphaq coupling, *shown*** — build the 2-parameter combined SR
   prediction and compare its 2×2 sub-Fisher / ρ against the GP's, quantifying the
   off-diagonal error the diagnostic is blind to. `scripts/run_coupling_matrix.py`
   already exists. This converts limitation §5(3) from prose into a number and is the
   result a Lyα audience will most want — but it is a *scope-quantification* bonus, not
   a prerequisite for the diagnostic.

7. **One sims-as-truth anchor** (GP ∂P/∂θ vs finite-difference of the actual PRIYA HF
   sims for ns + one mean-flux parameter) — the single most valuable *future-work*
   item, and the thing that would let a follow-up paper drop the "faithfulness to the
   emulator" qualifier. Genuinely expensive (paired HF runs); explicitly out of scope
   for this paper.

8. **Soften "Fisher's Mirage" branding** (borrowed from arXiv:2406.06067; the
   underlying fact — a function and its derivative are independently constrained — is
   elementary). Keep the demonstration, soften the marketing. Writing-only.

9. **A unit test for `derivative_faithful` + the masking logic** with a synthetic
   known-slope case. The two existing diagnostic tests are pure IO/plotting; the
   scientifically load-bearing function is untested. Cheap, good practice, not a
   publication gate.

---

## 6. The single highest-value next experiment

**Run the 5-seed (or more) re-fit of the z=3.6 value and Sobolev fronts and report
grad_err mean±spread per parameter, with category-flip flagged.**

This is the one experiment that simultaneously (a) discharges the only genuine
must-have, (b) directly tests the two weakest claims in the paper (Ap's
single-equation "selection-sensitive" box and hub's one-row "resistant" box), and
(c) puts an error bar on the headline ns Sobolev pass — all with the *existing*
pipeline and *no* new GP-truth or simulation machinery. It is the highest ratio of
"de-risks the publishable claim" to "effort," and it is the experiment a referee will
ask for first. Everything else (covariance curve, off-fiducial, coupling, sims anchor)
strengthens the paper but does not block it for a diagnostic framing.

If forced to name a *second*: the covariance-grounded σ_θ(grad_err) curve (§5.4),
purely because it is nearly free given `fisher.py` and it retires the "0.25 is
unmotivated" objection in one figure.

---

## 7. Bottom line

A careful, self-critical study asking a question the SR-emulator literature has
skipped, with a clean budget control and an honest scope. **Judged as a diagnostic /
failure-modes / methods paper — which is what it now claims to be — it is publishable
in OJA/JCAP (or a NeurIPS-ML4PS workshop) with one real experiment (an across-seed
band) plus three framing fixes (per-redshift taxonomy, diagonal-only-Fisher
limitation, branding).** The prior referee's sims-as-truth / off-fiducial /
covariance-gate "requirements" were calibrated to an emulator-replacement claim that
the authors have *correctly retracted*; for the diagnostic, those are nice-to-haves,
and only the across-seed ask survives as a blocker. The GP-as-oracle setup is honest
*because* the paper measures and claims only faithfulness-to-the-emulator. Make the
title/abstract carry that scope as loudly as the repo already does, add the seed band,
and this is a solid, honest methods contribution.
