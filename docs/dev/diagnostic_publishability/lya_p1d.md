# Is this publishable AS A DIAGNOSTIC / methods paper? — Lyα P1D referee verdict

**Reviewer lens:** Lyα-forest P1D observer/analyst (eBOSS/DESI/KODIAQ-SQUAD/XQ-100), the audience
that actually puts emulators inside Fisher matrices and MCMC.
**Date:** 2026-06-09
**Question judged:** NOT "can PySR replace the GP?" (the authors correctly retracted that — the
σ_perfect_1D ≡ σ_GP identity is a forced Jacobian). The question judged here is the *reframed* one:
**is this publishable as a derivative-faithfulness DIAGNOSTIC / failure-modes / methods paper** —
"when and why do per-parameter symbolic equations fail to be slope-faithful enough for a Fisher
forecast, and what does a Sobolev loss fix?" Judged against the bar for an honest, narrow methods
contribution — not against the bar for a sims-validated production emulator.

Materials read: `README.md`, `HANDOFF.md`, `docs/PARETO_FAITHFULNESS_WALKTHROUGH.md` (source of
truth + Scope & reproducibility), the four figures in `results/single_z_stage_pareto_diag/`,
`results/h_basis_test/h_basis.json`, the `grad_faith_*.csv` sidecars (ns-budget, Ap, hub verified
by hand), the plain-language draft `PAPER_NARRATIVE.md`, and the prior general-referee verdict
`docs/lya_referee_verdict.md`.

---

## 1. Verdict

**YES — publishable as a diagnostic / methods paper, WITH MINIMAL ADDITIONS.**

This is a real, novel-in-this-literature question, executed with unusual honesty, and it matters to
the exact community that builds P1D emulators for inference. The prior referee said "major revisions"
and listed four "required before acceptance" asks — but that verdict was implicitly graded against a
*full emulator-validation* bar. **For a diagnostic paper specifically, three of those four asks are
NICE-TO-HAVEs, not blockers.** What blocks publication is much narrower: the headline taxonomy is
**single-seed**, and several load-bearing verdicts rest on **one equation** (Ap on one row, hub on a
literal one-row front — I confirmed both in the CSVs). A diagnostic that *names categories* must show
those categories are not seed-luck. That is a 1–2 week add, not a re-architecture.

**Venue / form.** This is a methods/diagnostic paper, best as:
- **OJA (Open Journal of Astrophysics)** or **JCAP** as a short methods paper — first choice. OJA in
  particular rewards exactly this kind of honest, reproducible, narrow-scope methods note, and the
  emulator-free figure reproducer is a strong fit for its ethos.
- Equally at home in **RASTI** (RAS Techniques & Instruments) or as a methods section spun out for a
  **NeurIPS/ICML ML-for-physical-sciences workshop** (the "value-fit ⇏ derivative-fit, and a Sobolev
  loss is the cure" story travels well there). For the Lyα-cosmology community specifically, OJA/JCAP.

The title and abstract must carry the scope in plain words: *"A derivative-faithfulness diagnostic for
symbolic-regression emulators of the Lyα P1D"*, single-z headline + cross-z appendix, GP-as-oracle.
With the abstract framed that way, no reader is misled, and the contribution stands on its own.

---

## 2. Minimal work required to publish — MUST-HAVE vs NICE-TO-HAVE

### MUST-HAVE (blocks publication of the diagnostic)

**M1 — Across-seed spread on the z=3.6 headline (the one true blocker).**
A paper whose *product is a taxonomy* (it sorts 11 parameters into four named boxes) cannot rest on a
single PySR draw, because PySR is stochastic and the boxes are defined by a 0.25 gate that several
parameters clear or miss by a hair. I verified the fragility directly in the CSVs:
- **Ap** — the entire "a faithful equation exists" (selection-sensitive) claim rests on **exactly one
  row**: complexity 4, grad_err 0.108, the *only* `gate_pass=True` in `grad_faith_Ap.csv`. Every other
  candidate is 0.26–0.91. That is one lucky low-complexity equation, not a property of the front.
- **hub** — `grad_faith_hub.csv` is **literally one row** (complexity 20, the max; the feature enters
  only there). A "resistant / weak-under-determined" *verdict* drawn from a one-point front is not
  publishable as a finding.
- **ns** (the marquee Sobolev save) clears the gate at 0.193 vs 0.25 — a 23% margin, one draw.

Re-run the z=3.6 value and Sobolev fronts for **≥5 seeds** (and the ns budget@35), report
grad_err **median ± spread per parameter**, and **re-label any box that flips across seeds**. This is
the single thing that converts "observed categories in one draw" into "a taxonomy." Effort: low — the
pipeline exists; it is N seeds × the existing run, ~1 week of wall-time + a spread column on two
figures. *This is the only hard blocker.*

**M2 — Demote/qualify the two single-equation verdicts, and state the gate's covariance meaning (presentation, not new compute).**
Two cheap honesty fixes that the prior referee is right about and that a diagnostic paper must do:
- **(a)** Present the taxonomy explicitly as **per-redshift / "observed categories at z=3.6"**, not as
  a redshift-stable four-box object — because the cross-z table already shows it is not (the He II
  block flips fail at z=4.2; bhfeedback flips *toward* pass at z=4.2). The cross-z appendix is a
  strength; it just must be allowed to qualify the taxonomy rather than be presented beside an
  unqualified one. No new runs.
- **(b)** Give the **0.25 gate a one-paragraph covariance justification**: propagate a uniform X%
  per-bin slope error through the *single-z* z=3.6 Fisher matrix with the real KODIAQ-SQUAD covariance
  and show the σ_θ inflation vs grad_err. The Fisher + covariance code already exists in this repo;
  this is a ~1-day plot that turns 0.25 from "asserted operating point" into "the slope error at which
  σ_θ inflates by ~X%." For a *diagnostic*, this is borderline must-have because the gate **is** the
  instrument — a reader needs to know what a given grad_err *costs* in σ. (I rank it must-have-lite:
  if seed-spread M1 ships and the gate is at least *bracketed* by a covariance argument, that suffices.
  A full derived-tolerance treatment is nice-to-have.)

That is the whole MUST-HAVE list: **one seed sweep + two presentation fixes.** Everything below is
optional.

### NICE-TO-HAVE (strengthens; does NOT block a diagnostic paper)

- **N1 — herei×alphaq coupling, shown not just stated.** Build the 2-param combined SR prediction and
  compare the 2×2 sub-Fisher (or ρ) against the GP's. This is the single most *interesting* addition
  for a Lyα audience (the He II degeneracy is what we care about), and it would convert the "diagonal
  Fisher only" limitation paragraph into a quantified scope statement. Strongly encouraged — but a
  diagnostic paper is allowed to declare "per-parameter ⇒ diagonal only; off-diagonal is out of scope"
  as a stated boundary, so it does not block.
- **N2 — Disambiguate the z=4.2 He II degradation** (report the GP's own z=4.2 slope magnitude + its
  emulator error bar beside grad_err). Lets "faithfulness tracks the physics" be supported rather than
  asserted. Appendix-level; the honest caveat already in the walkthrough is enough to ship.
- **N3 — A unit test for `derivative_faithful` + the masking logic** with a synthetic known-slope case.
  Cheap, and the load-bearing function currently has no direct test. Do it; it is an afternoon.
- **N4 — Off-fiducial grad_err at ±1σ** (see §3 — I actually rank this the top *scientific* upgrade,
  but it is not a blocker for the narrow claim).

### The prior referee's four "required" asks — must-have FOR A DIAGNOSTIC PAPER? (explicit ruling)

| Prior referee's "required" ask | Must-have for a *diagnostic* paper? | Why |
|---|---|---|
| **1. Sims-as-truth anchor** (GP ∂P/∂θ vs HF-sim finite difference) | **NO — nice-to-have.** | The paper's claim *is explicitly* "faithfulness **to the emulator**," and it says so. Scoring an SR distillation against the object it distilled (the GP) is the *correct and self-consistent* truth for a distillation-fidelity diagnostic. Requiring sims-as-truth holds the paper to the *emulator-validation* bar it deliberately renounced. One anchor (ns + a mean-flux param) would strengthen the "is the GP slope itself trustworthy" worry and I'd welcome it, but a diagnostic that honestly brackets its scope to the GP does not *need* it. **Not a blocker.** |
| **2. Across-seed spread** | **YES — must-have.** | This is the one ask that *is* a blocker, because the deliverable is a taxonomy and the categories are defined by a gate that single equations clear by luck (Ap one row, hub one row, ns by 23%). Without a seed band the boxes are not shown to be real. (= M1.) |
| **3. Off-fiducial validation** | **NO — nice-to-have (but the best science upgrade; see §3).** | A *Fisher* forecast is by definition a local (linearised) object; scoring the slope *at the fiducial* is legitimate for the Fisher-faithfulness question as posed. Off-fiducial is what tells you whether the equation is usable *beyond* linearisation — genuinely valuable, and it directly tests the k^θ pathology — but it extends the claim rather than securing the stated one. **Not a blocker; highest-value optional.** |
| **4. Covariance-grounded gate** | **PARTIAL — must-have-lite (= M2b).** | For a diagnostic the gate *is* the instrument, so the reader must know what grad_err costs in σ — but this is a ~1-day plot with code already in-repo, not a research program. Bracketing 0.25 against the KODIAQ-SQUAD covariance is required; a fully *derived* tolerance is nice-to-have. |

**Net:** of the prior referee's four "required before acceptance" items, **only across-seed is a true
blocker for a diagnostic paper**, with covariance-gate as a cheap must-have-lite. Sims-as-truth and
off-fiducial are nice-to-haves that the narrowed, honestly-scoped claim does not require. The prior
"major revisions" was the right call *for a sims-validated emulator paper* and the wrong bar for the
diagnostic the authors actually wrote.

---

## 3. The single highest-value next experiment

**Off-fiducial grad_err: re-score the slope at a handful of points away from the fiducial (±1σ along
each parameter axis, within the prior), for the z=3.6 fronts.**

Why this one (above even the seed sweep, *scientifically*): the entire diagnostic currently lives at a
single point in parameter space, and a real P1D inference is **not** local — the posterior wanders far
from the fiducial, especially for the under-constrained IGM-thermal parameters. An equation whose slope
is faithful *only* at the linearisation point is of limited use for an actual forecast, and the paper's
own **k^θ pathology** argument (∂_θ k^θ = k^θ ln k flips sign at k=1 and blows up at small k) *predicts*
that some "faithful at fid" equations will fail off-fiducial near the prior edges. This experiment:
- directly tests the paper's own most interesting mechanistic claim (the k^θ poison) instead of just
  asserting it;
- tells a real Lyα analyst the thing they actually need — "is this safe across my posterior, or only at
  the peak?" — which is what makes the diagnostic *useful to my community rather than a curiosity*;
- is cheap: the gate code already evaluates grad_err at an arbitrary θ; this is a loop over a few θ
  offsets using the *already-trained* equations and the GP. A few GPU/CPU-hours, no retraining.

If forced to ship only one new run, **M1 (seed sweep) is the publication blocker** and must come first;
but **off-fiducial is the experiment that most upgrades the paper from "honest internal diagnostic" to
"a tool the P1D community will actually reach for."** Do M1 to publish; do off-fiducial to be cited.

---

## 4. Does this diagnostic matter to people doing real P1D inference? (the lens, briefly)

Yes, and concretely. P1D forecasting (eBOSS DR14/16, DESI, XQ-100, KODIAQ-SQUAD) is now almost entirely
emulator-based, and the symbolic/analytic-surrogate literature (the syren family, arXiv:2506.08783)
reports surrogates on **value RMSE essentially exclusively**. "Is a value-accurate surrogate
*derivative*-accurate enough to put inside a Fisher matrix?" is exactly the question that literature
skips, and the answer here ("often no, and you can't tell from the value RMSE") is a genuine, useful
warning. The per-parameter physics story is credible to this reviewer:
- **ns** as the pivot-scale tilt whose value-optimal fit nails P's shape but mis-estimates ∂P/∂ns — and
  the budget control (every row 13→35 fails the gate; I confirmed) cleanly proving it is *generative*,
  not search-starvation — is the headline and it is the strongest, most convincing single result.
- **He II reion block (herei/heref/alphaq)** faithful at z≤3.6 and degrading at z=4.2 because the imprint
  is redshift-localised is physically right in *direction* (heref ends He II reion z≈2.6–3.2, herei
  starts z≈3.5–4.1) — but, correctly caveated by the authors, conflated with the GP being least accurate
  at z=4.2. State it as "the gate becomes unscoreable where the slope is small (physics or emulator)."
- **bhfeedback priored-out / near-noise gradient** — correct; ε_AGN is effectively priored out and the
  gate genuinely can't adjudicate a near-zero slope. Fine as stated.
- **h weak/under-determined** — the basis test refuting the AP/k-rescaling guess (corr ≈ −0.25, ~6% var,
  committed JSON) is a nice, reproducible, self-correcting loop and to the authors' credit. But it
  *refutes* AP; it does not *establish* "weak/under-determined" — and the one-row front (M1) means this
  verdict must be softened to "the feature barely enters and Sobolev cannot rescue a ~1% signal."

**Scope:** z=3.6-only WITH the cross-z appendix **is acceptable for a diagnostic paper** — provided the
taxonomy is labeled per-redshift (M2a). The taxonomy does *not* need to be all-z to publish; it needs to
*admit* it is not all-z, which the cross-z data already force. That honesty is, in fact, part of the
result ("faithfulness tracks the physics, redshift by redshift").

**What would make it useful vs a curiosity:** the off-fiducial run (§3) + the herei×alphaq coupling
demo (N1). Those two turn "here are failure modes at one point" into "here is what you must check before
trusting an SR slope in your forecast, across your posterior and across the one degeneracy you care
about." That is the difference between a methods note people cite and one they admire and forget.

---

## 5. Bottom line

A careful, self-critical, reproducible diagnostic that asks a real and skipped question. The σ-ratio
retraction was correct and is to the authors' credit. Publishable as an OJA/JCAP methods paper after
**one seed sweep** (M1) and **two presentation fixes** (per-redshift taxonomy + a covariance bracket on
the gate, M2). The prior referee's sims-as-truth and off-fiducial "requirements" are nice-to-haves for
*this* narrowly-scoped claim, not blockers. Top scientific upgrade: off-fiducial grad_err.
