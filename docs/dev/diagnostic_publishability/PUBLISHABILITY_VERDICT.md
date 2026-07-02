# PUBLISHABILITY VERDICT — meta-synthesis of the four-panel referee review

**Synthesizer:** meta-reviewer, reconciling four independent referee reports
(`cosmo_methods.md`, `cosmo_fisher.md`, `lya_p1d.md`, `lya_emulator.md`).
**Date:** 2026-06-09
**Object judged:** the work *as a diagnostic / failure-modes / methods paper* —
"when and why do per-parameter symbolic-regression equations fail to be
derivative-faithful enough for a Fisher forecast, and what does a Sobolev
derivative-matching loss fix?" — **not** as a claim that PySR replaces the GP
emulator (that σ_PySR/σ_GP claim was correctly retracted as a forced Jacobian
identity, σ_perfect_1D ≡ σ_GP).

---

## Executive summary (8 lines)

1. **Unanimous (4/4): YES, publishable as a diagnostic/methods paper, with minimal additions** — not "as-is," not "not-yet."
2. **Consensus venue: OJA (best fit) or JCAP; a NeurIPS-ML4PS / RASTI workshop letter is viable for the short form.** NOT a PRIYA/DESI cosmology-results venue.
3. **One true blocker, named by all four: an across-seed PySR spread (≥5 seeds) on the z=3.6 value + Sobolev fronts** — the taxonomy is the deliverable and it currently rests on single-seed, often single-equation draws.
4. **Verified fragility (all four, against the raw CSVs): Ap's "selection-sensitive" box rests on ONE complexity-4 row (grad_err 0.108); hub's "resistant" box is a literal one-row front; ns's marquee Sobolev pass clears the 0.25 gate by only 23%.**
5. **Second near-blocker (3/4 must-have, 1/4 must-have-lite): a covariance-grounded gate** — propagate per-bin slope error through the in-repo single-z KODIAQ-SQUAD Fisher to convert the asserted 0.25 into a derived σ_θ-vs-grad_err tolerance (~half a day to a day; code exists).
6. **Plus three zero-cost framing fixes (4/4): label the taxonomy per-redshift (the cross-z figure already shows it isn't z-stable); state plainly a green taxonomy does NOT certify the joint Fisher (diagonal-only; herei×alphaq ρ≈+0.45 invisible by construction); soften "tracks the physics," the hub mechanism, and the "Fisher's Mirage" branding.**
7. **The case FOR is strong: the derivative-faithfulness axis is genuinely unreported by syren/CosmoPower, the Sobolev-loss + Fisher-consistent gate + budget control is a real extension, and the ns budget control (entire complexity-13→35 front never crosses the gate while a smaller Sobolev search does) is a clean, verified headline.**
8. **Prior referee's four asks, re-ruled for a diagnostic: across-seed = MUST (4/4); covariance gate = MUST-cheap / must-have-lite (4/4 do-it); sims-as-truth = NICE (4/4, but lya_emulator flags the truth data is already in-repo and ranks it the top strengthening); off-fiducial = NICE (4/4, lya_p1d ranks it the top *scientific* upgrade). You can start writing now; gate submission on items 1–2 of the checklist.**

---

## 1. Headline verdict

**Publishable as a diagnostic / methods paper — `yes-with-minimal-additions`. Unanimous, 4/4.**
Target **OJA** (first choice across all four) or **JCAP**; a **NeurIPS-ML4PS / RASTI**
workshop letter is a clean fast home for the short form. The single firm requirement
shared by all four panelists: **the title and abstract must lead with "diagnostic /
faithfulness *to the emulator*"** and must never phrase anything as a σ-forecast or an
emulator replacement. The repo already enforces this scope internally; the `.tex` must
match it.

There is no dissent on the verdict, the venue, or the single blocker. The panel
disagrees only on the *ranking* of the highest-value next experiment (§5), and that
disagreement is about what makes the paper *cited* versus what makes it *publishable* —
not about whether it can be published.

---

## 2. The case for / the case against

### The case FOR (why it clears the diagnostic bar)

- **The question is genuine and novel for this literature.** All four agree: the
  syren family (arXiv:2506.08783) and CosmoPower validate symbolic/analytic emulators
  on **value** RMSE essentially exclusively; none report whether the surrogate's
  **slope** ∂P/∂θ — the only thing a Fisher matrix consumes — matches the reference.
  This paper is the first to (a) define an operating metric for derivative
  faithfulness on the Lyα P1D, (b) demonstrate the value≠slope decoupling literally,
  and (c) offer a Sobolev derivative-matching objective + a Fisher-consistent
  validation gate as the cure.
- **The methods extension is real, not a relabeling.** The triplet — derivative gate +
  Sobolev loss + budget control — is the genuine step beyond syren/CosmoPower
  validation practice. The gate is implemented honestly (cosmo_methods verified
  `derivative_gate.py`: both candidate and GP gradients are central differences of raw
  linear-P predict; the linear/log discipline is correct, not just claimed).
- **The budget control is the strongest single result and it survives inspection.**
  Verified independently by multiple panelists against
  `decider_budget_z3.6/.../grad_faith_ns.csv`: the *entire* complexity-13→35 ns front
  has minimum grad_err 0.319 and never crosses the 0.25 gate, while a smaller
  Sobolev@18 search reaches 0.193/PASS. This converts "PySR can't" from a
  could-be-under-search complaint into a supported claim — the cure is a *better
  objective*, not *more search*. cosmo_methods: "this is the result I would build the
  paper around."
- **Honesty is well above field norm and is itself load-bearing for the verdict.**
  All four cite the volunteered σ-retraction, the GP-as-oracle caveat, the
  single-seed/single-z disclaimers, the "multi-z is a branch-name artifact" note, and
  the explicit admission that Sobolev *worsens* the best-loss pick for heref/alphaq.
  Because the scope is honestly narrowed, three of the prior referee's four asks
  legitimately downgrade (§4).
- **GP-as-oracle is the correctly-scoped object of study, not a flaw** — provided it is
  stated plainly. The diagnostic measures whether the SR equation reproduces the GP's
  slope; that is a well-posed, complete question that needs no simulation. lya_emulator
  is explicit: "GP-as-oracle is not a flaw — it is the correctly-scoped object of study."
- **Operator hygiene closes the prior referee's trig worry.** cosmo_fisher verified
  `configs/diagnostic.yaml`: binary +−*/, unary log/exp, no trig; the k^θ pathology is
  documented and is itself a citable "do not do this in SR-for-forecasting" rule.

### The case AGAINST (what genuinely blocks it)

- **The taxonomy is single-seed, and its load-bearing boxes rest on single equations.**
  This is the *only* item all four call a true blocker. PySR is stochastic, the boxes
  are defined by a 0.25 gate, and several parameters clear/miss it by a hair. Verified
  by multiple panelists against the CSVs:
  - **Ap** "a faithful equation exists" / "selection-sensitive" rests on **exactly one**
    row — complexity 4, grad_err 0.108, the *only* `gate_pass=True` in
    `grad_faith_Ap.csv`; every other candidate is 0.26–0.91.
  - **hub** "resistant / weak-under-determined" rests on a **literal one-row front**
    (complexity 20, the max — the feature enters only there).
  - **ns** (the marquee Sobolev save) clears 0.25 at 0.193 — a 23% margin, one draw.
  A taxonomy that is not shown to be seed-stable is one observation, not a finding.
- **The 0.25 gate is asserted, not derived.** Every pass/fail verdict in the paper
  inherits the arbitrariness of 0.25. The taxonomy is robust to the *exact* threshold
  for most parameters (dtau0/tau0 ≈ 0.003–0.01 and hub/bhfeedback ≈ 0.9–1.7 are nowhere
  near it), but ns, heref, omegamh2, and hireionz@z2.6 sit close enough to flip — and
  the gate *is* the instrument, so a referee can reject on the undefended threshold.
- **Three framing inconsistencies that a referee will catch** (all zero-cost to fix):
  the taxonomy is sold as a stable four-box object while the cross-z figure refutes it
  (bhfeedback improves to 0.37 at z=4.2; the He II block blows up); a green taxonomy is
  implied (in prose) to certify a forecast when the per-parameter construction
  certifies only the *diagonal* of the Fisher (the herei×alphaq ρ≈+0.45 coupling is
  invisible by construction); and "faithfulness tracks the physics" at z=4.2 launders
  GP-as-oracle into a physics claim, conflating small physical slope, GP being least
  accurate (~2% vs ~1%) at z=4.2, and SR failure.

**Net:** nothing in the "against" column is a re-architecture or a new research
program. The blocker is one parallelizable re-run of an existing pipeline; the
near-blocker is a half-day plot reusing in-repo Fisher code; the rest is writing.

---

## 3. Minimal publication checklist (deduplicated, prioritised)

| # | Item | Must / Nice | Flagged by | Rough effort |
|---|---|---|---|---|
| 1 | **Across-seed spread (≥5 PySR seeds) on z=3.6 value + Sobolev fronts** (incl. ns budget@35); report grad_err median±spread per parameter; **reclassify / demote any box that flips** | **MUST (the one true blocker)** | all 4 | Low–moderate: re-run existing pipeline ×5, no new code, embarrassingly parallel; ~1 cluster-day to ~1 week wall-time |
| 2 | **Covariance-grounded gate** — propagate per-bin slope error through the single-z KODIAQ-SQUAD Fisher (`fisher.py`, covariance in-repo); plot σ_θ inflation vs grad_err; turn 0.25 into a derived tolerance | **MUST (cheap)** — 3/4 must-have, lya_p1d must-have-lite; 4/4 say "do it" | all 4 | Low: ~half a day to 1 day, infrastructure exists |
| 3 | **Demote taxonomy to per-redshift / "observed categories at z=3.6"** (cross-z figure already shows it isn't z-stable); carry the heref/alphaq Sobolev-asterisk *in the table* | **MUST (framing, ~0 effort)** | all 4 | Writing only |
| 4 | **State plainly: a green taxonomy does NOT certify the joint Fisher** (diagonal-only; herei×alphaq ρ≈+0.45 invisible by construction) — as an explicit limitation, not a footnote | **MUST (framing, ~0 effort)** | cosmo_fisher, lya_p1d, lya_emulator (cosmo_methods via N4) | Writing only |
| 5 | **Soften overclaims:** "tracks the physics" → "the gate is unscoreable where the slope is small (physics or emulator)"; hub mechanism → "feature barely enters; Sobolev can't rescue a ~1% signal"; soften "Fisher's Mirage" branding (borrowed from arXiv:2406.06067) | **MUST (framing, ~0 effort)** | all 4 | Writing only |
| 6 | **Sims-as-truth anchor for ns + one mean-flux param** — local-linear / nearest-design-point ∂P/∂θ vs GP slope vs SR slope | **NICE** (but lya_emulator: truth data already in-repo — `mf_emulator_flux_vectors...hdf5` 600 design points + LOO set — so it's days, and the top strengthening) | all 4 | Moderate: days (data in hand); escalates to MUST only if any "tracks the physics" claim is retained |
| 7 | **Off-fiducial grad_err at ±1σ** along a few axes, using already-trained equations + GP; tests the k^θ pathology | **NICE** (lya_p1d ranks it the top *scientific* upgrade — "do M1 to publish; do this to be cited") | all 4 | Moderate: a few CPU/GPU-hours, no retraining |
| 8 | **herei×alphaq coupling, *shown*** — build the 2-param combined SR prediction, compare 2×2 sub-Fisher / ρ vs GP (`run_coupling_matrix.py` / `scripts/run_coupling_matrix.py` exists) | **NICE** (most-wanted by a Lyα audience; quantifies limitation #4) | all 4 | Moderate |
| 9 | **Unit test for `derivative_faithful` + masking** with a synthetic known-slope case (the load-bearing function is currently untested) | **NICE** (good practice) | cosmo_fisher, lya_p1d, lya_emulator | Trivial: an afternoon |
| 10 | **Promote the k^θ pathology to a short methods paragraph** ("∂_θ k^θ = k^θ ln k flips sign at k=1, explodes at small k — fine for value, poison for Fisher") + state the operator basis is derivative-safe | **NICE (citable)** | cosmo_fisher, lya_emulator | Writing only |

### Verdict on EACH of the prior referee's four asks (the explicit ask-by-ask ruling)

| Prior referee ask | Emulator-paper bar | **Diagnostic-paper bar (panel consensus)** | Reasoning |
|---|---|---|---|
| **Sims-as-truth anchor** | required / MUST | **NICE-TO-HAVE (4/4)** | The diagnostic is honestly scoped to faithfulness *to the emulator*; "does the SR equation reproduce the GP's slope?" is well-posed and needs no sim. Requiring sims conflates it with the bigger, different question "is the GP slope itself right?" — which the paper explicitly does not claim. lya_emulator sharpens this: the truth data (600 PRIYA design points + LOO set) is **already in the repo**, collapsing the cost from a resim campaign to days, so it is the recommended top *strengthening* — but still not a gate. **Caveat that escalates it to MUST for one specific claim:** if any "tracks the physics" language is kept, the anchor becomes required *for that sentence* (cosmo_methods, lya_emulator). |
| **Across-seed spread** | required / MUST | **MUST-HAVE — the one ask that survives (4/4 unanimous)** | The deliverable *is* a per-parameter classification and PySR is stochastic. The headline boxes rest on single equations (Ap one row, hub one row) and near-gate passes (ns +23%). One seed ≠ a taxonomy. This is the sole genuine blocker. |
| **Off-fiducial validation** | required / MUST | **NICE-TO-HAVE (4/4)** | A Fisher matrix is by definition local; scoring the slope *at the fiducial* is the correct evaluation point for the linearised question as posed, so the diagnostic is internally complete without it. It is a real strengthening (lya_p1d ranks it the highest-value *scientific* upgrade and notes it directly tests the k^θ pathology), but it extends the claim rather than securing the stated one. State as a limitation if not done. |
| **Covariance-grounded gate** | required / MUST | **MUST-HAVE in its cheap form (3/4 MUST, lya_p1d must-have-lite; 4/4 "do it")** | A diagnostic may use a *motivated* operating point rather than a hard-derived constant — but the gate is the instrument every verdict inherits, and the σ_θ(grad_err) propagation is ~half a day with `fisher.py` + the KODIAQ-SQUAD covariance already in-repo. The cost/benefit is so favorable that the panel treats it as a near-free must: it converts the central instrument from "asserted" to "grounded." A *fully derived* tolerance is nice-to-have; *bracketing* 0.25 against the covariance is required. |

**Net (panel):** of the prior referee's four "required before acceptance" items, **only
across-seed is a true blocker for a diagnostic**, with the covariance gate as a cheap
must-do; sims-as-truth and off-fiducial are nice-to-haves *because the paper correctly
and honestly narrows its scope to faithfulness-to-the-emulator.* The prior "major
revisions" was the right call for a *sims-validated emulator* paper and the wrong bar
for the diagnostic the authors actually wrote — it is really "minor-to-moderate
revisions": two cheap must-dos plus framing, all inside a week.

---

## 4. Recommended target venue + framing

**Venue:** **Open Journal of Astrophysics (OJA)** — first choice of all four panelists.
It is the cultural home for an honest, narrow, reproducible methods/diagnostic note;
the emulator-free figure reproducer (`make_diagnostic_figs.py` from tracked CSV
sidecars) fits its open-science ethos exactly, and the single-z / single-seed scope is
acceptable there *once the seed band ships*. **JCAP** is equally appropriate at a
slightly higher bar (it will want the seed band and the covariance bracket — both on the
must-list anyway). A **NeurIPS/ICML ML4PS workshop** or **RASTI** is the clean *fast*
home for the short form: "value-fit ⇏ derivative-fit, here is a Sobolev cure with a
budget control that rules out under-search" travels well there essentially as-is.
**Do NOT** target a PRIYA/DESI cosmology-results venue or an MNRAS/PRD flagship — those
will (correctly) demand the sims anchor and full multi-z joint forecast, pushing the
work back toward the emulator-replacement bar the authors deliberately renounced.

**Title (panel-favored shape):** *"Derivative faithfulness of symbolic-regression
emulators for the Lyα P1D: when a value-accurate surrogate has the wrong Fisher slope,
and a Sobolev cure."* The words **"diagnostic"** and **"faithfulness to the emulator"**
must appear in the abstract.

**Abstract angle (the non-negotiable scope sentence):** lead with the missing axis
(SR/analytic emulators are validated on value RMSE; nobody reports the slope a Fisher
matrix actually consumes); state the GP-as-oracle scope in plain words (a green verdict
certifies faithfulness *to the GP*, not to PRIYA's simulations, and certifies only the
*diagonal* of the Fisher); headline the budget control (no value-trained equation at any
searched complexity is faithful for ns; only a Sobolev objective fixes it); present the
result as **per-redshift observed categories at z=3.6 with a cross-z appendix**, not a
redshift-stable taxonomy. Never phrase anything as a σ-forecast or emulator replacement.

---

## 5. The single most important next experiment

**Run the across-seed spread (≥5 PySR seeds) on the z=3.6 value + Sobolev fronts
(including ns budget@35); report grad_err median±spread per parameter and flag every
category flip.** This is the panel's decisive recommendation for *what unblocks
publication*: it is the only item all four call a true blocker, it is the cheapest
experiment (re-run an existing, parallelizable pipeline ~5×; no new code, no sims, no GP
retraining), and it simultaneously (a) converts the four-box taxonomy from one
observation into a finding, (b) directly de-risks the two most fragile claims — the
single-equation Ap "selection-sensitive" box and the one-row hub "resistant" box — and
(c) puts an honest error bar on the marquee ns Sobolev pass (0.193 vs 0.25). If a
category survives the band, the paper is solid; if Ap or hub flips, the honest restatement
("the feature barely enters and the front is unstable across seeds") is itself
publishable. **Pair it with the covariance-grounded gate (checklist #2)** — also cheap,
single-z, reuses `fisher.py` — and you have discharged the entire must-do list.

**Note on the panel's internal split (for honesty, not indecision):** three panelists
rank the seed sweep as *the* next experiment outright; lya_p1d and lya_emulator each name
a different #1 — off-fiducial grad_err (lya_p1d: the top *scientific* upgrade, what turns
the paper from "honest internal diagnostic" into "a tool the community reaches for") and
the in-repo sims-truth anchor (lya_emulator: highest confidence-per-effort, the only
addition that attacks the GP-as-oracle limitation, with the data already in the tree).
The reconciliation is clean and unanimous when phrased as a sequence: **do the seed sweep
to *publish*; do off-fiducial and/or the sims anchor to *be cited*.** Start with the seed
sweep + covariance gate now — you can begin writing in parallel.

---

## Bottom line

A genuine, novel, honestly-scoped methods/diagnostic contribution with a clean, verified
headline (the ns budget control) and field-leading internal honesty (the σ-retraction
*enables* this paper). **You can begin writing it up now.** Submission is gated on exactly
**two cheap experiments — an across-seed band (the one real blocker) and a
covariance-grounded gate — plus three zero-cost framing fixes** (per-redshift taxonomy,
diagonal-only-Fisher limitation, softened "tracks the physics"/hub/branding). Re-scored to
the diagnostic bar, the prior referee's "major revisions" is minor-to-moderate: of its four
asks, only across-seed truly blocks; the covariance gate is a near-free must; sims-as-truth
and off-fiducial are nice-to-haves the honestly-narrowed claim does not require. Aim at OJA
(or JCAP), lead the title and abstract with "diagnostic / faithfulness to the emulator,"
and this is a solid, citable paper.
