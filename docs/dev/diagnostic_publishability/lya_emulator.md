# Is this publishable as a DIAGNOSTIC / methods paper?

**Reviewer lens:** Lyα / P1D emulator expert (PRIYA-adjacent), judging the work
**specifically as a diagnostic / failure-modes / methods contribution** — NOT as a
claim that PySR replaces the GP emulator. The σ_PySR/σ_GP forecast claim has been
correctly retracted (it was a forced Jacobian identity, σ_perfect_1D ≡ σ_GP); this
review judges only what remains: *when and why do per-parameter symbolic equations
fail to be derivative-faithful enough for a Fisher forecast, and what does a Sobolev
derivative-matching loss fix?*

Materials read: `README.md`, `HANDOFF.md`, `docs/PARETO_FAITHFULNESS_WALKTHROUGH.md`
(source of truth), the four figures (opened), `results/h_basis_test/h_basis.json`,
the `grad_faith_*.csv` sidecars (ns budget@35, hub value@20 + Sobolev, Ap value@20),
the plain-language draft `PAPER_NARRATIVE.md`, and the prior general-referee verdict
`docs/lya_referee_verdict.md`. I also verified the headline numbers against the CSVs
and inspected what sim-truth data is actually reachable in this tree.

---

## 1. Verdict

**YES — publishable as a diagnostic / methods paper, WITH minimal additions.**

Not "not-yet," and not "yes as-is." The core contribution is real, novel for this
literature, unusually honest, and the load-bearing result (the ns budget control) is
clean and survives inspection of the raw CSV. The prior referee said "major
revisions" — but that verdict was written against an *emulator-replacement* bar
(it literally asks for sims-as-truth, across-seed, off-fiducial, and a
covariance-grounded gate "before acceptance"). Re-scoped to a **diagnostic** bar,
most of those collapse from MUST-HAVE to NICE-TO-HAVE. Two things remain genuine
blockers, and both are days of work, not a year.

**Venue / form.** A methods / short paper, not a flagship emulator paper:

- **Best fit: JCAP or The Open Journal of Astrophysics (OJA), as a methods paper.**
  OJA in particular is the right cultural home — it rewards exactly this kind of
  honest, reproducible, narrow methodological note, and the emulator-free figure
  reproducer (`make_diagnostic_figs.py` from tracked CSVs) is a strong fit for OJA's
  reproducibility ethos.
- **Also viable: the NeurIPS/ICML "ML for the Physical Sciences" / ML4PS workshop
  track**, or *RAS Techniques and Instruments (RASTI)*, framed as "derivative
  faithfulness is the missing axis in symbolic-regression emulators." The
  CS/ML-facing framing (value-fit ⇏ derivative-fit, Sobolev cure, budget control)
  travels well there and is lower-friction than a full journal.
- **Do NOT** target a PRIYA/DESI cosmology-results venue or frame it as a P1D
  forecasting result — the GP-as-oracle scope (below) caps that, and the σ-retraction
  already conceded it.

**Title should carry the scope in it**, e.g. *"Derivative faithfulness of symbolic
emulators: when a value-accurate Lyα P1D surrogate has the wrong Fisher slope, and a
Sobolev cure."* The word "diagnostic" and "to the emulator" should be in the abstract.

---

## 2. The minimal set of additional work — MUST-HAVE vs NICE-TO-HAVE

### The prior referee's four "required" asks, re-scored FOR A DIAGNOSTIC PAPER

| Prior ask | For an *emulator-replacement* claim | **For a DIAGNOSTIC paper** | Why |
|---|---|---|---|
| **Sims-as-truth anchor** | MUST | **NICE-TO-HAVE (one cheap sanity check makes it MUST-strengthening, see §3)** | A diagnostic that *measures faithfulness to the emulator* does not logically require the sims, IF that scope is stated plainly. But see the sharpened version below — there is a cheap, in-repo version that is worth promoting to "do it." |
| **Across-seed spread** | MUST | **MUST-HAVE** | This is the one that genuinely blocks the diagnostic. The taxonomy is the product, and three of its four boxes (selection-sensitive, the near-gate passers, the cross-z reshuffles) currently rest on single equations from a single seed. A diagnostic whose *categories flip across seeds* is not a diagnostic. |
| **Off-fiducial validation** | MUST | **NICE-TO-HAVE** | The paper can honestly scope itself to the linearisation point: a Fisher matrix *is* evaluated at the fiducial, so a fiducial-point gate is the right object for the stated question. Off-fiducial is a real strengthening (and the k^θ argument predicts trouble) but is not load-bearing for the diagnostic claim. State it as a limitation. |
| **Covariance-grounded gate (derive 0.25)** | MUST | **MUST-HAVE (but cheap)** | The gate is the diagnostic's only free knob, and every "pass/fail" verdict — the entire taxonomy — is defined relative to it. "25% is sub-dominant to the covariance" is currently *asserted*, never shown. This is ~half a day with the KODIAQ-SQUAD covariance already in the repo, and it converts the gate from an arbitrary line into a derived tolerance. Cheap enough that there is no excuse to ship without it. |

### Consolidated list

**MUST-HAVE (blocks publication of the diagnostic):**

1. **Across-seed spread on the z=3.6 fronts (≥5 PySR seeds).** Re-run the value@20
   and Sobolev@20 fronts for ≥5 seeds and report `grad_err` as mean ± spread per
   parameter, and — critically — report **how often each parameter lands in its
   assigned taxonomy box.** This is the blocker because the data already show the
   fragility: I confirmed from the CSVs that the **Ap "selection-sensitive" verdict
   rests on a single complexity-4 equation** (grad_err 0.108, the only pass on a
   front where every other row is 0.26–0.91), and the **hub value@20 front is
   literally one row** (complexity 20, the max). One lucky equation and a one-point
   front cannot anchor a taxonomy box. Reclassify anything that flips; demote
   single-equation boxes to "indicative." Effort: re-running existing drivers with 5
   seeds — a cluster day or two, no new code.

2. **Quantify the 0.25 gate against the KODIAQ-SQUAD covariance.** Propagate a
   controlled per-bin slope error through the single-z Fisher matrix (covariance
   already in `data/kodiaq_gp/`) and plot σ_θ inflation vs `grad_err`. This turns
   0.25 from an asserted operating point into a derived tolerance and tells the
   reader what a given `grad_err` *costs* in σ. Half a day; it is the difference
   between "we chose 0.25" and "0.25 corresponds to a <X% σ bias." Without it the
   central metric is ungrounded and a referee will (rightly) not let the taxonomy
   stand on an arbitrary line.

**NICE-TO-HAVE (strengthens, does not block):**

3. **Sims-as-truth anchor for 1–2 parameters** (ns + one mean-flux). NICE-TO-HAVE in
   principle — but see §3, because in *this* repo it is unexpectedly cheap and I
   recommend doing it anyway. It is the single thing that most raises confidence.

4. **Off-fiducial re-scoring at ±1σ** for 2–3 parameters. Tests whether a
   fiducial-faithful slope stays faithful into the prior; the k^θ pathology predicts
   it may not. Scope as a limitation if not done.

5. **Disambiguate the z=4.2 He II degradation** — report the GP's own z=4.2 slope
   magnitude and its ~2% emulator error bar alongside `grad_err`, so "faithfulness
   tracks the physics" is supported rather than conflated with the GP being weakest
   exactly there. (Currently the He II block "blow-up" at z=4.2 conflates small
   physical slope, unreliable GP slope, and SR failure.)

6. **The herei×alphaq coupling, shown not just stated.** Build the 2-parameter
   combined prediction and compare the 2×2 sub-Fisher (or ρ) against the GP's. This
   is the one a Lyα audience most wants, and it converts the "diagonal-only"
   limitation into a quantified scope statement. Worth it but genuinely optional for
   a diagnostic.

7. **A unit test for `derivative_faithful` / the masking logic** with a synthetic
   known-slope case. The two committed tests are pure plotting/IO; the
   scientifically load-bearing function is untested. Trivial, do it.

**Presentation fixes (free, do before submission):**

- Carry the heref/alphaq asterisk *in the taxonomy table*, not buried in a footnote
  (Sobolev's best-loss pick rises 0.154→0.206 and 0.152→0.173; they stay green only
  via best-faith).
- Either present the taxonomy explicitly **per-redshift**, or demote it to "observed
  categories at z=3.6" — bhfeedback flips toward pass at z=4.2 while the He II block
  flips to fail, so the four boxes are not a redshift-stable object and should not be
  sold as one.
- State plainly in the conclusions that **a green taxonomy does NOT imply a faithful
  joint forecast** (diagonal Fisher only; the off-diagonal herei×alphaq error is
  invisible by construction).
- Soften the hub mechanism to "the feature barely enters under value-search and
  Sobolev cannot rescue a ~1% signal" — the h-basis test *refutes* AP, it does not
  *establish* "weak/under-determined" (see §4). Confirm and state the operator basis
  is derivative-safe (no trig, k^θ dropped) as a short methods paragraph — this is a
  genuinely citable point and currently an aside.

---

## 3. The single highest-value next experiment

**A sims-as-truth slope anchor for ns + one mean-flux parameter, built from the GP
training flux vectors already in the repo.**

This is #1 because it is the *only* addition that attacks the framework's deepest
limitation — GP-as-oracle — and because I found that **the sim-truth data needed is
already present in this tree**, which collapses the prior referee's most expensive
ask from a year-long resim campaign to a few days:

- `data/kodiaq_gp/mf_emulator_flux_vectors_tau1000000.hdf5` holds
  `flux_vectors (600, 2236)` with `params (600, 10)` — the **600 PRIYA design-point
  simulation P1D measurements** the GP was actually trained on (13 z-bins, ~172
  k-bins). These are the sims, not GP output.
- `lya_emulator_full/dtau-48-48/loo_fps.hdf5` (and `loo_t0.hdf5`) are the
  **leave-one-out** flux-power products — the emulator's own accuracy-validation set.
- (Note: the `data/single_z_1pvar/hf_*_npoints50.hdf5` 1pvar files are **GP-sampled,
  not sims** — `regen_1pvar.py` reads them from `gp.predict`. So the *training*
  pipeline is GP-as-oracle end to end; the only sim truth in the tree is the 600
  design points and the LOO set. This is exactly why the anchor matters.)

**The experiment:** for ns and a mean-flux parameter, estimate the *simulation*
∂P_F/∂θ at (or near) the fiducial — by a local linear fit of the 600 design-point
flux vectors projected onto that parameter axis, or a nearest-design-point finite
difference — and compare it to (a) the GP posterior-mean slope the gate trusts and
(b) the symbolic equation's slope. Three numbers, one plot.

**What it buys, decisively:** it tells the reader whether `grad_err` measures *SR
failure* or *GP slope noise*. If the GP slope tracks the sim slope for these
parameters, the whole GP-as-oracle framework is vindicated as a proxy and every
"Mirage"/"resistant" verdict stops being ambiguous. If it does not, that is itself a
publishable and important finding (the GP's own derivative is the weak link, which
reframes the paper). Either outcome is a win, and it directly answers the prior
referee's single biggest objection (§4.1 of `lya_referee_verdict.md`) without
touching a simulation queue.

If forced to rank: **(1) the sims-truth anchor** (highest confidence-per-effort,
data in hand), then **(2) across-seed spread** (the formal blocker), then **(3) the
covariance-grounded gate** (cheapest, grounds the metric). All three are ≤ a week
combined.

---

## 4. Answering the lens questions directly

**Is GP-as-oracle acceptable for a diagnostic, if stated plainly?**
**Yes.** The diagnostic's claim is literally "does a symbolic equation reproduce the
GP's slopes" — the GP is the *definition* of the target, not an approximation to
something else. As long as the abstract and conclusions say "faithfulness *to the
emulator*, not to the simulations" (the walkthrough already does, in its Scope
section), GP-as-oracle is not a flaw — it is the correctly-scoped object of study.
The σ-retraction already conceded the thing that GP-as-oracle *cannot* support (a
real σ forecast). What remains is internally consistent. The one caveat: the paper
must not let GP-as-oracle silently launder into a physics claim — the z=4.2 He II
"tracks the physics" line is the one place it currently does, and must be reframed.

**Is a sims-as-truth anchor a MUST-HAVE for a diagnostic, or only for
emulator-replacement?**
**Only for emulator-replacement.** For the diagnostic it is a NICE-TO-HAVE that I am
nonetheless recommending you do, purely because the data is already in the repo and
it is the highest-value addition (§3). A diagnostic paper *can* honestly publish
GP-as-oracle with the scope stated. An emulator-replacement paper cannot — but that
is not this paper anymore.

**Is the h-basis-test refutation sound, and is it enough?**
**Sound, yes; enough, no — and it is correctly not being oversold.** The test
(`h_basis.json`) is a clean, committed, reproducible refutation of the authors' *own
prior guess* that h acts as an Alcock–Paczynski-like k-rescaling: ∂P/∂h correlates
only ≈ −0.25 with ∂P/∂lnk and the template explains ~6% of variance at all three
redshifts. As a *refutation of AP* it is sound and it is exactly the kind of
self-correcting loop a methods paper should show. But — as the prior referee
correctly notes and the draft now concedes — refuting AP does not *establish*
"weak/under-determined"; a 6% overlap is weak evidence either way. The honest
statement is the negative one: "h is NOT a k-rescaling basis wall; under value-search
its feature barely enters (x0 first at complexity 20, the max) and Sobolev cannot
rescue a ~1% signal." That is enough *for the claim it now makes*. It would be
over-reach to present a positive mechanism. The draft's current softening ("basis
test refutes AP; resistance is weak-signal") is the right altitude.

**Single addition that most raises confidence without a year-long project:**
the sims-truth slope anchor of §3 — because the data is in the repo (`600` design
points + LOO set), it is days not months, and it is the one thing that converts
GP-as-oracle from a stated caveat into a *validated* proxy (or finds the real weak
link). Nothing else has that confidence-per-effort ratio.

---

## 5. Bottom line

This is a careful, self-critical, reproducible methods study asking a question the
SR-emulator literature (the syren family, arXiv:2506.08783) has skipped: is a
value-accurate symbolic surrogate *derivative*-accurate enough to put inside a Fisher
matrix? The budget control alone — verified clean against the raw CSV (the entire
complexity-13→35 ns front never crosses the gate while a smaller Sobolev search does)
— is worth publishing. The retraction of the σ-claim is to the authors' credit and
*enables* this paper rather than weakening it.

Publish it as a methods paper (JCAP / OJA / ML4PS), after: **(i) an across-seed
spread that hardens the taxonomy** (the one true blocker), and **(ii) a
covariance-grounded gate** (cheap, grounds the metric). Add the **sims-truth slope
anchor** (data already in the repo) as the highest-value strengthening, and scope
off-fiducial / the herei×alphaq coupling as honest limitations. Re-scored to the
diagnostic bar, the prior referee's "major revisions" is really "minor-to-moderate
revisions" — two cheap must-haves and one high-value optional, all inside a week.
