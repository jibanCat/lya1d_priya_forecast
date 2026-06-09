# Referee report — "Derivative-faithfulness of symbolic-regression emulators for the Lyman-α P1D"

**Referee:** senior Lyman-α forest / P1D cosmologist (eBOSS/DESI/XQ-100/KODIAQ-SQUAD, PRIYA)
**Date:** 2026-06-09
**Materials read:** `README.md`, `HANDOFF.md`, `docs/PARETO_FAITHFULNESS_WALKTHROUGH.md`, the four diagnostic figures (`pareto_faithfulness`, `faithfulness_scorecard`, `ns_budget_panel`, `crossz_faithfulness`), the committed `grad_faith_*.csv` sidecars (z=3.6 value/Sobolev, the z=3.6 budget@35, and the z=2.6/4.2 retrains), `results/h_basis_test/h_basis.json`, the code (`pareto_diag.py`, `derivative_gate.py`, `grad_faith_io.py`, `sobolev_loss.py`, `parameters.py`, `eval_grad_faithfulness.py`, `make_diagnostic_figs.py`, `h_basis_test.py`), and the plain-language draft `PAPER_NARRATIVE.md`.

---

## 1. Recommendation

**Major revisions.** The core idea is genuine and the execution is unusually honest, but the headline result is single-redshift / single-seed against a GP treated as truth, and several load-bearing physical claims (the "resistant" verdicts, the He II cross-z story) rest on one or two equations or on emulator-limited regimes. The contribution is publishable in JCAP/OJA after the scope is broadened and a handful of robustness checks are added; it is not acceptable as-is on the strength of a z=3.6, seed-42 demonstration.

---

## 2. Summary of the contribution (as I understand it)

The authors distil the PRIYA multi-fidelity GP emulator of the Lyα 1D flux power spectrum into compact, **per-parameter** symbolic equations (PySR), combined additively as a Taylor-style expansion around a fiducial cosmology. The actual contribution is *not* the distillation (that is prior work / the student pipeline) but a **diagnostic**: for each of the 11 PRIYA parameters, does an equation that reproduces the GP's *values* P_F also reproduce its *slopes* ∂P_F/∂θ — the only quantity a Fisher forecast consumes?

The instrument is a per-candidate metric `grad_err = median_k |∂P_F^eq/∂θ ÷ ∂P_F^GP/∂θ − 1|` evaluated at the fiducial point over non-negligible k-bins, in **linear** P_F (Fisher-consistent), with an operating-point gate at 0.25. The findings:

- **"Fisher's Mirage":** an equation can have tiny value error and a badly wrong slope. The data bear this out starkly — e.g. bhfeedback candidates with value_mse ≈ 3×10⁻⁵ but grad_err ≈ 1.3–1.7.
- A **four-way taxonomy** (robustly faithful / selection-sensitive / generative Mirage / resistant).
- A **Sobolev** derivative-matching training loss (λ=5, on the normalised log-P slope) as the cure, demonstrated cleanly on ns (0.60 → 0.19).
- A **budget control**: ns value-search at maxsize=35 reaches the *lowest* value error of any series (3.8×10⁻⁴) yet still fails the gate (0.319) — so the ns Mirage is *generative*, not search-starvation.
- A **cross-z** retrain (z=2.6/3.6/4.2) showing the taxonomy is not redshift-uniform.
- An **h "basis test"** refuting the authors' own prior guess that h acts as an Alcock–Paczynski-like k-rescaling.

The authors explicitly **drop** an earlier σ_PySR/σ_GP forecast claim after an internal review found it confounded by construction (σ_perfect_1D ≡ σ_GP is a forced Jacobian identity). I regard that retraction as correct and to their credit.

---

## 3. Strengths

1. **The central question is real and, to my knowledge, novel in this literature.** P1D forecasting (eBOSS DR14/DR16, DESI, XQ-100, KODIAQ-SQUAD) is now overwhelmingly emulator-based, and symbolic/analytic surrogates (the syren family, arXiv:2506.08783) are reported on *value* RMSE almost exclusively. Asking whether a value-accurate surrogate is *derivative*-accurate is exactly the right question for anyone who wants to put an SR emulator inside a Fisher matrix, and the syren-style Pareto plots indeed never report it. This is a useful methodological contribution, not a solution looking for a problem.

2. **Unusual internal honesty.** The σ-ratio retraction, the "GP-as-oracle, not sims" caveat, the single-seed/single-z disclaimers, the "multi-z is a branch-name artifact" note, and the explicit footnote that Sobolev *worsens* the best-loss pick for heref (0.154→0.206) and alphaq (0.152→0.173) are all the kind of disclosure referees usually have to extract. I verified each against the CSVs and they are accurate.

3. **The budget control is the strongest single result and it is clean.** I checked `decider_budget_z3.6/.../grad_faith_ns.csv`: best-loss at complexity 35 is value_mse 3.82×10⁻⁴ / grad_err 0.319 (FAIL), and the entire complexity-13→35 front never crosses 0.25, while Sobolev@18 reaches 0.193 at 4.74×10⁻⁴. The paired framing (the cure is a *better objective*, not *more search*) is well supported and is the part a methods-skeptical referee will find most convincing.

4. **Reproducibility of the figures is genuinely good.** The four figures and every taxonomy number regenerate emulator-free from tracked `pareto_*.csv` + `grad_faith_*.csv` via `make_diagnostic_figs.py`; the GP-only pieces (sidecar regeneration, h basis test) are gated to fail loudly on a bare clone, and the h-test result is committed as JSON. The linear-vs-log discipline is implemented correctly: `derivative_gate.py` differences raw `gp.predict`/`refit.predict` (linear P_F) so `grad_err` is genuinely a linear-slope ratio, while `sobolev_loss.py` trains on the normalised log-P slope — the "cure and diagnostic live in different spaces" caveat is real and correctly bridged.

5. **The figures mostly communicate.** The scorecard and ns budget panel are clear and self-labelled (gate line, the two resisters annotated with their numbers, the budget/Sobolev endpoints called out). A reader gets the Mirage in one glance from the scorecard.

---

## 4. Major concerns (these drive the recommendation)

### 4.1 GP-as-oracle: the diagnostic never touches the simulations
The entire result measures faithfulness *to the emulator*, with the GP posterior mean treated as exact truth and its finite-difference slope as the gold standard. This is disclosed, but it is more than a caveat — it caps what the paper can claim. The GP's *own* ∂P_F/∂θ is a smoothed, prior-shaped object with its own error (the authors themselves note ~1–2% value accuracy, larger at z=4.2), and near a sparsely-sampled fiducial in 11-D the GP slope can be dominated by kernel choice rather than by the sims. So a "Mirage" verdict can mean *the SR equation disagrees with a GP slope that is itself unreliable*. The whole framework would be far more compelling with at least one anchor to the PRIYA simulations themselves — e.g. compare ∂P_F/∂θ from the GP against a finite-difference of the actual HF sims at the few parameters where paired runs exist, for one or two parameters (ns and a mean-flux parameter). Without that, the reader cannot tell whether `grad_err` measures SR failure or GP slope noise. This is the single biggest gap for a Lyα-cosmology audience.

### 4.2 The "resistant" verdicts are under-evidenced
- **hub.** Under value@20 the hub front has **exactly one** Fisher-safe candidate (complexity 20, the max; x0 enters only there) — I confirmed `grad_faith_hub.csv` is a single row. A taxonomy verdict ("resistant") and a mechanism claim ("weak/under-determined") drawn from a one-point front is fragile. The h basis test (corr ≈ −0.25, ~6% variance against the ∂P/∂lnk template) is a nice, committed, reproducible refutation of the AP guess — but refuting AP is not the same as *establishing* "weak/under-determined." A 6% variance overlap with the k-rescaling template at the fiducial is weak evidence either way; h's ~1% P1D effect is the real argument, and that is a known statement about the GP, not a discovery here. State the hub verdict as "under value-search the feature barely enters and Sobolev cannot rescue a ~1% signal," and stop short of a positive mechanistic claim.
- **bhfeedback.** The "gate can't adjudicate" framing is physically reasonable (ε_AGN is effectively priored out and ∂P/∂bhfeedback is near noise), but it doubles as an escape hatch: any parameter the gate can't score gets filed under "resistant — not the equation's fault." That is defensible for bhfeedback specifically, but the paper should not present "resistant" as a single category when it actually contains two very different things (a possible expressivity/under-search wall vs. an unscoreable target). The cross-z table makes this worse: bhfeedback's grad_err *improves* to 0.37 at z=4.2 while the He II block blows up — i.e. "resistant" is itself redshift-dependent, which undercuts the four-box taxonomy as a stable object.

### 4.3 Single-seed, single-z headline with near-gate margins
The headline taxonomy is z=3.6, seed-42, one draw. PySR is stochastic, and the marquee result (ns Sobolev = 0.193) clears a *chosen* 0.25 gate by 23%. Several other "passes" are similarly close (heref Sobolev best-loss 0.206; omegamh2 0.198; hireionz at z=2.6 0.324 is already a *fail*). With no across-seed spread reported, a reader cannot tell which side of the gate these parameters truly sit on. The selection-sensitive category is the most exposed: for Ap, the *entire* "a faithful equation exists" claim rests on a **single** low-complexity outlier (complexity 4, grad_err 0.108) in a front where every other candidate fails (0.26–0.91) — that is one lucky equation, not a robust property of the front, and it may well move across seeds. The 0.25 gate also needs a quantitative tie to the KODIAQ-SQUAD covariance: the draft asserts a 25%-per-bin slope error is "sub-dominant," but never demonstrates the resulting σ_θ inflation, so the operating point is currently asserted, not derived.

### 4.4 The cross-z "physics" story is partly an emulator artifact, and the authors know it
The He II block (herei/heref/alphaq) "blowing up" at z=4.2 is presented as faithfulness *tracking the physics* (the imprint is redshift-localised, slope→0 away from the reionisation epoch). I find the physical direction credible — heref ends He II reion at z≈2.6–3.2, herei starts it at z≈3.5–4.1, so a weak slope far from epoch is expected. **But** the same z=4.2 is where the authors concede the GP is least accurate (~2% vs ~1%), so the "blow-up" conflates (i) a genuinely small physical slope, (ii) an unreliable GP slope, and (iii) SR failure — and the diagnostic cannot separate them. As written, "faithfulness tracks the physics" is over-stated: it tracks *the GP's slope magnitude*, which at z=4.2 is itself suspect. This needs to be reframed as "the gate becomes unscoreable where the slope is small, whether because of physics or emulator error," with the physics offered as the likely but unproven driver.

### 4.5 The most important physical limitation is invisible to the diagnostic by construction
The per-parameter / diagonal-Fisher construction cannot see parameter couplings, and the named example — the herei×alphaq posterior correlation (ρ ≈ +0.45) — is exactly the He II degeneracy a Lyα referee cares about most. The draft handles this correctly in prose (both are *individually* slope-faithful; the coupling is off-diagonal/combine-level). But this means the diagnostic, as a tool for validating an SR emulator *for inference*, certifies only the diagonal of the Fisher matrix. For a method whose selling point is "is this safe for a forecast?", silence on the off-diagonal is a real limitation, not a footnote, and the paper should say plainly that a green taxonomy does **not** imply a faithful joint forecast.

---

## 5. Minor points

- **"Fisher's Mirage" branding.** The phenomenon (value-fit ⇏ derivative-fit) is correct and the bhfeedback/ns data demonstrate it well, but the name is borrowed (arXiv:2406.06067) and risks overselling a fairly elementary fact (a function and its derivative are independently constrained). Keep the demonstration, soften the branding.
- **Taxonomy stability.** Given §4.2 and §4.4, the four-box taxonomy is not redshift-stable (bhfeedback flips toward "pass" at z=4.2; the He II block flips toward "fail"). Either present the taxonomy explicitly as *per-redshift*, or demote it from "taxonomy" to "observed categories at z=3.6."
- **heref/alphaq mislabel.** They are listed "robustly faithful," but Sobolev's *best-loss* pick rises (0.154→0.206, 0.152→0.173) and they only stay green via a best-faith equation. This is footnoted, but the taxonomy table itself should carry the asterisk, not bury it.
- **Showcase equations vs graded equations.** The τ₀/Aₚ equations displayed in the narrative are multi-z showcase fits (with z⁴ envelopes) and are *not* the z=3.6 objects the diagnostic grades. The draft now flags this, but in the paper itself the two must not appear adjacent without a loud label — a careful reader will otherwise think those equations were graded.
- **k^θ pathology.** The point that ∂_θ k^θ = k^θ ln k flips sign at k=1 and explodes at small k (poison for Fisher, fine for value) is a genuinely nice, citable observation about why SR-for-forecasting must restrict its operator set. Promote it from an aside to a short methods paragraph.
- **No trig in the operator set.** Not stated in the materials I read, but if oscillatory unary operators are in play, their derivatives will wreck Fisher slopes; confirm the operator basis is derivative-safe and state it.
- **`log_space=True` sidecar headers.** These refer to the *value* re-scoring, not to grad_err (which is linear). This is correctly footnoted but is a foot-gun for anyone opening the CSVs; consider renaming the flag.
- **Test coverage.** Only two test files touch the new diagnostic (`test_grad_faith_io.py`, `test_pareto_diag.py`), both pure-plotting/IO. The scientifically load-bearing function — `derivative_faithful` and the masking logic — deserves a unit test with a synthetic known-slope case.

---

## 6. Experiments / changes required before acceptance

1. **One sims-as-truth anchor (required).** For at least two parameters (ns and one mean-flux parameter), compare the GP's ∂P_F/∂θ against a finite difference of the actual PRIYA HF simulations at the fiducial. Show that the GP slope the gate trusts is itself reliable; otherwise every "Mirage"/"resistant" verdict is ambiguous between SR error and GP slope error.
2. **Across-seed spread (required).** Re-run the z=3.6 value and Sobolev fronts for ≥5 PySR seeds and report grad_err mean±spread per parameter. Reclassify any parameter whose category flips across seeds (I expect Ap and the near-gate passers to be fragile). The ns budget result and the ns Sobolev pass in particular need a seed band.
3. **Off-fiducial validation (required).** The gate is evaluated only at the fiducial. Re-score grad_err at a few off-fiducial points within the prior (e.g. ±1σ along each axis) — a slope that is faithful only at the linearisation point is of limited use for a real (non-local) forecast, and the k^θ pathology argument predicts trouble near prior edges.
4. **Quantify the 0.25 gate against the covariance (required).** Propagate a controlled per-bin slope error through the single-z Fisher matrix with the real KODIAQ-SQUAD covariance and show the resulting σ_θ bias as a function of grad_err. This turns 0.25 from an asserted operating point into a derived tolerance and tells the reader what a given grad_err *costs* in σ.
5. **The herei×alphaq coupling, shown not just stated (strongly encouraged).** Build the 2-parameter combined SR prediction and compare the recovered 2×2 sub-Fisher (or the ρ) against the GP's, to demonstrate the magnitude of the off-diagonal error the per-parameter diagnostic cannot see. This is the result a Lyα audience will most want, and it would convert a "limitation paragraph" into a quantified scope statement.
6. **Disambiguate the z=4.2 He II degradation (encouraged).** Separate "small physical slope" from "GP unreliable at z=4.2" — e.g. report the GP's own z=4.2 slope magnitude and its emulator error bar alongside grad_err, so the "tracks the physics" claim is supported rather than asserted.
7. **Harden hub before any positive mechanistic claim (encouraged).** A one-point value@20 front cannot support "weak/under-determined" as a *finding*. Either widen the search until the hub feature enters at lower complexity, or restrict the hub claim to "the feature barely enters and a ~1% signal is below what Sobolev can rescue."

---

### Closing

This is a careful, self-critical study asking a question the SR-emulator literature has skipped, and the budget control alone is worth publishing. But the headline rests on a GP-as-oracle, single-seed, single-redshift, fiducial-point evaluation, and three of the four taxonomy boxes ("selection-sensitive," "resistant," the z=4.2 He II story) currently lean on one or two equations or on a regime where the emulator itself is weakest. Add a simulation anchor, a seed band, off-fiducial scores, and a covariance-grounded gate, and this becomes a solid methods paper that I would be glad to see in print. I look forward to a revised version.
