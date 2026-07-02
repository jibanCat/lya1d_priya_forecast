# Documentation & scientific-consistency audit — Lyα P1D PySR forecast/diagnostic

**Auditor:** Lyα/cosmology domain specialist (housecleaning pass)
**Date:** 2026-06-09
**Repo:** `/home/mfho/lya1d_priya_forecast`, branch `stage10-multiz-sobolev`
**Scope:** all root `README*`/`*.md`, `docs/*.md`, `notebooks/*`, cross-checked
against committed `results/` sidecars and the source of truth in memory.

---

## Executive summary

The project **pivoted on 2026-06-08** from a *forecast* claim (σ_PySR/σ_GP
"ladder") to a **derivative-faithfulness diagnostic / failure-modes** result.
The pivot is driven by a 4-agent adversarial review (2026-06-05) that showed the
σ-ratio is confounded by construction (σ_perfect_1D ≡ σ_GP is a forced Jacobian
identity anchored at P_GP(fid); the GP-slice fallback silently prints GP-derived
σ in the σ_PySR column).

**Two docs carry the current science and are excellent:**
`HANDOFF.md` (refreshed 2026-06-09) and
`docs/PARETO_FAITHFULNESS_WALKTHROUGH.md` (the designated source of truth). I
verified their headline numbers against the committed sidecars and the code path
— they are **fully consistent** (see "Consistency checks", below).

**Everything a newcomer reads first is stale.** `README.md`, `README_v2.md`,
`docs/ONBOARDING.md`, `docs/REPRODUCE.md`, and the top of `docs/PAPER_NOTES.md`
all present the deprecated σ_PySR/σ_GP forecast as the headline, with no banner
redirecting to the diagnostic. **No entry-point doc links to `HANDOFF.md` or the
walkthrough.** A newcomer would reconstruct the *abandoned* thesis and quote
Phase 2's 2.35% rel-err as the paper headline — exactly the claim the review
killed. All three notebooks teach only the deprecated σ-ladder modes.

The fix is mostly **redirection and banners**, not rewriting — the stale docs are
still accurate descriptions of *machinery that still exists* (the single-z
pipeline, the additive combine), just no longer the *scientific headline*.

---

## (a) README / doc updates needed — prioritised

### P0 — a newcomer is actively misdirected

1. **`README.md` — add a top banner + reframe the one-liner.** The opening
   ("how close is `σ_PySR` to `σ_GP`, parameter by parameter?") is the
   deprecated thesis. Add a banner: *"As of 2026-06-08 this project is a
   derivative-faithfulness **diagnostic**, not a σ_PySR/σ_GP forecast. Start with
   `HANDOFF.md` and `docs/PARETO_FAITHFULNESS_WALKTHROUGH.md`. The σ-ladder
   workflow below still runs but is **not** the scientific result — see the
   review verdict in `HANDOFF.md`."* Also fix the stale "Project status
   (2026-05-07)" block that quotes "2.35% mean / 7.05% p99 / 12.11% max" as the
   headline.

2. **Add a redirect pointer in every entry doc.** None of `README.md`,
   `README_v2.md`, `docs/ONBOARDING.md`, `docs/REPRODUCE.md` link to `HANDOFF.md`
   or the walkthrough. Add a one-line "current science" pointer at the top of
   each. (`docs/PAPER_NOTES.md` is the only doc that even names the walkthrough,
   and it still leads with the Phase 2 σ-ratio.)

3. **`docs/PAPER_NOTES.md` — demote the "🎯 quotable single result" TL;DR.** It
   tells the paper author to headline Phase 2's 2.35%/7.05%/12.11% rel-err and
   the σ-ratio scorecard — the confounded claim. Replace/precede with: the paper
   is now a diagnostic; the headline is the per-parameter faithfulness taxonomy +
   the ns Sobolev recovery (0.603→0.193) + the budget control. Note that the
   Phase-2 machinery and its rel-err numbers remain valid as an *emulator-quality*
   statement but cannot be sold as a *forecast-fidelity* (σ) claim.

### P1 — internally contradictory or half-corrected

4. **`docs/ONBOARDING.md` § 4.2 — finish the correction.** It *already*
   documents "σ_perfect_1D ≡ σ_GP for the additive combine" (the review's central
   finding) yet still concludes "the meaningful headline metric is `σ_PySR /
   σ_GP`." That is the precise claim the review retired. Reframe: the σ_PySR/σ_GP
   ratio is a Jacobian self-comparison, not an emulator test; the headline metric
   is now `grad_err` (median_k |∂P_F^eq/∂θ ÷ ∂P_F^GP/∂θ − 1|, gate 0.25).

5. **`docs/REPRODUCE.md` / `README_v2.md` — label as "deprecated forecast
   path".** Both reproduce the Phase 1.5 / Phase 2 σ-scorecards as the deliverable
   with no caveat. Keep them (the pipeline is real and tests pass) but add a
   header: *"This reproduces the σ-ladder, which is **no longer the headline** —
   see review verdict. For the current diagnostic, reproduce via
   `scripts/make_diagnostic_figs.py` (emulator-free)."*

6. **`docs/AP_REMEDIATION_PLAN.md` and `docs/PAIR_FIT_PLAN.md` — mark as
   superseded/parked.** Phase 3 Ap remediation and the 4-pair coupling are
   carry-over from the forecast era. memory/`active_work.md` lists Phase 3 as
   "paused, not abandoned." Add a one-line status banner so a newcomer doesn't
   treat them as active.

### P2 — minor / hygiene

7. **`docs/FIGURES.md`** describes the old per-forecast figure set
   (`fig01_gp_at_fiducial` … eBOSS DR14 σ panels), not the diagnostic figures
   (`pareto_faithfulness`, `summary_scorecard`/`faithfulness_scorecard`,
   `ns_money_panel`/`ns_budget_panel`, `crossz_faithfulness`). Add a section for
   the diagnostic figures or point to the walkthrough.

8. **Figure-name drift in the walkthrough.** The walkthrough embeds
   `summary_scorecard.png` and `ns_money_panel.png`; the results dir *also* now
   has newer `faithfulness_scorecard.png` and `ns_budget_panel.png` (both
   committed, with PDFs). Both old names still exist so no broken links, but
   reconcile which pair is canonical to avoid a future dangling reference.

9. **`docs/SR_EMULATOR_LITERATURE_NOTES.md`** is dated and feeds "Stage 8" — fine
   as background, but note it predates the diagnostic redirect.

---

## (b) Stale / contradictory scientific statements to fix

- **"σ_PySR/σ_GP is the headline metric"** (README.md, README_v2.md §1e/§2g,
  ONBOARDING §4.2, PAPER_NOTES TL;DR). Contradicts the review verdict and
  `HANDOFF.md`. The ratio is a forced Jacobian identity (σ_perfect_1D ≡ σ_GP),
  not an emulator/forecast test. **This is the single most important fix.**
- **"9/11 params get Fisher-faithful equations" framed as a forecast win.** Per
  the review (memory/`review_verdict_sr_emulator.md`), the GP-slice fallback
  prints GP-derived σ in the PySR column, so the "9/11" count is GP-contaminated.
  The current honest framing (HANDOFF/walkthrough) is a 4-category *taxonomy*,
  not a pass-count.
- **"Phase 2: 2.35% mean / 7.05% p99 / 12.11% max rel-err" as the paper
  headline** (PAPER_NOTES TL;DR, README.md status block). Still a valid
  *emulator-accuracy* statement, but it is no longer the paper's claim and must
  not be quoted as forecast fidelity.
- **The "h = AP / k-rescaling" hypothesis** appears as a live guess in older
  notes; it is now **refuted** (`scripts/h_basis_test.py`, 2026-06-09: corr
  ≈ −0.25, ~6% variance). The walkthrough/HANDOFF already state the refutation;
  ensure no other doc still asserts "hub acts like an Alcock–Paczyński
  distortion."
- **Residual code-comment inconsistency (not a doc, but flag it):**
  `src/priya_forecast/derivative_gate.py:67` docstring for the *multi-z* gate
  still says it matches `∂logP_GP/∂θ`, whereas the single-z gate (and the
  walkthrough's "fixed mislabel" note) operate on **linear P_F** slopes (both
  `refit.predict` and `gp.predict` return raw P_F — verified). Reconcile the
  multi-z docstring with the linear-P convention.
- **IGM-thermal verdicts stated z-uniformly are wrong.** The He II reion block
  (herei, heref, alphaq) is faithful at z ≤ 3.6 and blows up at z = 4.2 (its
  imprint is redshift-localised; the GP slope is near-noise away from the reion
  epoch). Any doc that states a single faithfulness verdict for these params
  without a per-z caveat is misleading. (HANDOFF/walkthrough handle this; older
  notes may not.)

---

## (c) What a reproduction notebook must show

The existing notebooks (`01_gp_only`, `02_forecast_only`, `03_refit_and_forecast`)
**all teach the deprecated σ-ladder** and none demonstrates the diagnostic. A new
`notebooks/04_faithfulness_diagnostic.ipynb` (the headline reproduction) should:

1. **Run emulator-free from committed sidecars.** Call
   `scripts/make_diagnostic_figs.py --out-dir results/single_z_stage_pareto_diag`
   and render `pareto_faithfulness.png`. Verified: all 70 `grad_faith_*.csv`
   sidecars and their paired `pareto_*.csv` are **committed**, and
   `make_diagnostic_figs.py` reads only those — no GP/Julia/PySR needed. State
   this prominently (it is the repo's reproducibility selling point).
2. **Define the metric explicitly:** `grad_err = median_k |∂P_F^eq/∂θ ÷
   ∂P_F^GP/∂θ − 1|` at fiducial, in **linear P_F** (Fisher-consistent), gate 0.25.
   Note that `value_mse` is in log-P and that `grad_err` is NOT ∂logP (correct
   the old mislabel up front).
3. **Make the Mirage literal (the ns money panel):** show value@budget (maxsize
   35) reaching the *lowest* value_mse yet staying red (grad_err 0.319), while
   Sobolev@20 clears the gate (0.193) at comparable value_mse. Value accuracy ⊥
   derivative faithfulness.
4. **Walk the 4-category taxonomy** with the z=3.6 table: robustly faithful
   {dtau0, tau0, heref, alphaq, hireionz}; selection-sensitive {Ap, herei,
   omegamh2}; generative-Mirage-cured-by-Sobolev {ns}; resistant {hub,
   bhfeedback}. Tie each to physics (mean-flux/IGM amplitudes vs tilt-about-pivot
   vs weak/under-determined response).
5. **Show the budget control** (ns at maxsize=35 still fails) so the reader sees
   the Mirage is *generative*, not search-starvation — this directly answers the
   review's central objection.
6. **Show the cross-z panel** (z=2.6/3.6/4.2, retrained) and state the
   non-uniformity: the He II reion block is faithful at z ≤ 3.6 and blows up at
   z=4.2; caveat the GP's ~2% accuracy at z=4.2.
7. **State the honest scope** at the end: this is a Jacobian/derivative-fidelity
   diagnostic under a GP-as-oracle, NOT an emulator-vs-simulations test; truth is
   the GP, not the simulations (per the review). List the open decisive
   experiments (joint herei×alphaq refit; sims-as-truth validation).

A second, optional notebook could *reproduce-and-deprecate* the σ-ladder: run
`02_forecast_only`, then numerically show σ_perfect_1D ≡ σ_GP to make the review's
"it's a Jacobian" point concrete and self-evident.

---

## Consistency checks performed (all PASS)

- **Headline ns numbers** match committed sidecars exactly: value@20 best-loss
  grad_err = 0.603, Sobolev@20 = 0.193, budget@35 = 0.319 (read from
  `results/{single_z_stage6_log,single_z_stage9,decider_budget_z3.6}/refit/z3.6/grad_faith_ns.csv`).
  Consistent with the walkthrough and HANDOFF tables.
- **Emulator-free repro path is sound:** 70 grad_faith sidecars + paired pareto
  CSVs are git-tracked; `make_diagnostic_figs.py` hard-codes those three dirs and
  loads only CSVs. The diagnostic figures + PDFs are committed under
  `results/single_z_stage_pareto_diag/`.
- **Metric label is correct in code:** `grad_err` differences linear-P slopes
  (`equation_param_gradient`→`refit.predict`, `gp_param_gradient`→`gp.predict`,
  both raw P_F). The sidecar header `log_space=True` is only the SR
  target/normalization flag — no contradiction with the "linear P_F gate" claim.
  (Lone exception: the *multi-z* gate docstring still says ∂logP — code comment,
  flagged above.)

---

## Bottom line

The **science is in good shape and the current docs that describe it
(`HANDOFF.md`, the walkthrough) are accurate and data-backed.** The
housecleaning debt is entirely in the **entry-point layer**: the first docs a
newcomer reads still sell the retired σ_PySR/σ_GP forecast and never point to the
current diagnostic. Highest-leverage action: P0 banners + redirect links so the
diagnostic redirect is discoverable, then demote the Phase-2 σ-ratio headline in
`PAPER_NOTES.md` and finish the half-done correction in `ONBOARDING.md §4.2`. A
single emulator-free diagnostic notebook would complete the reproduction story.
