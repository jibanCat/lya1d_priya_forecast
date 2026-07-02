# PR #6 review — Lyα/PRIYA correctness lens

**Reviewer focus:** per-parameter physical mechanisms, cross-z reading, validity of the
h basis test (AP refutation), single-z scope, and walkthrough-vs-sidecar agreement.
Branch `stage10-multiz-sobolev`, base `main`, PR #6.

## (1) Verdict: **APPROVE-WITH-NITS**

The science is sound and the numbers are reproducible. Every load-bearing number in
`docs/PARETO_FAITHFULNESS_WALKTHROUGH.md` was re-derived from the committed sidecars and
matched exactly; the h basis test was re-run against the live GP and reproduced its
claimed corr/variance. The physics taxonomy is internally consistent with the parameter
priors and the He II reionization epochs. The blocking-class issues are about
**figure reproducibility/provenance**, not about a wrong scientific claim. No
science-correctness blocker found.

## (2) What is correct and well-built

- **grad_err is the production gate, not a re-implementation.** `scripts/eval_grad_faithfulness.py:38-48`
  (`median_rel_error`) uses identical `floor_frac=1e-3`, `keep = |target| >= floor_frac*amax`,
  `median(|cand/target - 1|) <= tol` logic to `src/priya_forecast/derivative_gate.py:42-59`
  (`derivative_faithful`), and the candidate set is the same `_filter_fisher_safe(df, n_features=3)`
  used by the real forecast (`single_z/forecast.py:168,215`). The sidecars therefore reproduce
  the gate the production path applies, which is the right way to build a diagnostic.
- **Metric spaces are now labeled correctly.** Confirmed `gp.predict` returns linear P_F
  (values ~10–72, all >1), so `gp_param_gradient`/`equation_param_gradient` difference linear-P
  predictions → `grad_err` is a linear-P slope ratio, the Fisher-consistent quantity. The
  corrected note (walkthrough lines 48-54) is accurate; `value_mse` is correctly log-P.
- **All z=3.6 table numbers verified** (lines 109-119): best-loss, best-faith, anypass (✓/✗),
  and `x0@` all match the committed sidecars to 3 digits (spot-checked ns 0.603/0.512/8,
  hub 1.000/x0@20, bhfeedback 1.715/1.334/x0@11, plus the full 11-row sweep).
- **Budget control is honest and reproduces.** `results/decider_budget_z3.6/.../pareto_ns.csv`
  genuinely reaches complexity 35 (29 rows); best-loss grad_err 0.319, min over the whole
  13→35 front 0.319, none ≤ 0.25 — exactly as claimed (lines 121-135). value_mse 3.8e-4 (budget)
  vs 4.7e-4 (Sobolev) and the "~24% worse" arithmetic check out. The "Mirage is generative, not
  a budget shortfall" conclusion is supported by the data.
- **Cross-z numbers all match** (lines 234-246): z=2.6 and z=4.2 Sobolev best-loss grad_err
  recomputed for all 11 params; e.g. z=4.2 herei 0.709, heref 2.690, alphaq 1.556, hub 1.215,
  bhfeedback 0.374 — each matches.
- **h basis test reproduced live** (`scripts/h_basis_test.py`): corr(dP/dh, dP/dlnk) =
  −0.208/−0.250/−0.263 at z=2.6/3.6/4.2 and AP-frac-var = 0.061/0.062/0.068. Both match the
  walkthrough ("corr ≈ −0.25", "~6% variance"). I additionally checked the AP-only R² with no
  P nuisance column (0.043/0.063/0.069) — same conclusion, so the refutation does **not** hinge
  on the joint-fit construction.
- **h basis test is a valid AP test and the refutation is sound.** Under k→k(1+ε) the linear
  response of P at fixed k is dP/dε = k·dP/dk = dP/dlnk, so `Tap = dP/dlnk` is the correct
  k-rescaling template. The test also regresses against P (the amplitude/measure mode), so a
  pure-amplitude AP-like response would also be captured — and it still reaches only ~6%. The
  conclusion "h is weak/under-determined, not a basis wall" is the physically correct reading,
  and it correctly overturns the earlier AP guess.
- **Physics taxonomy matches the priors/epochs.** Verified against `PARAMS_11D`: hub prior
  (0.65,0.75) and omegamh2 (0.14,0.146) are tight/weak-signal → "under-determined" reading is
  right; bhfeedback (0.03,0.07) tiny effect → "priored out / weak gradient" is right; herei
  fid 4.0 prior (3.5,4.5) and heref fid 2.765 prior (2.2,3.2) match the stated He II reion
  epochs. The cross-z story (He II block faithful at z≤3.6, blows up at z=4.2 because the imprint
  is redshift-localised → GP slope tiny → gate can't adjudicate, the bhfeedback mechanism
  switched on by z) is physically coherent and is exactly the floor_frac-masking failure mode.
- **Single-z scope is flagged** (status line, panel caption, and lines 256-258: per-redshift
  IGM verdicts, 9-z-bin Fisher dominated by informative redshifts, z=4.2 emulator-limited caveat).
- **No NaN/inf in any committed sidecar** — every evaluated candidate had a usable GP gradient,
  so no "anypass" verdict is silently riding on a masked-out row.

## (3) Concrete issues (file:line + fix)

1. **[blocking-ish — provenance] Three of the four embedded figures have no committed generator.**
   `docs/PARETO_FAITHFULNESS_WALKTHROUGH.md` embeds `summary_scorecard.png` (line 81),
   `ns_money_panel.png` (line 86), and `crossz_faithfulness.png` (line 230). No committed script
   writes any of these names (`grep -rln` over `scripts/`,`src/` returns nothing).
   `scripts/make_diagnostic_figs.py` instead writes `faithfulness_scorecard.png` (line 113) and
   `ns_budget_panel.png` (line 145) — names the walkthrough never references. So the PR's "Figures
   regenerate emulator-free: `python scripts/make_diagnostic_figs.py`" claim only covers 1 of the
   3 embedded supplementary figures; the other two committed PNGs are orphaned blobs from an
   earlier figure script. **Fix:** either (a) rename the script outputs to the embedded names
   (`faithfulness_scorecard`→`summary_scorecard`, `ns_budget_panel`→`ns_money_panel`) and add the
   cross-z figure generator to `make_diagnostic_figs.py`, or (b) repoint the walkthrough image
   links to the names the script actually emits. As-is the embedded figures are not reproducible.

2. **[nit] Dead `return out_path` in `pareto_diag.py`.** `src/priya_forecast/pareto_diag.py:121-122`
   has the statement twice; the second is unreachable. Drop line 122.

3. **[nit] Dead/misleading variable in the scorecard.** `scripts/make_diagnostic_figs.py:105`
   computes `bf = bestloss(SOBOLEV, p)` (re-fetching the same best-loss value already in `s`) with
   a comment "# same here; note best-faith in text", but `bf` is never used and the value is
   best-loss, not best-faith. Remove the line or compute the real best-faith if intended.

4. **[nit — clarity, not correctness] The cure and the gate live in different metric spaces;
   the transfer is empirical.** The Sobolev loss matches **normalised-log-P** slopes
   (`sobolev_loss.py:72-76`: `grad_phys = (lp_p − lp_m)/denom`, ×width/std), while the gate scores
   **linear-P** slope ratios. These are related by 1/P, so matching one strongly constrains the
   other, and the empirical ns result (0.60→0.19 on the linear-P gate) shows the transfer works —
   but the walkthrough notes the space difference without stating that the cross-space transfer is
   empirical, not exact. One sentence would close the loop and pre-empt a referee asking "why does
   a log-P objective move a linear-P metric?".

5. **[nit] herei×alphaq coupling caveat is correctly scoped but worth a forward-pointer.** The
   walkthrough (lines 156-161) correctly states the known +0.45 coupling is an off-diagonal/
   combine-level effect invisible to a per-parameter 1D diagnostic. That is right (the 1D marginal
   slopes for both are individually faithful here), but since the headline finding for this project
   is that coupling, a one-line pointer to where it *is* measured (the multi-D combine) would keep
   a referee from reading "individually faithful" as "coupling resolved."

## (4) What would block merge

- **Nothing on the science.** Numbers, taxonomy, cross-z reading, and the AP refutation are
  correct and reproduce.
- **Issue #1 (figure provenance) should be fixed before this is presented as the paper's
  source-of-truth writeup**, because the walkthrough is explicitly billed as such and two of its
  three supplementary figures (plus the cross-z figure) cannot currently be regenerated from the
  committed code. This is a reproducibility defect, not a correctness defect — fixable in minutes
  by renaming outputs or repointing links — so it does not block the merge of the *diagnostic*,
  only its claim of full emulator-free reproducibility.
