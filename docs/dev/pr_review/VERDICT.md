# PR #6 — Meta-review VERDICT

> **⚠️ SUPERSEDED IN PART (2026-07-01).** The consensus bullet below (§2, "The metric
> lives in the right space … `grad_err` is a ratio of **linear-P** slopes … correctly
> labelled") is **obsolete**. Since this review, the gate was switched to **log-space**
> (`derivative_gate.py` log-transforms then calls `predict_log`; every production sidecar
> carries `log_space=True`; commit `7aa26af` + `docs/PARETO_FAITHFULNESS_WALKTHROUGH.md:48-57`).
> Log-space is **not** a mislabel and **not** a proxy: because the deployed model is the
> fiducial-anchored additive combine, `∂P_F/∂θ|fid = P_GP(fid)·∂logP_eq/∂θ`, so the anchor
> cancels in the GP ratio and the equation's **log**-slope ratio equals the deployed
> combine's **linear** (Fisher-space) slope ratio — the Fisher-relevant quantity for the
> linear-P_F covariance. This review scored the *un-deployed* standalone equation. The M1
> (figure-generator) and M2 (multi-z Sobolev log/linear guard) must-fixes below still stand.

Branch `stage10-multiz-sobolev` → `main`. Synthesised from four independent lenses:
bayesian_fisher, cs_ml, cosmology, symbolic_regression. All four ran the code and
re-derived numbers from the committed sidecars; this verdict deduplicates and
prioritises their findings.

## (1) Overall verdict

**MERGE-AFTER-NITS** — the diagnostic is correct, reproducible, and emulator-free as
claimed; merge once the figure-provenance gap (M1) and the multi-z log/linear guard
(M2) are closed. No correctness blocker exists in the exercised single-z path.

## (2) Consensus (all four lenses agree)

All four referees independently returned **APPROVE-WITH-NITS** and converge on the
same load-bearing facts:

- **The metric lives in the right space.** `grad_err` is a ratio of **linear-P** slopes
  (`refit.predict`/`gp.predict` both return raw `P_F`, `refit_1d_pysr.py:275` applies
  `np.exp` when `log_space=True`), so it is Fisher-consistent; `value_mse` is log-P.
  Both are correctly labelled in the walkthrough. The earlier `∂logP` mislabel (commit
  87d1482) is genuinely fixed. (all four)
- **The eval reproduces the production gate, it does not re-implement it.**
  `median_rel_error` (`eval_grad_faithfulness.py:38-48`) is a byte-for-byte twin of
  `derivative_gate.derivative_faithful` (`derivative_gate.py:42-59`) — same
  `floor_frac=1e-3` mask, same median ratio, same all-masked→fail edge — and the
  candidate set uses the same `_filter_fisher_safe(df, n_features=3)` as the real
  forecast. (cs_ml, cosmology, symbolic_regression)
- **Dropping the sigma-ratio is justified, not a dodge.** σ_perfect_1D ≡ σ_GP was a
  construction-forced tautology (4-agent review); replacing it with `grad_err` and
  plotting with no GP-slice fallback is the honest move. (bayesian_fisher, corroborated
  by the others' metric checks)
- **The budget control is a fair search-starvation test.** `decider_budget_z3.6`
  genuinely reaches complexity 35 at value-loss (`ns_calib.log`: loss 0.4419); best
  `grad_err` over the whole 13→35 front is 0.319 (fails the 0.25 gate) while Sobolev
  passes at 0.193 on a *smaller* budget. The "objective, not budget" conclusion is in
  the raw CSV, not the plotting. (all four; numbers reproduce exactly)
- **The taxonomy / Mirage is real in the data, not a plotting artifact.** Low value
  error with bad slope (bhfeedback `value_mse≈3e-5`, `grad_err≈1.7`) is present in the
  committed sidecars; the Fisher-safe filter runs before either metric so it cannot
  manufacture the value/derivative decoupling. (bayesian_fisher, symbolic_regression)
- **The h-basis test is a valid AP refutation and reproduces live.** `dP/dlnk` is the
  correct k-rescaling template, P is included to catch amplitude modes; corr(dP/dh,
  dP/dlnk) ≈ −0.25 (negative) and ≈6% variance reproduce, and the conclusion holds even
  with the P nuisance dropped (AP-only R² ≈ 0.04–0.07). (cosmology, bayesian_fisher,
  symbolic_regression)
- **Tests are real and pass.** `test_grad_faith_io.py` + `test_pareto_diag.py` = 5
  passed (x0 word-boundary, comment-skip, bool round-trip, gray-fallback, left-join
  NaN); stage10 suite green. Diagnostic modules import with `student_projects` stripped
  from `sys.path` → emulator-free contract holds. (cs_ml, symbolic_regression)
- **The single-z science is sound.** Every z=3.6 table number, the cross-z sweep, and
  the physics taxonomy (under-determined hub/omegamh2, priored-out bhfeedback, He II
  reion epochs, z=4.2 emulator-limited caveat) all reproduce and are internally
  consistent with `PARAMS_11D`. (cosmology)

## (3) Prioritised, deduplicated action list

### MUST-FIX before merge

- **M1 — Three of four walkthrough figures have no committed generator (provenance).**
  `docs/PARETO_FAITHFULNESS_WALKTHROUGH.md` embeds `summary_scorecard.png` (L81),
  `ns_money_panel.png` (L86), and `crossz_faithfulness.png` (L230), but the only
  committed figure script `scripts/make_diagnostic_figs.py` writes the *differently
  named* `faithfulness_scorecard.png` (L113) and `ns_budget_panel.png` (L145) and never
  emits a cross-z panel (verified: `grep` finds no generator for the embedded names).
  Only `pareto_faithfulness.png` (L29) is regenerable. This directly undercuts the PR's
  "figures regenerate emulator-free" claim for the document billed as the paper's
  source-of-truth. *Fix:* either rename the script outputs to match the embedded names
  (`faithfulness_scorecard`→`summary_scorecard`, `ns_budget_panel`→`ns_money_panel`) and
  add the cross-z generator, or repoint the walkthrough `![]()` links to the names the
  script actually emits. Minutes of work; no diagnostic-code change.
  *Raised by:* cs_ml (#1), cosmology (#1), symbolic_regression (N-adjacent) — **3 of 4
  lenses, the single most-cited issue.**

- **M2 — Latent multi-z Sobolev log/linear mismatch ships un-guarded (Fisher
  correctness).** The multi-z path matches a *log-P* target gradient against a
  *linear-P* tree derivative: `_build_training_matrix_multiz`
  (`refit_1d_pysr.py:632-709`) has no `np.log` (multi-z `Y` is normalized **linear**
  P_F, only the single-z builder logs `Y`), while `_fidelity_grad_weights_multiz`
  (`sobolev_loss.py:115-117`) builds the target as `np.log(gp.predict(...))`. This PR
  exercises only single-z (`log_space=True`, where the discipline is correct) and the
  multi-z money plot is dropped (`HANDOFF.md:84-85`), so it is **not exercised by any
  claim here** — but the multi-z Sobolev path is reachable via the gated driver and
  would silently corrupt a multi-z forecast. *Fix before merge:* gate multi-z
  `use_sobolev=True` behind an explicit `assert`/raise (cheap, prevents accidental
  corruption); the real fix (log `Y` in `_build_training_matrix_multiz`, threading a
  `log_space` flag through `refit_1d_multiz_for_param`) is the follow-up.
  *Raised by:* bayesian_fisher (NIT-1, "must be fixed before any multi-z Sobolev run").

### NICE-TO-HAVE (non-blocking)

- **N1 — Silent k-grid coupling in eval and the production gate (correctness footgun).**
  `eval_grad_faithfulness.py` computes `target` on `k_grid = kodiaq_k_grid(kmin,kmax,48)`
  (L68/77) but the candidate/`value_mse` use the stored grid `kg = d["kfkms_lf_z"][0]`
  (L83); `median_rel_error` divides elementwise. Identical only at default bounds
  (verified max-diff 0.0); non-default `--kmin/--kmax` silently misaligns. Same coupling
  lives in the production gated builder (`pipeline.py:285` vs `forecast.py:224`).
  *Fix:* evaluate `target` on `kg`, or `assert np.allclose(k_grid, kg)`.
  *Raised by:* symbolic_regression (N1), cs_ml (#4) — 2 lenses; shipped CSVs valid, so
  not a blocker, but the highest-value soon-fix.

- **N2 — Budget-control pysr provenance not archived.** Fairness rests on "only maxsize
  20→35"; the complexity-35 front + `ns_calib.log` corroborate it, but no
  `pysr_kwargs.json`/manifest is committed, so a referee can't confirm
  niterations/populations were held fixed. *Fix:* commit the CLI / `pysr_kwargs.json`
  next to `pareto_ns.csv`, or quote the full command in the walkthrough.
  *Raised by:* symbolic_regression (N2).

- **N3 — Dead code: duplicate `return out_path`.** `src/priya_forecast/pareto_diag.py:122`
  is unreachable (verified). Delete it.
  *Raised by:* bayesian_fisher, cs_ml (#2), cosmology (#2) — trivial.

- **N4 — Dead/misleading `bf` in scorecard.** `scripts/make_diagnostic_figs.py:105`
  `bf = bestloss(SOBOLEV, p)` is assigned, never used, re-fetches best-*loss* under a
  "best-faith" comment (verified). Delete or compute the real best-faith.
  *Raised by:* cs_ml (#3), cosmology (#3), symbolic_regression (N4) — 3 lenses, trivial.

- **N5 — h-basis "AP-frac var" is a partial-regression statistic, not plain R².**
  `h_basis_test.py:53-58` reports `(ap·y)/(y·y)` from a joint 2-feature lstsq — "unique
  AP contribution after partialling out P," not clean R² (equal only for orthogonal
  predictors; here corr²≈0.06 happens to agree). Conclusion unchanged. *Fix:* relabel
  "AP-unique variance (partial)" or report single-feature corr² ≈ 0.06, and state the
  verdict on the (negative) correlation sign+magnitude.
  *Raised by:* bayesian_fisher (NIT-2), symbolic_regression (N5).

- **N6 — Cross-space transfer is empirical, state it.** The Sobolev loss matches
  normalised-log-P slopes while the gate scores linear-P slopes (related by 1/P); the
  ns result (0.60→0.19) shows the transfer works but the walkthrough doesn't say it is
  empirical, not exact. One sentence. *Raised by:* cosmology (#4).

- **N7 — Money-panel near-overlapping minima.** "value@budget reaches the lowest
  value_mse of any series" is true for best-loss points; the Sobolev front dips to
  4.07e-4 (vs budget 3.82e-4). Prose is honest; add a "best-loss-to-best-loss" note on
  the figure. *Raised by:* bayesian_fisher (NIT-3).

- **N8 — Misc minor:** unused joined `gate_pass` column in the plotter (second source of
  truth, cs_ml #5); Sobolev forward-diff h=1e-4 vs gate central-diff h=1e-3 — benign,
  one design-doc sentence (symbolic_regression N3); hardcoded `/home/mfho/student_projects`
  PYTHONPATH in `make_grad_faith_sidecars.sh` — non-portable, env-default it (cs_ml #6);
  herei×alphaq coupling forward-pointer to the multi-D combine (cosmology #5).

## (4) Recommendation to the author

This is a strong, honest PR: the diagnostic does exactly what it claims, the metric is
in the Fisher-consistent space, the gate reproduction is byte-exact, and four
independent reviewers re-derived every load-bearing number from the committed
sidecars without an emulator. Merge it after two cheap fixes. First, close the
figure-provenance gap (M1) — rename `make_diagnostic_figs.py`'s outputs to the embedded
names and add the cross-z generator, or repoint the walkthrough links — so the document
billed as the paper's source-of-truth is actually regenerable end-to-end. Second, drop
an `assert`/raise guard on multi-z `use_sobolev=True` (M2) so the known log/linear
mismatch in the un-exercised multi-z path can't silently corrupt a future forecast; the
proper log-`Y` fix is a tracked follow-up, not a merge blocker. Fold in the trivial
dead-code deletions (N3, N4) while you're there, and add the k-grid `np.allclose` guard
(N1) before anyone re-runs the sidecars with non-default bounds. None of this touches
the correct, tested diagnostic core.

---

### Executive summary
1. **VERDICT: MERGE-AFTER-NITS** — diagnostic is correct, Fisher-consistent, reproducible, emulator-free; all 4 lenses APPROVE-WITH-NITS, no correctness blocker in the single-z path.
2. **M1 (must-fix, 3 of 4 lenses):** 3 of 4 walkthrough figures (`summary_scorecard`, `ns_money_panel`, `crossz_faithfulness`) have no committed generator — `make_diagnostic_figs.py` writes different names. Rename outputs / add cross-z gen, or repoint links.
3. **M2 (must-fix, Fisher):** latent multi-z Sobolev log/linear mismatch (`refit_1d_pysr.py:632-709` linear-Y vs `sobolev_loss.py:115-117` log target) ships un-guarded; add an `assert` on multi-z `use_sobolev=True` now, log-`Y` fix as follow-up.
4. **N1 (top nice-to-have, 2 lenses):** silent k-grid coupling (`eval_grad_faithfulness.py` L77 `k_grid` vs L83 `kg`, also in the production gate) — add `np.allclose` guard before non-default `--kmin/--kmax` reruns.
5. Trivial cleanups: duplicate `return out_path` (`pareto_diag.py:122`), dead `bf` (`make_diagnostic_figs.py:105`); doc nits: h-basis partial-R² relabel, empirical cross-space-transfer sentence, archived budget pysr manifest.
6. **Recommendation: APPROVE and merge after M1 + M2; everything else is follow-up.**
