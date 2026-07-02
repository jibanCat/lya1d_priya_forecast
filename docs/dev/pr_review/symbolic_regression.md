# PR #6 review — symbolic_regression lens

Branch: `stage10-multiz-sobolev` → `main`. Reviewer lens: SR specifics —
grad_err correctness, Pareto-front handling, Fisher-safe filter, best-loss vs
best-faith convention, budget-control fairness, taxonomy-as-artifact risk.

## (1) Verdict: **APPROVE-WITH-NITS**

The diagnostic is correctly built and, critically, **reproducible**: I re-ran
`eval_grad_faithfulness.py` for ns (stage9 Sobolev) and `h_basis_test.py` and got
byte-identical numbers to the shipped sidecars and the walkthrough claims. The
grad_err metric is computed in the right space, the Fisher-safe filter and
best-loss selection exactly mirror the production gate, the budget control is a
genuine larger-maxsize run (front reaches complexity 35), and the value/derivative
decoupling that underpins the taxonomy is real — not a re-labeling of value error.
The nits below are documentation/provenance, not correctness blockers.

## (2) What is correct and well-built

**grad_err is in the right space and matches the gate.**
`gp_param_gradient` (derivative_gate.py:15) calls `gp.predict` with no log, and
`equation_param_gradient` (derivative_gate.py:29) calls `refit.predict`, which
returns linear `P_F` (`np.exp` when `log_space=True`, refit_1d_pysr.py:275). So
`grad_err = median_k |∂P_eq/∂θ ÷ ∂P_GP/∂θ − 1|` is a ratio of **linear-P** slopes
— Fisher-consistent, matching the corrected walkthrough label (commit 87d1482).
`value_mse` correctly uses `predict_log` vs `np.log(gp.predict)` (log-P). The two
spaces are not confused.

**`gate_pass` reproduces the production gate exactly.** `median_rel_error`
(eval_grad_faithfulness.py:38) is line-for-line identical to
`derivative_faithful` (derivative_gate.py:42): same `floor_frac=1e-3` mask, same
`median(|cand/target − 1|)`, same all-masked → fail edge case (returns
`inf <= tol` → False). And `_filter_fisher_safe(df, n_features=3)` is the *same*
filter the gated builder uses (forecast.py:215). So the sidecar's pass/fail column
is the production verdict, not a re-implementation that could drift.

**Best-loss convention is consistent everywhere.** The eval sorts
`safe.sort_values("Loss")` and prints `best_loss = rows[0]` (ascending loss) plus a
separate `best_faith = min(..., key=grad_err)` (eval:138-139). The gated builder
selects in the same ascending-loss order (forecast.py:220). The figures' `bestloss`
helper (make_diagnostic_figs.py:37) re-sorts by `Loss` and takes `iloc[0]`. The
walkthrough taxonomy table is explicitly the best-loss column with best-faith noted
in parens (commit 87d1482 fixed an earlier mix). Verified.

**Budget control is a fair search-starvation test.** `decider_budget_z3.6`
genuinely searched a larger budget: its `pareto_ns.csv` reaches complexity 35 (vs
stage9 Sobolev max 18, stage6 value max 19), and `ns_calib.log` records
`eq complexity=35, loss=0.4419`. Best grad_err over the whole 13→35 front is 0.319
— still failing the 0.25 gate — while Sobolev at a *smaller* budget passes (0.193).
This grants the value objective MORE freedom and shows the failure persists, which
is the right shape for "objective, not budget."

**The taxonomy is not a pipeline artifact.** The load-bearing claim — value
accuracy ⇏ derivative faithfulness — is visible in the raw CSVs: budget ns reaches
the *lowest* value_mse of any series (3.8e-4) yet stays red (grad_err 0.319), while
Sobolev passes (0.193) at a slightly *higher* value_mse (4.7e-4). Low value error
with bad slope is exactly the Mirage, and it is in the data, not the plotting. The
Fisher-safe filter only removes numerically invalid candidates (x0-free,
|const|>100, stencil-blowup) *before* either metric is computed, so it cannot
manufacture the decoupling.

**hub "x0@20" claim is data-backed.** I checked stage6 `pareto_hub.csv`: x0 first
appears only at complexity 20 (the maxsize), the single Fisher-safe candidate has
grad_err 1.0004, and the sidecar has exactly one row. The "under-search / weak
signal" diagnosis is real.

**h basis test REFUTES AP — reproduced.** `h_basis_test.py` reran to
corr(dPdh, dPdlnk) = −0.21/−0.25/−0.26 and AP-frac-var ≈ 0.061/0.062/0.068 at
z=2.6/3.6/4.2, matching the walkthrough's "≈ −0.25, ~6%." The refutation is honest
and reproducible.

**Tests pass.** `test_grad_faith_io.py` + `test_pareto_diag.py` (5) and the
stage10 suite (16) all green. The x0 word-boundary test (x01 not matched) and the
sidecar bool round-trip are good targeted tests.

## (3) Concrete issues (file:line + fix)

**N1 — Latent k-grid coupling is silent and undocumented (eval + production
gate).** In `eval_grad_faithfulness.py` the GP target is on
`k_grid = kodiaq_k_grid(args.kmin, args.kmax, 48)` (geomspace) (line 68/77), but
the candidate gradient and value_mse use `kg = d["kfkms_lf_z"][0]` (the *stored
1pvar* grid) (line 81/95). `median_rel_error` then divides `g/target`
**elementwise** assuming the two grids coincide. They DO coincide at the default
bounds — I verified `max|geomspace(0.001,0.04,48) − stored_1pvar_k| == 0.0` — but
only because regen_1pvar.py uses the identical defaults. Pass `--kmin/--kmax` (or
point at a regen with different bounds) and you silently compare mismatched k-bins,
or crash on a shape mismatch. The *production* gated builder has the exact same
coupling (`tgt` on `k_refit` at pipeline.py:285 vs candidate on `d["kfkms_lf_z"][0]`
at forecast.py:224). Fix: in `eval_grad_faithfulness.py`, evaluate the GP target on
`kg` (the same grid the candidate uses), e.g. compute `target` *after* loading
`kg` and pass `k_grid=kg`; or assert `np.allclose(k_grid, kg)` and error out
otherwise. At minimum, add a one-line comment at line 68 and a guard. (Shipped CSVs
are valid; this is a footgun for anyone re-running with non-default bounds.)

**N2 — Budget control's full pysr provenance is not archived
(decider_budget_z3.6).** The fairness argument is "only maxsize changed (20→35)."
The complexity-35 front and `ns_calib.log` corroborate the larger maxsize, but the
log does not print niterations/populations/operators, and no run manifest /
pysr_kwargs is committed. A referee can't verify that niterations/populations were
held fixed. Fix: commit the exact CLI / a `pysr_kwargs.json` next to
`pareto_ns.csv`, or quote the full command in the walkthrough's budget section.
Cheap, and it closes the one remaining "is the control actually controlled?" gap.

**N3 — Sobolev loss uses forward diff h=1e-4; gate/eval use central diff h=1e-3
(sobolev_loss.py:13, derivative_gate.py:16/31).** The training loss estimates the
gradient with a one-sided difference at a different step than the arbiter gate.
Benign (the loss only needs a cheap in-search gradient; the central-diff gate is
the judge), but it means "the thing Sobolev optimizes" and "the thing the gate
measures" are not the identical operator. Worth one sentence in the design doc so
it isn't read as an inconsistency. No code change required.

**N4 — Dead variable in scorecard (make_diagnostic_figs.py:105).**
`bf = bestloss(SOBOLEV, p)` is computed and never used; the comment says "note
best-faith in text" but it re-fetches the best-*loss* value and discards it.
Remove the line (or actually compute/annotate best-faith if that was the intent).

**N5 — AP-frac-var is a partial-regression statistic, not plain R²
(h_basis_test.py:53-58).** "~6% variance" comes from a 2-regressor lstsq
(`X = [Tap, P]`) reporting only the AP component's `(ap@y)/(y@y)`. Defensible (it
isolates the AP contribution after allowing a P-amplitude term), but a reader will
assume simple R². corr ≈ −0.25 already gives corr² ≈ 0.06, which happens to agree —
so just state "corr² ≈ 0.06, and AP-unique variance ≈ 6% after partialling out P"
to pre-empt the question. Docs-only.

## (4) Blocking issues

**None.** Nothing here blocks merge. The diagnostic's correctness, the
Fisher-consistent metric space, the gate-faithful reproduction, the budget
control, and the AP refutation all hold up under independent re-execution. N1 is
the most important to address soon (it's a correctness footgun for re-runs, and it
also lurks in the production gate), but with the default bounds the committed
artifacts are correct.
