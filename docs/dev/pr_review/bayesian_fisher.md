# PR #6 review — bayesian_fisher lens

Branch: `stage10-multiz-sobolev` → `main`. Reviewer lens: is the Fisher/forecast
logic correct? (`grad_err` definition, linear-vs-log discipline, dropping the
sigma-ratio, statistical soundness of `value_mse` / budget control / h basis test.)

All numbers below were independently re-derived from the committed CSVs and the
code; the two new test files pass (`tests/test_grad_faith_io.py`,
`tests/test_pareto_diag.py`, 5 passed).

## (1) Verdict: APPROVE-WITH-NITS

The Fisher-relevant logic in the *exercised* (single-z) path is correct and, in
several places, a genuine fix of previously-flagged errors. The metric is
Fisher-consistent (linear-P slopes vs a linear-P covariance), the linear/log
discipline is internally consistent, dropping the sigma-ratio is well-justified
(it was a construction-forced tautology), and the budget control + h basis test
are statistically defensible. The nits are: (a) a *latent* log/linear mismatch
in the **multi-z** Sobolev path that this PR does not exercise but still ships
un-fixed, and (b) one over-precise statistical claim in the h basis test. Neither
blocks merge of the diagnostic, but (a) must be fixed before any multi-z Sobolev
run.

## (2) What is correct and well-built

**`grad_err` is the Fisher-consistent quantity, and the relabel is correct.**
`eval_grad_faithfulness.py:110-112` differences `refit.predict` and
`gp.predict`. I verified both return **raw linear P_F**: `Refit1DResult.predict`
applies `exp()` when `log_space=True` (`refit_1d_pysr.py:273-275`), and
`GaussianLikelihood.model_at` returns `model.predict` un-logged
(`likelihood.py:155-165`) and differences it against a **linear-P** KSData
covariance in `_stencil_derivative` (`fisher.py:130-150`). So the Fisher matrix
consumes `∂P_F/∂θ` in linear space, and `grad_err` is a ratio of linear-P slopes
— exactly Fisher-consistent. The earlier `∂logP` mislabel (commit 87d1482) is
genuinely corrected; the walkthrough's "Metric space" note
(`PARETO_FAITHFULNESS_WALKTHROUGH.md:49-54`) is accurate.

**The linear/log discipline is consistent across the three quantities:**
- gate / `grad_err`: linear-P slope ratio (Fisher space). ✓
- `value_mse`: log-P MSE, `mean((logP_eq − logP_GP)²)`
  (`eval_grad_faithfulness.py:113-118`), correctly labeled as log-P everywhere.
- single-z Sobolev loss: trains in normalized-log-P, and the target gradient is
  dimensionally matched. I checked: with `log_space=True` the single-z
  `_build_training_matrix` logs `Y` (`refit_1d_pysr.py:380-387`), the tree's
  FD `grad` is ∂(norm-logP)/∂θ_norm, and `_fidelity_grad_weights`
  (`sobolev_loss.py:59-77`) builds `∂logP/∂θ_phys · width / std_k`
  = ∂(norm-logP)/∂θ_norm. Units agree. ✓

**Dropping the sigma-ratio is justified, not a dodge.** The 4-agent review
(`memory/review_verdict_sr_emulator.md`) established that `σ_perfect_1D ≡ σ_GP`
is forced by the at-fid-anchored additive combine, so the σ-ladder tested a
2-point Jacobian, not an emulator, and the GP-slice fallback leaked GP-derived σ
into the σ_PySR column. Replacing it with `grad_err` (the slope error the
Jacobian actually is) and plotting **with no GP-slice fallback** (a parameter
with no faithful eq shows up all-red) is the honest move. The walkthrough states
this explicitly (lines 19-26).

**Budget control is sound and load-bearing.** I confirmed the budget run is a
genuine *value-loss* run at maxsize=35 (Loss ~0.44, vs Sobolev's ~11) and re-derived
every quoted number from `results/decider_budget_z3.6/.../grad_faith_ns.csv`:
best-loss (cmplx 35) `grad_err=0.319`, `value_mse=3.82e-4`; Sobolev best-loss
(cmplx 18) `grad_err=0.193`, `value_mse=4.74e-4`. The "~24% higher value error"
is 4.74/3.82 = 1.24 ✓. The control correctly isolates *objective* from *budget*:
deeper value search reaches lower value error but never crosses the gate.

**Self-consistency checks all pass:** across all 661 candidate rows in the
committed sidecars, `gate_pass==True ⟺ grad_err<=0.25` with zero violations;
all `value_mse` are finite and non-negative; `n_keep` is 48 (full grid) except
110 rows at 47 (one near-zero-grad bin masked by `floor_frac=1e-3`), so the
median is computed over a near-complete, non-pathological bin set. The
bhfeedback Mirage is real in the data: `value_mse≈3e-5` (best value fit of any
param) with `grad_err≈1.7` — value-accurate, slope-wrong, exactly the failure
mode the diagnostic targets.

## (3) Concrete issues

**NIT-1 (should-fix before any multi-z Sobolev run; latent in this PR).**
The blocking bug the review flagged for Stage 10 is **still present** in the
multi-z path: the multi-z Sobolev loss matches a *log-P* target gradient against
a *linear-P* tree derivative.
- `refit_1d_pysr.py:632-709` (`_build_training_matrix_multiz`) has **no `log()`**
  — verified the whole block contains no `np.log`; the multi-z target `Y` is
  normalized **linear** P_F (only the single-z builder logs `Y`).
- `sobolev_loss.py:115-117` (`_fidelity_grad_weights_multiz`) computes
  `lp_p = np.log(gp.predict(...))`, i.e. the target is `∂logP/∂θ`.
- `make_sobolev_loss` (`sobolev_loss.py:13-36`) finite-differences the *tree*,
  which in multi-z is trained on normalized **linear** P → `grad` is
  ∂(norm-linear-P)/∂θ_norm, mismatched to the log-P `weights`.

This PR's figures are all **single-z** (stage9, `log_space=True`), where the
discipline is correct, and `HANDOFF.md:84-85` confirms the multi-z money plot is
dropped — so the bug is not exercised by any claim here. But the multi-z Sobolev
code path is still reachable via the gated multi-z driver and ships broken.
Fix: either log `Y` in `_build_training_matrix_multiz` (mirroring the single-z
`log_space` branch and threading a `log_space` flag through
`refit_1d_multiz_for_param`), or drop the `np.log` in
`_fidelity_grad_weights_multiz` so target and tree both live in linear-P. The
former is preferred (keeps multi-z consistent with the single-z log target).
Recommend gating multi-z `use_sobolev=True` behind an explicit assert until then.

**NIT-2 (over-precise stat claim; h basis test).**
`scripts/h_basis_test.py:53-58`: the `AP-frac var` column is
`(ap·y)/(y·y)` where `ap = (Tap − mean)·beta[0]` and `beta` is the *partial*
coefficient from a joint 2-feature regression of `y=dP/dh` on `[dP/dlnk, P]`.
Because `dP/dlnk` and `P` are strongly collinear (both smooth in k), `beta[0]`
is the AP slope *controlling for P*, so `(ap·y)/(y·y)` is a "unique AP
contribution," not the clean fraction-of-variance the label and the verdict
("~6% of the dP/dh variance") imply (that equality holds only for orthogonal
predictors). The *qualitative* refutation is nonetheless sound and is what
should carry the weight: `corr(dP/dh, dP/dlnk) ≈ −0.25` is **negative**, so
dP/dh is anti-correlated with the k-rescaling template — a k-rescaling
hypothesis predicts a *strong positive* correlation, which is refuted. Fix:
state the verdict on the correlation (sign + magnitude), and either relabel the
variance column "unique AP contribution (partial)" or compute the honest
single-feature R² = `corr(dP/dh, dP/dlnk)²` (≈0.06, consistent) instead of the
partial-regression projection. Low effort, no change to the conclusion.

**NIT-3 (presentation, not an error).**
`PARETO_FAITHFULNESS_WALKTHROUGH.md:71` — "value@budget reaches the *lowest*
value_mse of any series (3.8×10⁻⁴)" is true *for best-loss points*, but the
money panel colours *all* candidates and the Sobolev front dips to 4.07e-4 at
complexity 13 (vs budget's 3.82e-4 min). The prose correctly frames this as "a
paired comparison, not an equality," so it is internally honest; consider noting
on the figure that the comparison is best-loss-to-best-loss to forestall a
referee eyeballing the near-overlapping minima.

**Minor:** `pareto_diag.py:121-122` has a duplicated `return out_path`
(dead second line). Harmless; delete.

## (4) Anything that blocks merge

Nothing blocks merge of the diagnostic as scoped (single-z). The metric is
Fisher-correct, the discipline is consistent, the dropped sigma-ratio is
well-justified, and every quoted number reproduces from the committed CSVs.

The one item that must be tracked: **NIT-1 (multi-z Sobolev log/linear mismatch)
must be fixed before the multi-z Sobolev path is run for any result**, since it
silently matches gradients in inconsistent spaces and would corrupt a multi-z
forecast. Since this PR does not present any multi-z Sobolev result, it is a
follow-up, not a merge blocker — but it should not be forgotten when multi-z
production resumes (recommend an assert-guard in the interim).
