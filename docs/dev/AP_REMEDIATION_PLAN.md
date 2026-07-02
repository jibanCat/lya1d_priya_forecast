# Ap remediation plan — fixing the gradient-at-fid mismatch

> **Status**: pending. Not in PR #2. Plan doc only — implementation
> deferred until after PR #2 merges and paper draft starts.
>
> See `PAPER_NOTES.md § D8.6` for the Ap regression context (Phase 1
> 0.79× σ_GP → Phase 2 2.62× σ_GP) and the abandoned strategies.

## Problem statement

Phase 2 production landed with `Ap σ_PySR/σ_GP = 2.62×` at fid (Ap eq has
the right *function values* — mean LF rel-err 1.87% — but the wrong
*local slope at fid*: 0.276 vs the GP's true ∂P_F/∂θ_Ap_norm). PySR's
training loss penalizes function values across the Sobol prior cube
(global rel-err); the slope at the specific point fid is decoupled from
the loss.

**Ruled out by user**:
- Post-hoc rescale (multiply contribution by `c = target_grad / current_grad`):
  rejected as cheating because if applied to all params, σ_PySR = σ_GP by
  construction → trivializes the Fisher comparison.

**Ruled out by smoke test**:
- Log-target training (smoke at `scripts/smoke/refit_ap_log_target_smoke.py`):
  better off-fid Lipschitz (max LF 19.4% vs 23.8%) but slope-at-fid 13×
  steeper → would give σ_PySR/σ_GP ≈ 0.20× (overshoot in opposite direction).

## Two principled fixes (pick one for Phase 3)

### Option A: Grad-matching loss term

Extend `JULIA_LOSS_FUNCTION_ANOVA` in
`src/priya_forecast/dim_balanced_loss.py` to add a slope-matching
penalty:

```julia
function dim_balanced_loss_with_grad(prediction, X, y, target_grad, lambda)
    base = ANOVA_loss(prediction, X, y)
    # Compute the eq's gradient at the fid point via finite-difference.
    fid_idx = first_row_at_theta_norm_eq_fid_norm(X)  # precomputed
    h = small_step
    grad_at_fid = (prediction[fid_idx + h_dx0] - prediction[fid_idx - h_dx0]) / (2*h)
    # Slope-matching penalty.
    return base + lambda * (grad_at_fid - target_grad)^2
end
```

**`target_grad`**: precomputed once per param via finite-diff of the GP
at fid. Stored in the param's payload. Specifically:

```python
target_grad = (
    LF_GP.predict(theta_fid + h*e_i, k_grid, z) -
    LF_GP.predict(theta_fid - h*e_i, k_grid, z)
) / (2 * h * (param_max - param_min))    # in normalized-θ slope units
```

**`lambda`**: hyperparameter. Start at λ = 1 (equal weight to global
function and local slope); tune up/down based on smoke results.

**Implementation steps**:

1. Add a Julia callback to PySR's `loss_function` that:
   a. Receives `prediction, X, y` (full batch).
   b. Identifies which rows correspond to fid (via a precomputed mask).
   c. Approximates the gradient at fid via finite-difference along x0.
   d. Adds `lambda * (grad - target_grad)²` to the ANOVA base loss.
2. Threading consideration: PySR's `parallelism="multithreading"` runs
   evals in parallel; need atomic atomic gradient accumulation. Or run
   `parallelism="serial"` for grad-matching fits (slower but simpler).
3. Pareto-pick: existing filters still apply. Add a "slope sanity"
   filter that rejects eqs whose gradient at fid differs from
   target_grad by > 5×.
4. Test: grad-matching smoke for Ap with lambda=1 first; check σ_Ap drops
   into [0.7×, 1.5×] σ_GP range.

**Cost**: ~half day to wire + 1 hour smoke. Risk: lambda tuning may
require iteration.

**Risks**:
- Grad finite-diff requires perturbing a fid row; if the Sobol training
  set doesn't contain a row at exactly θ=fid, need to construct one
  (cheap: 1 extra row in training matrix).
- Multithreading + Julia atomic ops can be tricky; `parallelism=serial`
  is the safer fallback (~3-5× slower but reproducible).

**Why this works**: PySR's genetic search optimizes the loss; the loss
now penalizes both global rel-err AND local slope; the eqs that survive
on the Pareto front are *both* faithful across the cube *and* have the
right slope at fid. The Fisher gradient becomes a *learned* property
honestly.

### Option B: Split-learning eq family via PySR `TemplateExpressionSpec`

PySR ≥ 1.x supports `TemplateExpressionSpec`
(see `https://github.com/MilesCranmer/PySR/discussions/787`) which lets
you fix the OUTER form of the eq and search inside sub-expressions.

Force the eq to take the form

```
eq(x0, x1, x2, x3) = c * x0 + g(x0, x1, x2, x3)
```

with `c` a fitted scalar coefficient and `g` discouraged from any
linear-in-x0 component. Specifically:

```python
spec = TemplateExpressionSpec(
    function_symbols=["g"],
    combine="c_linear * x0 + g(x0, x1, x2, x3)",
)
# c_linear is a free scalar in PySR's expression tree.
```

By construction the slope-at-fid for this eq equals `c_linear` (when
all other terms have zero linear-in-x0 component, which we enforce via
penalty on `g`'s linear term).

**`c_linear` matching**: at training time, PySR optimizes `c_linear`
along with `g` to minimize the existing ANOVA loss. The slope-at-fid is
then automatically `c_linear` by structure. We can additionally:
- Pin `c_linear = target_grad` as an initial value; PySR converges
  around it.
- OR add a soft penalty `(c_linear - target_grad)²` to bias the search.
- OR leave it free and check post-hoc whether `c_linear ≈ target_grad`.

**Implementation steps**:

1. Verify `pysr.TemplateExpressionSpec` is available in the cluster's
   PySR version (`pip show pysr | grep -i version`).
2. Modify `SMART_REFIT_PYSR_KWARGS` to accept a `template_expression_spec`
   when fitting Ap (or any param with a known linear-leading-term
   structure).
3. The same recipe could apply to other amplitude-like params — `tau0`
   could have a similar leading linear term.

**Cost**: ~half day. Risk: depends on TemplateExpressionSpec API stability.

**Why this works**: forcing `eq = c·x0 + ...` makes the linear-in-θ slope
explicit. PySR can fit it as a scalar (and we constrain it to match the
target gradient via initial value or soft penalty). The rest of the eq
captures the higher-order structure.

## Comparison

| | Option A (grad-matching loss) | Option B (split-learning) |
|---|---|---|
| **principled** | yes (gradient enters loss) | yes (gradient is structural) |
| **Julia changes** | yes (custom loss callback) | none (uses upstream API) |
| **PySR version** | any | needs PySR ≥ 1.x |
| **runtime overhead** | ~10% (extra grad finite-diff per eval) | none significant |
| **flexibility** | applies to any param via lambda tuning | requires designing the eq family |
| **cost** | ~half day + 1h smoke | ~half day + 1h smoke |

## Decision criteria (for whichever we pick)

A successful Phase 3 Ap fix must:

1. Land σ_Ap/σ_GP at fid in `[0.7×, 1.5×]`.
2. Multi-D Sobol hold-out p99 ≤ Phase 2's 5.15% (no regression on
   off-fid Lipschitz).
3. At-fid identity preserved (`hybrid.predict(fid) ≈ GP(fid)`).
4. Eq passes all 3 Pareto filters (well-behaved, stencil-safe, no
   pathological constants).

If both options succeed, prefer **Option A** as a global fix (applies
to all params via the loss; could improve `herei`, `heref`, `alphaq`
too) over Option B (per-param eq family change).

## What gets paper'd

Independent of which option we pick, the paper should:

- Report Phase 2 σ-ratios honestly (PR #2 production scorecard).
- Note Ap as a known limitation from D8.6.
- Reference Option A or B as the principled future fix; cite as an
  open subproblem for follow-up if Phase 3 doesn't land before submission.

## Timeline

1. Pick Option A or Option B (user decision).
2. Smoke test for Ap (~1 h compute + ~half day wiring).
3. If successful: re-fit Ap (and other PySR-routed params) on SLURM (~10 min wall).
4. Re-aggregate Phase 3 + multi-D hold-out + closure (~15 min).
5. Update `PAPER_NOTES § D8.6` with Phase 3 result + replace D8.6's "ruled out" entries.
6. PR Phase 3 (smaller scope than Phase 2: just per-1D rerun + new loss/template).

Estimated total: 1 day if Option A; 1-1.5 days if Option B (depends on
TemplateExpressionSpec API).
