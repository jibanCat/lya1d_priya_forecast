# Stage 9 — Sobolev derivative-matching loss for PySR refits

**Date:** 2026-06-04
**Branch:** `stage9-sobolev-loss`
**Status:** design approved, pending spec review

## 1. Background and motivation

Stage 8 established that the `aq` operator + derivative-validation gate are a
**filter, not a generator**: they reject derivative-unfaithful equations but
cannot make the PySR search *produce* faithful ones. For key parameters (ns,
hub) the value-loss search never produced a gradient-faithful equation, so the
gate dropped them to GP-slice — a failure for a headline science parameter.

The ratio-response target (lever #1) is an *indirect* generative fix (reshape
the target so a good value-fit tends to imply a good gradient); a spike showed
it fixes 8–9/11 params but not hub. The **direct** fix is a **Sobolev loss**:
penalize the derivative error `[∂_θ logP_SR − ∂_θ logP_GP]²` *during* the genetic
search, so the search optimizes the Fisher quantity itself. (Sobolev training is
standard in PINN/ML but, per the SR-cosmology literature, novel here — see
`docs/SR_EMULATOR_LITERATURE_NOTES.md`.)

## 2. Feasibility (spike-confirmed)

A spike ran a custom PySR `loss_function` end-to-end on this install (PySR
1.5.10 + the project Julia env). The mechanism:
- the candidate tree's derivative `∂eq/∂θ` is **finite-differenced inside the
  loss** (evaluate the tree at `X` and at `X` shifted by `+h` in the θ-row),
  using `eval_tree_array` twice — no `eval_grad_tree_array` needed (the
  version-fragility risk the old HANDOFF flagged is thereby avoided);
- the GP **target gradient** `∂logP_GP/∂θ` (one scalar per training point) is
  passed to PySR via the per-point **`weights`** channel.

Both work. The loss runs and a fit completes.

## 3. Approach

**Architecture is unchanged** — per-parameter 1D refits, PySR inputs
`(θ_norm, k_norm, resolution[, z_norm])`, the additive-Taylor combine, and the
Fisher forecast are all untouched. The change is confined to **the PySR loss and
the training data fed to one refit**.

### 3.1 The Sobolev loss (`src/priya_forecast/sobolev_loss.py`)

A custom Julia `loss_function` string, mirroring the plumbing of the existing
`dim_balanced_loss.JULIA_LOSS_FUNCTION`:

```
L(tree, dataset, options) =
      mean( (eq(X)            − y)        ^2 )      # value term (MSE)
  + λ· mean( (eq_grad_x0(X)   − weights)  ^2 )      # Sobolev term
```
where `eq_grad_x0 = (eq(X + h·e0) − eq(X)) / h` is the in-loss finite-diff θ
derivative, `y` is the (normalized) `logP` target, and `weights[i]` is the GP
target gradient at point `i`. `λ` is injected as a constant when the loss string
is built (`make_sobolev_loss(lam, h)` returns the Julia string).

### 3.2 The target gradient (per-fidelity)

`weights[i] = ∂logP/∂θ` at training point `i`, computed once from the GP via
finite difference (the existing `derivative_gate.gp_param_gradient` pattern).

**Critical (the lever-#1 lesson):** training rows are multi-fidelity (LF + HF).
Each row's target gradient must use **its own fidelity** — `∂logP_LF/∂θ` for LF
rows, `∂logP_HF/∂θ` for HF rows — NOT a single cross-fidelity gradient. The
training-matrix builder already tracks which rows are LF vs HF
(`fidelity_arrays`); the weights are assembled per-fidelity accordingly.

**Normalization consistency (essential):** the loss's `y` is the *normalized*
target `(logP − mean_k)/std_k`, so `eq` predicts normalized logP and the in-loss
`eq_grad_x0` is `∂(normalized logP)/∂θ`. The target gradient must therefore be in
the SAME normalized space: `weights[i] = (∂logP_GP/∂θ at point i) / std_k(i)`.
Additionally, the in-loss finite-diff is taken w.r.t. the *normalized* input
`x0 = θ_norm`, so the chain-rule factor `∂θ_phys/∂θ_norm = (x_param_max −
x_param_min)` must be applied consistently to both `eq_grad` and `weights` (or
cancel — simplest is to define the target as `∂(normalized logP)/∂x0` directly,
i.e. `(∂logP_GP/∂θ_phys)·(x_param_max−x_param_min)/std_k`). The plan pins one
convention and tests it.

### 3.3 Config + wiring

- `PySRConfig` (or the refit knobs) gains `use_sobolev: bool = False` and
  `sobolev_lambda: float = 1.0`.
- `refit_1d_for_param`: when `use_sobolev`, build the per-row target-gradient
  array, pass it as PySR `weights`, and set `loss_function = make_sobolev_loss(
  sobolev_lambda, h)` (dropping `elementwise_loss`, as the ANOVA path does).
- `refit_one_param_single_z` / the SLURM/CLI thread the flag through (mirrors the
  existing `--target-space` plumbing).

### 3.4 Reuse from Stage 8

- `derivative_gate` **validates** Sobolev-trained equations (does each param pass
  at low tol?). The gate stays in selection as the safety net.
- The `aq` operator stays (pole-free division is strictly better for
  derivatives).

## 4. Evaluation

Success is measured on **gradient faithfulness per parameter**, NOT the
single-z σ_PySR/σ_GP metric (rank-deficiency-confounded — Stage 8):

1. Per-param `median_k |∂eq/∂θ ÷ ∂logP_GP/∂θ − 1|` at fid (the gate metric).
   **Primary target: ns AND hub now pass** (the Stage 8 stragglers), and no
   regression on the params that already passed.
2. The well-conditioned **multi-z** σ_PySR/σ_GP after mirroring (secondary).

A `results/single_z_stage9/COMPARISON.md` reports per-param gradient error
(value-loss vs ratio-response spike vs Sobolev) and the gate pass-count.

## 5. Testing (TDD)

1. `make_sobolev_loss(lam, h)` returns a Julia string containing the value term,
   the finite-diff θ-derivative, the `weights` reference, and the injected `lam`.
2. Per-fidelity target-gradient assembly: on a stub multi-fidelity payload, the
   weights for LF rows use the LF gradient and HF rows the HF gradient (shapes +
   values).
3. A gated (`RUN_SLOW_REFIT`) end-to-end: refit ns with `use_sobolev=True` on
   real KODIAQ and assert the resulting equation passes `derivative_faithful` at
   tol 0.25 (the thing the value-loss could not achieve).
4. No-regression: `use_sobolev=False` leaves the existing refit byte-identical.

## 6. Risks and watch-items

- **λ tuning.** `λ=1` is a guess; too small → no derivative pull, too large →
  value accuracy suffers. Tune empirically on ns/hub gradient error; expose as a
  config knob. Possibly a brief λ-sweep in the plan.
- **Loss cost ≈ 2×** (two tree evals per loss call). Acceptable — runs on SLURM;
  watch wall-time per param.
- **Per-fidelity gradient correctness** — the exact failure mode that broke
  lever #1; covered by test #2 and the gated ns end-to-end.
- **Finite-diff step `h`** in the loss: too small → cancellation noise in the
  normalized space, too large → bias. Default `h≈1e-4` (spike value); expose if
  needed.
- **PySR `weights` semantics:** we hijack `weights` for the target gradient (the
  custom loss ignores them as MSE weights). Documented; no other code path uses
  weights in the refit.

## 7. Out of scope

- Multi-z mirror of the Sobolev refit (after single-z ns/hub success).
- `eval_grad_tree_array` (finite-diff-in-loss is the chosen, robust path).
- The stashed lever-#1 (`log_ratio`) plumbing — left stashed; Sobolev supersedes
  it as the generative fix. Revisit only if Sobolev underperforms.
