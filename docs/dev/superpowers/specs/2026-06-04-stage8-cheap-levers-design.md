# Stage 8 (cheap levers) — `aq` operator + derivative-validation gate

**Date:** 2026-06-04
**Branch:** `stage8-sobolev-derivative-loss`
**Status:** design approved, pending spec review

## 1. Background and motivation

The PySR surrogate is value-accurate but **derivative-unfaithful** ("Fisher's
Mirage", arXiv:2406.06067): σ_PySR/σ_GP spans physically-impossible sub-1
ratios because Fisher is a pure derivative object and is acutely sensitive to
derivative noise. Stage 6 (`log(P)` target) attenuated the severity; Stage 7
confirmed it **persists** at multi-z (mean |log10(σ_PySR/σ_GP)| ≈ 0.35, and 3
params — ns/bhfeedback/dtau0 — produced no Fisher-safe equation at all).

The SR-cosmology literature (syren family: arXiv:2311.15865, 2506.08783,
2510.18749; full notes in `docs/SR_EMULATOR_LITERATURE_NOTES.md`) never
validates derivative accuracy or uses a derivative-matching loss. It achieves
smooth derivatives **indirectly**, via two recipes we are missing:

- **analytic-quotient operator** `aq(x,y)=x/√(1+y²)` (bounded, pole-free)
  instead of raw division, which the literature adopts "for numerical
  stability";
- parsimony + selection discipline (reject train/val gap; manual Pareto pick).

Stage 8 (this spec) implements those two cheap, well-trodden levers and adds a
**derivative-validation selection gate** (novel here, cheap to build on
existing scaffolding). The novel/risky Sobolev *training* loss is **deferred**
— we measure the Mirage reduction from the cheap levers first.

## 2. Scope

In scope:
- **Lever #2 — `aq` operator** (replace raw `/`).
- **Lever #3 — derivative-validation selection gate.**
- A shared **custom-operator registry** + `sympify_equation` helper (the
  round-trip fix the feasibility spike identified).
- Applied **single-z first** (fast test bed vs the Stage 6 z=3.6 baseline);
  multi-z mirrors after, once single-z shows a win.

Out of scope (deferred):
- **Lever #1 — ratio-response target** `log[P(θ)/P(θ_fid)]` (feasible but the
  largest surface; revisit if the Mirage persists after #2+#3).
- **Sobolev training loss** (the novel, risky Julia `loss_function` path).

## 3. Design

### 3.1 Custom-operator registry (shared foundation)

New module `src/priya_forecast/custom_operators.py`:

- `CUSTOM_OPERATORS`: a single source of truth mapping each custom operator
  name to `(julia_def, sympy_callable)`. Initial entry: `aq`.
  - `julia_def`: `"aq(x,y) = x / sqrt(1 + y^2)"` (for `binary_operators`).
  - `sympy_callable`: `lambda x, y: x / sympy.sqrt(1 + y**2)` (for PySR
    `extra_sympy_mappings` AND for `sympify` locals).
- `EXTRA_SYMPY_MAPPINGS` — derived `{name: callable}` for PySR kwargs.
- `sympify_equation(equation_str) -> sympy.Expr` — wraps `sympy.sympify` with
  `locals=` populated from the registry, so a raw equation string containing
  `aq(...)` parses to a differentiable sympy expression. **The feasibility
  spike confirmed bare `sympify` leaves `aq` undefined; this helper is the
  fix.**

Every existing bare `sympy.sympify(equation_str)` in the eval/filter path is
replaced with `sympify_equation(...)`:
- `refit_1d_pysr.Refit1DResult.predict_normalized` (the lambdify path),
- `pareto_filters.is_fisher_stencil_safe` (and any sibling parsers there).

### 3.2 Lever #2 — `aq` operator

- `SMART_REFIT_PYSR_KWARGS["binary_operators"]`: replace `"/"` with the `aq`
  julia def → `["+", "-", "*", CUSTOM_OPERATORS["aq"].julia_def, "^"]`.
  Raw `/` is **dropped** (keeping it lets the search reintroduce poles, which
  is exactly the derivative pathology we are removing). `DEFAULT_PYSR_KWARGS`
  gets the same treatment for consistency.
- Add `extra_sympy_mappings=EXTRA_SYMPY_MAPPINGS` to both kwarg dicts.
- `complexity_of_operators` / `constraints` for `aq` mirror what `/` had
  (no special penalty; `^` constraint unchanged).

### 3.3 Lever #3 — derivative-validation selection gate

A new filter in the existing **filter-then-pick** selection (alongside
`_filter_fisher_safe`), gating on derivative faithfulness:

- **GP target gradient.** New helper
  `gp_param_gradient(gp, fid, k_grid, z, param_idx, h) -> ndarray` (shape
  `(n_k,)`): central finite difference of `gp.predict` in θ around fid (the
  same quantity Fisher's stencil uses). Computed once per (param, z).
- **Equation gradient.** `equation_param_gradient(refit, fid, k_grid, z, h)`:
  the **same central finite-difference stencil** applied to the candidate's own
  `refit.predict(θ_phys, k, z)` around fid. Using the identical stencil for both
  the GP target and the equation makes the comparison apples-to-apples and
  measures precisely the derivative the Fisher matrix will consume (the Mirage
  is a property of that stencil derivative, so finite-diff — not symbolic diff —
  is the right probe here). `sympify_equation`/`sympy.diff` remains for
  `is_fisher_stencil_safe`'s symbolic checks, but the gate uses finite-diff.
- **Gate metric:** `median_k | (∂eq/∂θ) / (∂P_GP/∂θ) − 1 |`. Reject the
  equation if it exceeds `derivative_tol` (config, **default 0.25**). Guard the
  ratio against near-zero GP gradient bins (mask |∂P_GP/∂θ| below a small
  floor before taking the median).
- **Integration:** the selection becomes `filter_fisher_safe` →
  `filter_derivative_faithful` → `pick_equation(best_loss)`. If **all**
  equations fail the derivative gate, **GP-slice fallback** (as today) — but
  now the surviving pick is the most derivative-faithful Fisher-safe member,
  not merely `best_loss`.

### 3.4 Config surface

- `PySRConfig` (or the refit path) gains `use_aq_operator: bool = True` and
  `derivative_tol: float = 0.25`. Single-z `PipelineConfig` exposes
  `derivative_tol`. Defaults make the levers on-by-default; setting
  `derivative_tol` very large disables the gate (for A/B comparison).

## 4. Success metric

Re-run the single-z z=3.6 `refit_and_forecast` (real KODIAQ) with aq + gate,
and compare to the Stage 6 baseline (`results/single_z_stage6_log/`):

| metric | Stage 6 baseline | Stage 8 target |
|---|---|---|
| mean \|log10(σ_PySR/σ_GP)\| | 0.366 | lower |
| sub-1 Mirage count | 7/11 | fewer |
| deep-Mirage (<0.2×) count | 0 | 0 |
| params with no usable equation | (varies) | no worse |

Win = lower mean |log10| and fewer sub-1 ratios without increasing GP-slice
fallbacks. Captured in a `results/single_z_stage8/COMPARISON.md`.

## 5. Testing (TDD)

1. **Registry consistency:** every `CUSTOM_OPERATORS` entry has a julia def and
   a sympy callable; `EXTRA_SYMPY_MAPPINGS` derives from it.
2. **`sympify_equation` round-trip:** a string with `aq(x0, 2*x1)` →
   differentiable sympy expr; `sympy.diff` + lambdify eval match the closed
   form (the spike, as a committed test).
3. **`aq` in PySR kwargs:** `SMART_REFIT_PYSR_KWARGS` has no `/`, has the aq
   julia def + `extra_sympy_mappings`.
4. **`gp_param_gradient`:** on a stub GP with a known analytic θ-dependence,
   the finite-diff gradient matches closed form (rtol).
5. **`equation_param_gradient`:** on a known equation + norm, matches the
   analytic ∂P/∂θ.
6. **Gate filter:** synthetic Pareto frame with one derivative-faithful and one
   derivative-wrong equation (same value RMSE) → gate keeps the faithful one,
   rejects the other; all-fail → empty (triggers GP-slice fallback).
7. **No-regression:** existing single-z tests pass with `/`→`aq` (equations
   that used `/` are re-derivable; the eval path handles `aq`).
8. **Gated end-to-end** (`RUN_SLOW_REFIT`): single-z `refit_and_forecast` at
   z=3.6 with aq+gate completes and emits a COMPARISON.

## 6. Risks and watch-items

- **`aq` reduces expressiveness vs `/`:** some targets fit worse without raw
  division. Mitigation: the success metric tracks "no worse" on usable-equation
  count; if a param regresses badly we can re-enable `/` per-run via
  `use_aq_operator=False`.
- **Gate too strict → more GP-slice fallbacks:** `derivative_tol=0.25` is a
  guess. Watch the fallback count; tune the default from the z=3.6 run.
- **Near-zero GP-gradient bins** make the ratio explode — masked with a floor
  before the median.
- **PySR custom-operator + sympy round-trip** is confirmed feasible (spike),
  but every parse site must use `sympify_equation` — missing one silently
  breaks on `aq`. Test #7 guards this.

## 7. Out of scope

- Ratio-response target (#1) and the Sobolev training loss — deferred to a
  later stage, gated on whether #2+#3 close the Mirage enough.
- Multi-z application — mirror single-z after the single-z win is shown (own
  follow-up, not this spec's plan).
