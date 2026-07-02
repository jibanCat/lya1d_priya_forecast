# CSV schema — the per-param refit sidecars

**code git** `7aa26af` · 2026-06-30. This `sobolev/` dir (and its siblings `value/`,
`budget35_*/`, `sens_maxsize*/`, and the `seed_band/*/` dirs) share this layout:
`refit/z<z>/` holds two files per PRIYA parameter.

### `pareto_<param>.csv` — the PySR Pareto front
Columns: `Complexity, Loss, Equation` (+ `score, sympy_format, lambda_format`).
`Loss` is the **training objective** (Sobolev = MSE + λ·‖∂eq−∂GP‖² for the Sobolev
dirs, plain MSE for `value/`), so it is **not comparable across losses** — use
`value_mse` (below) for cross-objective comparison. Inputs are `x0=θ_norm, x1=k_norm,
x2=resolution`. Produced by `scripts/refit_one_param_single_z.py`.

### `grad_faith_<param>.csv` — the emulator-scored sidecar
A `#`-comment provenance header (`# param= z= tol= log_space= git= source=`, where
`git=` is the short hash of the code that produced the file — the provenance convention,
`priya_forecast.provenance.git_stamp`) then columns:
`Complexity, Loss, grad_err, value_mse, n_keep, gate_pass, x0_enters`.
- `grad_err` = the production gate metric, `median_k |∂logP_eq/∂θ ÷ ∂logP_GP/∂θ − 1|`
  at fiducial (**log-space**), gate 0.25.
- `value_mse` = `mean_k (logP_eq − logP_GP)²` over the θ×k grid — the cross-objective-
  comparable value loss.
- `gate_pass` = grad_err ≤ tol; `x0_enters` = the equation uses the parameter feature.
Produced by `scripts/eval_grad_faithfulness.py` (via `scripts/make_grad_faith_sidecars.sh`).
Read + pick the Pareto **knee** with `priya_forecast.grad_faith_io.{read_grad_faith_sidecar,knee_row}`.

`sobolev/refit/z3.6/{refits,payloads}/<param>.pkl` are the pickled `Refit1DResult` +
its training payload (for the prediction figures); committed for reproducibility.

The 5-seed `seed_band/z3.6_seed<S>_{value,sobolev,budget}/refit/z3.6/` dirs share this
schema and are aggregated by `scripts/aggregate_seed_band.py` → `seed_band/seed_band_summary.json`.
