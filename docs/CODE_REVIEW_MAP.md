# Code review map — where each result of the paper comes from

A guide for reading the code by eye: the order in which results flow, and, for each core
file, the function to read and the statement of the paper it backs. It names functions, not
line numbers, so it does not go stale when a file is edited.

Companions: [`REPRODUCE.md`](../REPRODUCE.md) says how to run everything and has the table
"Selection rules and settings"; [`scripts/README.md`](../scripts/README.md) shows the plotting
function behind each figure.

---

## Part 1 — The pipeline, in the order results flow

| # | stage | entry point | what it writes |
|---|---|---|---|
| 1 | define the run | `scripts/submit_paper_production.sh` | the grid: redshifts, seeds, Sobolev weight, `maxsize`, search budget; `RUN_MANIFEST.md` |
| 2 | one fit | `slurm/single_z_refit.slurm` → `scripts/refit_one_param_single_z.py` → `single_z/refit.refit_one_param_single_z` → `refit_1d_pysr.refit_1d_for_param` | `pareto_<param>.csv`; with `--save-artifacts` also `refits/<param>.pkl` and `payloads/<param>.pkl` |
| 3 | sweep files for scoring | `scripts/regen_1pvar.py` → `single_z/training_data.regenerate_param` | `data/single_z_1pvar/{lf,hf}_<param>_npoints50.hdf5` |
| 4 | score every valid candidate | `scripts/make_grad_faith_sidecars.sh` → `scripts/eval_grad_faithfulness.py` → `grad_faith_score.score_pareto` | `grad_faith_<param>.csv` beside each Pareto file |
| 5 | aggregate the seeds | `scripts/aggregate_seed_band.py` | `seed_band/seed_band_summary.json` |
| 6 | multi-parameter test | `scripts/regen_multid.py` | `figures/multid_z3.6/multid_bestworst.csv` and `multid_bestworst_perk.csv` |
| 7 | figures and tables | `scripts/make_figures.py`; `paper_figures.taxonomy`, `paper_figures.equations`; `notebooks/reproduce_paper.ipynb` | the seven figures; Tables 2 to 5 |
| 8 | Sec. 5.1 joint fit | `scripts/run_multid_pysr.py`, `scripts/aggregate_joint_sweep.py` | `results/joint_multid_sweep/` |

`python -m priya_forecast.rerun` runs stages 2 to 4 for a whole grid in one process, into
`results/tutorial_reruns/` (`rerun.run_grid`).

---

## Part 2 — Core science files, mapped to the paper

Suggested reading order: `parameters.py` → `refit_1d_pysr.py` (the two functions named below)
→ `sobolev_loss.py` → `derivative_gate.py` → `grad_faith_score.py` → `grad_faith_io.py` →
`pareto_filters.py` → `refit_taylor.py`.

### `src/priya_forecast/parameters.py` — Table 1
- `PARAMS_11D`: the eleven parameters with their ranges and fiducial values. This is the
  source of Table 1.
- `A_P` is stored in units of 1e-9 (fiducial 1.46); only the emulator sees the physical value.
- `with_overrides`, `override_params`: used by the rerun notebook only. `PARAMS_11D` itself is
  never changed.

### `src/priya_forecast/refit_1d_pysr.py` — the fit
- `_generate_1pvar_inline`: the training set of one fit. Fifty values of one parameter across
  its range, the other ten at the fiducial point, at low and high fidelity. It is built in
  memory from the emulator at fit time.
- `SMART_REFIT_PYSR_KWARGS`: the operators (`+ − × aq ^`; `exp log square`; `^` is listed and
  constrained out of the search).
- `refit_1d_for_param`: the single-redshift production fit. It refuses the Sobolev loss on a
  linear target, because the Sobolev term matches a log-slope. Its returned equation is the
  lowest-loss row of the Pareto front (`pareto["loss"].idxmin()`): the paper's "lowest-loss"
  rule.
- `Refit1DResult.predict` returns the flux power; `predict_log` returns its logarithm.
- `refit_1d_multiz_for_param` raises `NotImplementedError` for the Sobolev loss. The paper's
  fits are single-redshift.

### `src/priya_forecast/single_z/refit.py` — the retry
- `refit_one_param_single_z`: repeats the fit with a new seed, up to five attempts, until the
  front holds at least one valid candidate (`max_retries=4`).

### `src/priya_forecast/sobolev_loss.py` — the Sobolev loss, Eq. 10
- `make_sobolev_loss(lam, h)`: the loss PySR minimises, `mse + λ·gmse`. The equation's slope
  is finite-differenced inside the loss; the emulator's slope arrives through PySR's
  per-point `weights`.
- `_fidelity_grad_weights`: the target slope, a central difference of the emulator's `log P`,
  rescaled to the normalised variables of the fit.

### `src/priya_forecast/derivative_gate.py` — the finite differences
- `gp_param_gradient`, `equation_param_gradient`: the same central difference for the emulator
  and for an equation, at the fiducial point, step `1e-3·max(|θ_fid|, 1)`. With
  `log_space=True`, which is what production uses, they difference `log P`.
- The module docstring still writes the metric in linear form; the numbers of record are in
  log space.

### `src/priya_forecast/grad_faith_score.py` — the derivative error, Eq. 9
- `median_rel_error`: the median over k-bins of `|slope_eq / slope_GP − 1|`, skipping bins
  where the emulator's slope is below `1e-3` of its peak.
- `score_pareto`: scores every valid candidate of one Pareto file. It rebuilds the equation's
  normalisation from the sweep file of stage 3 (`single_z/forecast.per_param_local_norm`, Eq. 2),
  and stops if that file's k-grid differs from the scoring grid. The sweep file must come from
  the same emulator as the fit; nothing checks that.

### `src/priya_forecast/grad_faith_io.py` — score files and the knee
- `write_grad_faith_sidecar`, `read_grad_faith_sidecar`: the score file, with its header line
  `# param= z= tol= log_space= git= source=`.
- `knee_row`: the Pareto knee. Among the rows of a score file (valid candidates only), the
  simplest one whose loss is within 10% of the best (`rel_tol=0.1`).

### `src/priya_forecast/pareto_filters.py` and `single_z/forecast.py` — valid candidates
- `single_z/forecast._filter_fisher_safe` keeps an equation if all three hold:
  `equation_uses_param` (it contains the parameter); `has_pathological_constant` is false (no
  constant above 100); `is_fisher_stencil_safe` (it stays finite when each input is shifted by
  ±0.05 and ±0.1 around the middle of its normalised range).
- `single_z/forecast.build_refit_from_pareto_gated`: the lowest-loss valid candidate that also
  passes the 0.25 threshold. This is the rule behind Table 5 and Figures 3 and 4.

### `src/priya_forecast/refit_taylor.py` — the additive combine, Eqs. 3 to 5
- `AdditiveTaylorModel.predict` with `log_space=True`:
  `log P = log P_GP(fid) + Σ_i [log f_i(θ_i) − log f_i(fid)]`, then the exponential. At the
  fiducial point every deviation cancels and the model returns the emulator's value.

### Background, not used for a result of the paper
`likelihood.py`, `ksdata_likelihood.py`, `fisher.py` and the `multi_z/` package implement a
Fisher forecast. `scripts/run_multid_pysr.py` uses the Fisher matrix for the rank diagnostic of
Sec. 5.1; the paper reports no forecast.

---

## Part 3 — Things to keep in mind while reading

1. **The derivative error is in log space.** The fits train on `log P`, the Sobolev term
   matches a log-slope, and the score compares log-slopes.
2. **Three selection rules are in use**, and each table and figure uses one of them: lowest-loss,
   lowest-loss valid, and the Pareto knee. The table in `REPRODUCE.md` says which is used where.
3. **Scoring depends on a file that is not committed**: the sweep files of stage 3. For the
   committed results they match the sweeps the fits were trained on.
4. **The fits are not deterministic.** PySR runs multithreaded, so a rerun reproduces an
   equation's faithful or unfaithful class, not its digits.
5. **Provenance.** Score files, the seed-band summary and the multi-parameter tables carry a
   `git=` stamp. The committed Pareto files, saved fits and `.tex` table files do not; their record is
   `RUN_MANIFEST.md`.
