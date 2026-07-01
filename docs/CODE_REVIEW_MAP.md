# Code review map — reproduce all results end-to-end

A by-eye review guide to the pipeline that produces every result in the paper: what to
run, in what order, and — for each core file — the exact functions to read and which
**paper claim** they back. Line numbers are as of code git `ffbe28f` and drift a little;
the function names are stable. Companion: `REPRODUCE.md` (how to run) and
`docs/PAPER_MFHO_DECISION_SHEET_2026-07-01.md` (open author decisions).

---

## Part 1 — Rerun everything (one command)

```bash
scripts/submit_paper_production.sh          # SLURM: all per-z fits + value baseline
                                            # + 5-seed band + maxsize/budget sweeps
```
Self-documenting: writes `results/.../RUN_MANIFEST.md` with the git stamp + every job id.
Tiers (see `REPRODUCE.md`): **Tier 3** re-runs the fits (needs Julia/PySR + SLURM); **Tier 2**
the GP-backed figures; **Tier 1** the emulator-free figures/tables (~2 min).

---

## Part 2 — The pipeline, in the order results flow

| # | stage | file | key entry |
|---|---|---|---|
| 1 | submit / define the run | `scripts/submit_paper_production.sh` | the whole grid (z, seeds, λ, maxsize) |
| 2 | SLURM array (per param, z) | `slurm/single_z_refit.slurm` | one task per param |
| 3 | refit one param | `scripts/refit_one_param_single_z.py` → `refit_1d_pysr.refit_1d_for_param` | the PySR fit |
| 4 | GP-score each candidate | `scripts/make_grad_faith_sidecars.sh` → `scripts/eval_grad_faithfulness.py` | the `grad_faith_*.csv` sidecars |
| 5 | aggregate seeds | `scripts/aggregate_seed_band.py` | `seed_band_summary.json` |
| 6 | figures + tables | `notebooks/reproduce_paper.ipynb` (+ `scripts/regen_*`, `paper_figures.py`) | every paper float |

---

## Part 3 — Core science files (read these — mapped to paper claims)

### Review order (≈1,040 lines = the whole method)
`parameters.py` → `sobolev_loss.py` → `derivative_gate.py` → `refit_taylor.py` → `grad_faith_io.py`.

### `src/priya_forecast/parameters.py` (136 lines) — **Table 1**
- `class Param` (`:20`) — `fid`, `prior=(lo,hi)`, `latex`; `lo<fid<hi` enforced at load.
- `PARAMS_11D` (`:56`) — the 11 params, fiducials, priors. **This is the source of truth for
  Table 1.** ⚠️ *Known discrepancy:* the paper's Table 1 (L173) lists z_Hei max 4.1 / z_Hef min
  2.6, but the emulator hypercube here is [3.5,4.5] / [2.2,3.2] — your open `\mfho` decision.
- Note `A_P` is stored as `A_P/1e-9` (fid 1.46 = 1.46e-9); only the GP sees the physical value.

### `src/priya_forecast/sobolev_loss.py` (177 lines) — **the Sobolev method (Eq. sobolev)**
- `make_sobolev_loss(lam, h)` (`:13`) — the Julia `loss_function` string PySR minimizes:
  `MSE(pred, y) + λ·MSE(∂pred/∂x0, target_grad)`, with `∂pred/∂x0` finite-differenced *inside*
  the loss (`:25-31`) and `target_grad` delivered via PySR's per-point `weights`.
- `_fidelity_grad_weights` (`:39`) / `_fidelity_grad_weights_multiz` (`:80`) — build the target:
  **`∂logP_GP/∂θ`** (takes `np.log` of `gp.predict`, `:72-74` / `:115-116`), normalized by the
  param width and per-k std. **This is why the target is log-space.**
- `sobolev_target_weights[_multiz]` (`:125/:160`) — assemble LF+HF rows to match the training matrix.

### `src/priya_forecast/derivative_gate.py` (114 lines) — **`grad_err` (Eq. grad_err) + the gate**
- `gp_param_gradient` (`:22`) — the GP slope; `log_space=True` → `np.log` before differencing (`:39-40`)
  → returns `∂logP_GP/∂θ`.
- `equation_param_gradient` (`:44`) — the equation slope; `log_space` → uses `refit.predict_log` (`:56-57`).
- `derivative_faithful` (`:65`) — `median_k |cand/target − 1| ≤ tol` (tol 0.25). The **floor mask** (`:78`,
  `floor_frac=1e-3`) drops bins where the GP slope is near-zero; applied to **both** sides (`:81`).
- ⚠️ *Doc nit:* the module docstring (`:8`) still shows the linear `|dP_eq/dP_GP−1|` (the `log_space=False`
  default); **production runs `log_space=True`** (log-slope). The numbers of record are log-space.

### `src/priya_forecast/refit_taylor.py` (540 lines) — **the additive combine + the Fisher-consistency identity**
- `mode="local_anchored"` (`:350`) — the deployed model:
  `P_F(θ) = P_GP(fid) + Σ_i [P_F_i(θ_i) − P_F_i(fid_i)]`; at θ=fid every deviation cancels → `P_GP(fid)`.
- Multi-z `predict` (`:250-280`): `out_log = logP_GP(fid) + Σ (logP_eq(θ) − logP_eq(fid))`, then
  `np.exp` (`:280`). **This is the anchor-cancellation** the paper cites: differentiating at fid gives
  `∂P_F/∂θ = P_GP(fid)·∂logP_eq/∂θ`, so `P_GP(fid)` cancels in the GP ratio and the log-slope ratio
  equals the deployed model's linear (Fisher-space) slope ratio. Single-z equivalent at `:445`.

### `src/priya_forecast/grad_faith_io.py` (77 lines) — **sidecar schema + the knee selection**
- `write_grad_faith_sidecar` (`:35`) — the `# param= z= tol= log_space= git= source=` header (the
  provenance convention) + columns; `read_grad_faith_sidecar` (`:53`) skips the `#` header.
- `knee_row` (`:59`) — **the Pareto-knee pick**: lowest complexity within `rel_tol=10%` of best loss.
  ⚠️ *Important:* the paper's Tables 6/7 use the **knee**, not plain best-loss (`idxmin`), which
  degenerates to the most-complex equation on a truncated front (`:64-66`).

### `src/priya_forecast/refit_1d_pysr.py` (1171 lines) — the refit engine (skim; mostly plumbing)
- `Refit1DResult.predict` (`:266`) returns raw `P_F`; `predict_log` (`:277`) returns `logP_F`
  (`log_space=True` applies `exp`/`log`). These are what the gate calls.
- `refit_1d_for_param` (`:1001`) — the single-z production path. **Guard `:1088`:** `use_sobolev`
  requires `log_space=True` (else the log target vs linear model is self-contradictory).
- `refit_1d_multiz_for_param` (`:835`) — multi-z; **guarded `:873`** (`NotImplementedError`, the M2
  fix) — multi-z Sobolev is disabled (linear target vs log weights). Single-z is the production path.
- The Pareto CSV writer (`:~1155`) emits the `# git=` provenance header (skim).

---

## Part 4 — Scoring, aggregation, figures

- `scripts/eval_grad_faithfulness.py` (169) — `median_rel_error` (`:38`) = the gate metric;
  `main` (`:51`) scores every Fisher-safe candidate (`_filter_fisher_safe`, `:86`) and writes the
  sidecar. **`--log-space` is the default now** (`:62`); `--linear-space` opts out.
- `scripts/aggregate_seed_band.py` (137) — `GATE=0.25` (`:29`), `SEEDS=0..4` (`:30`); per param takes
  the **knee** `grad_err` (`best_loss_grad_err` → `knee_row`, `:46-55`) across seeds → median/min/max.
- `scripts/make_diagnostic_figs.py` (274), `regen_maxsize_sensitivity.py` (202), `regen_multid.py` (487),
  `src/priya_forecast/paper_figures.py` (415, the reusable Tier-1 module).
- GP-prediction plots live in the **paper repo**: `$FIGREPO/scripts/regen_fig{1,3,4}.py` + `regen_table2.py`.
- `notebooks/reproduce_paper.ipynb` — the single notebook that runs every figure/table (Tier-1 + Tier-2).

### Fisher forecast (background only — the paper dropped the σ presentation)
`likelihood.py` (238) + `ksdata_likelihood.py` (239) + `fisher.py` (417) — the KODIAQ-SQUAD
**linear-P_F** covariance + Fisher matrix (no `np.log` in `fisher.py`; this is what makes the
log-slope-of-the-combine the Fisher-relevant quantity — see `refit_taylor.py` above).

---

## Part 5 — Things to keep in mind while reviewing
1. **grad_err is log-space and that is correct** — not a mislabel: via the anchored combine
   (`refit_taylor.py`) the log-slope ratio = the deployed model's linear Fisher-space slope ratio.
   `docs/pr_review/VERDICT.md`'s "linear-P, correctly labelled" bullet is **superseded** (banner there).
2. **Knee, not best-loss** — Tables 6/7 use `knee_row`; the committed `taxonomy_table.tex` /
   `per_param_equations.tex` on disk are the *old best-loss* cut (different numbers) — don't mix them.
3. **Single-z is production; multi-z Sobolev is disabled** (guarded, M2).
4. **The z=4.2 IGM-thermal "blow-ups"** are the gate hitting a noise-level GP slope (relative-only
   floor), not unfaithful equations — state IGM-thermal verdicts per-z.
5. Open author decisions (priors L173, log-P eq conventions, wall-time, etc.) are in the decision sheet.
