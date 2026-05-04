# Local paper-writing handoff

> **Read this if you are a fresh Claude (or a human) on the user's
> laptop, picking up the paper writing.**
>
> The pipeline ran on the U-Mich Greatlakes cluster (kodiaq emulator
> + lyaemu + PySR/Julia) and produced the deliverables that are
> committed in this repo. To re-render figures locally without the
> cluster GP emulator, use `scripts/replot.py` (see below).

---

## Quick start

```bash
# 1. Clone and set up.
git clone https://github.com/jibanCat/lya1d_priya_forecast
cd lya1d_priya_forecast
git checkout fine-tune-pysr            # current paper branch

# 2. Minimal Python deps (NO pysr / julia / lyaemu / GPy needed locally).
pip install numpy scipy matplotlib sympy h5py pytest hypothesis

# 3. Re-render figures from cached refits (~10 s).
PYTHONPATH=src python scripts/replot.py \
    --results-dir results/refit_optionC_z2.6-4.2

# 4. Look at the figures + scorecard.
open results/refit_optionC_z2.6-4.2/scorecard.md
open results/refit_optionC_z2.6-4.2/corner.pdf
open results/refit_optionC_z2.6-4.2/resolution_correction_grid_cosmo.pdf
# … etc.
```

---

## Where things are

### Documentation (read first)

| File | What's in it |
|---|---|
| `docs/PAPER_NOTES.md` | **Master document.** Every modification we made beyond the student's pipeline, with reasoning. Has design-decision section (D1–D6), related work (Cabayol-Garcia 2023, Yang+ 2025), hyper-cost ledger, and a methods-section paragraph at the bottom that's paste-ready. |
| `docs/PYSR_PERFORMANCE.md` | Wall-time benchmark + speed analysis for the methods. |
| `docs/FIGURES.md`, `docs/PYSR_HYPOTHESIS.md` | Earlier-session diagnostics (still useful context). |
| `docs/ONBOARDING.md` | Student-onboarding write-up. |

### Forecast outputs (committed for paper writing)

| Path | Contents |
|---|---|
| `results/refit_optionC_z2.6-4.2/` | **Headline scorecard**: multi-z (z=2.6→4.2), per-1D PySR + additive-Taylor combine, kodiaq emulator, KSData covariance optionally, production priors. Contains: `scorecard.md`, `per_param_summary.md`, `resolution_correction.{md,json,grid_{cosmo,astro}.{png,pdf},equations.md}`, `resolution_correction_param_variation_{cosmo,astro}.{png,pdf}`, `holdout_validation_{cosmo,astro}.{png,pdf}`, `corner.{png,pdf}`, `fisher.npz`, `refits/<param>.pkl` (× 11). |
| `results/refit_optionC_z2.6-4.2_ksdata/` | Same as above but with `--use-ksdata` (real Karacayli+ 2021 cov). Compare hybrid/GP ratios vs synthetic-cov version. |
| `results/refit_kodiaq_optionB_z3.6/` | Single-z (z=3.6) baseline: per-1D + additive-Taylor at one redshift. Useful for ablations. |
| `results/refit_multid_z2.6-4.2/` | (When SLURM lands) Multi-D cross-coupled forecast: one PySR equation over 6 cross-coupled θs + (k, r, z). The "headline" PySR equation goes into the paper from `multid_equation.md`. |

### Source

| Path | What's there |
|---|---|
| `src/priya_forecast/` | All modules: GP wrapper, PySR refits (per-1D + multi-D), Fisher, KSData likelihood, dim-balanced loss (ANOVA + corr²), Pareto filters, deliverable plotting. |
| `scripts/` | End-to-end drivers: `refit_all_11_params.py` (single-z), `refit_one_param.py` (per-param SLURM-array unit), `precompute_payloads.py`, `multi_z_aggregate.py`, `refit_multid_subset.py`, **`replot.py`** (lightweight local). |
| `slurm/` | GreatLakes templates. Cluster-only. |
| `tests/` | 228 tests pass (no pysr/julia/lyaemu needed for the lightweight subset). |

---

## Lightweight local replot — what works, what doesn't

`scripts/replot.py` regenerates everything from `refits/*.pkl` +
`fisher.npz`. **No GP emulator, no PySR, no Julia required**.

| Output | Local replot? | Notes |
|---|---|---|
| `per_param_summary.md` | ✓ | full equations, prettified |
| `resolution_correction.{md,json}` | ✓ | HF/LF ratio table |
| `resolution_correction_grid_{cosmo,astro}.{pdf,png}` | ✓ | the headline figure |
| `resolution_correction_equations.md` | ✓ | symbolic expressions |
| `resolution_correction_param_variation_{cosmo,astro}.{pdf,png}` | ✓ | R(k; θ) per quantile |
| `corner.{pdf,png}` | ✓ | from `fisher.npz` |
| `holdout_validation_{cosmo,astro}.{pdf,png}` | ✗ | **needs the GP emulator on the cluster**. The committed PNGs/PDFs are the cluster's outputs — keep those, don't replot. |

To re-run the full pipeline (including hold-out validation): need to be
on Greatlakes with `lyaemu` installed and the kodiaq emulator at
`/nfs/turbo/umor-yueyingn/mfho/birdgroup/lya_xq100/kodiaq_2_2_4_6-48-48`.

---

## Reproducibility (paper-final fits)

The current results were generated with `parallelism="multithreading",
deterministic=False` (the genetic search uses non-determinism for ~5×
speed-up). For paper-final reproducibility, re-run with
`parallelism="serial", deterministic=True` once at the end:

```bash
# Edit src/priya_forecast/refit_1d_pysr.py:DEFAULT_PYSR_KWARGS
# and src/priya_forecast/refit_multi_d.py to set:
#   parallelism="serial", procs=1, deterministic=True
# then re-run the SLURM jobs. Adds ~4-5× wall time but makes the
# equations bit-reproducible.
```

This is documented in `docs/PYSR_PERFORMANCE.md` "Reproducibility
footnote".

---

## What's the "headline number"?

From the most recent multi-z aggregate at
`results/refit_optionC_z2.6-4.2/scorecard.md` (per-1D + additive-Taylor,
production priors, KSData covariance):

| param | hybrid σ / GP σ | notes |
|---|---|---|
| Ap | 0.66× | (overconstrained — multi-D fit may improve) |
| ns | 1.27× | clean |
| tau0 | 1.40× | clean |
| dtau0 | (fixed at 0) | Kim convention |
| hub, omegamh2, bhfeedback | 1.0×–1.3× | prior-dominated, clean |
| heref, alphaq | 1.05×, 0.63× | mixed — borderline |
| herei | 4.18× | needs multi-D PySR |
| **hireionz** | **broken (no x0 in eq)** | needs multi-D PySR; in the cross-coupled subset |

The multi-D PySR run (`results/refit_multid_z2.6-4.2/`) is the
follow-up: a single equation over `{ns, Ap, herei, heref, alphaq,
hireionz}` × `(k, resolution, z)` to capture cross-couplings the
per-1D + additive Taylor cannot.

---

## Outstanding (good to mention in the paper or follow-up)

1. **Hold-out validation errors are mixed**: per-1D + additive-Taylor
   gives ~1–3% mean rel-err on cosmology + mean-flux but the IGM
   thermal block can be 5–20% on hold-out. The multi-D fit should
   improve this on the cross-coupled subset; for the few remaining
   weakly-coupled params (omegamh2, bhfeedback) the error matters
   less because they're prior-dominated in Fisher anyway.

2. **Residual-PySR**: scaffolded but not yet run — `refit_residual.py`
   + `run_residual_pysr.py`. If multi-D doesn't fully resolve the IGM
   thermal cross-coupling (`herei × alphaq` from the Phase 5
   coupling-matrix headline), train a 2nd PySR on the residual.

3. **Log-log target representation** (Cabayol-Garcia 2023): we use
   linear `flux_norm`. Switching to `log P_F` vs `log k` may give
   PySR cleaner equations for cosmology tilts. Future experiment.

---

## Memory (for Claude)

The `~/.claude/projects/-home-mfho-lya1d-priya-forecast/memory/`
directory has 10+ memory files capturing user preferences, physics
context, and workflow rules:
- `student_pysr_contract.md` — the verbatim student-pipeline contract.
- `forecast_deliverables.md` — what every forecast run must produce.
- `feedback_replicate_exactly.md` — don't silently swap student components.
- `feedback_pysr_operators.md` — drop sin/cos.
- `feedback_pysr_speed.md` — multithreading > strict reproducibility for iteration.
- `at_fid_anchor_for_multiz.md` — multi-z normalization must anchor at fid.
- `igm_thermal_z_dependence.md` — herei/heref/alphaq/hireionz are z-dependent.
- `p1d_physics_regimes.md` — k-regime physics: pivot tilt, peculiar-velocity dip, resolution loss, cosmic-variance floor.
- `active_work.md` — TODO snapshot at session end.

Read those before adjusting any pipeline architecture.
