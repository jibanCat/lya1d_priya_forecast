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
    --results-dir results/refit_optionC_z2.6-4.2_ksdata

# 4. Look at the figures + scorecard.
open results/refit_optionC_z2.6-4.2_ksdata/scorecard.md
open results/refit_optionC_z2.6-4.2_ksdata/corner.pdf
open results/refit_optionC_z2.6-4.2_ksdata/resolution_correction_grid_cosmo.pdf
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
| `results/refit_optionC_z2.6-4.2_ksdata/` | **Headline scorecard** (paper number): multi-z (z=2.6→4.2), per-1D PySR + additive-Taylor combine, kodiaq emulator, **real KSData (Karacayli+ 2021) covariance**, production priors. Contains: `scorecard.md`, `per_param_summary.md`, `resolution_correction.{md,json,grid_{cosmo,astro}.{png,pdf},equations.md}`, `resolution_correction_param_variation_{cosmo,astro}.{png,pdf}`, `holdout_validation_{cosmo,astro}.{png,pdf}`, `corner.{png,pdf}`, `fisher.npz`. The `refits/<param>.pkl` (×11) live in the synthetic-cov dir below and are shared (the only difference between the two dirs is the covariance, not the PySR fits). |
| `results/refit_optionC_z2.6-4.2/` | Same per-1D PySR refits, but with a synthetic 5%-diagonal covariance instead of the real KSData cov. **Kept for ablation only — the numbers there are NOT the paper headline.** Contains the same set of artifacts plus the `refits/<param>.pkl` ×11. |
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

# Paper content guide

Four sections below cover everything the paper needs from this work.
Reproducibility + memory pointers continue after.

## 1. Per-1D + additive-Taylor headline result (what works)

Per-parameter 1D PySR equations on at-fid-anchored normalized P_F,
multi-z z=2.6→4.2 (9 z-bins), KODIAQ-SQUAD + XQ-100 production GP
emulator, KSData full cross-z covariance, 11D Fisher with production
Gaussian priors (hub σ=0.015, omegamh2 σ=0.001, bhfeedback σ=0.005,
tau0 σ=0.331 = Kim 0.304·1.090). dtau0 fixed at 0 (Kim mean-flux
convention).

**Headline scorecard** (numbers from
`results/refit_optionC_z2.6-4.2_ksdata/scorecard.md`, post BLOCKER #1
gate fix that routes broken refits to GP-slice):

| param | σ_PySR / σ_GP | route | flag |
|---|---|---|---|
| Ap | 0.77× | PySR | mildly overconstrained |
| ns | 1.31× | PySR | clean |
| tau0 | 1.40× | PySR | clean |
| hub | 1.27× | PySR | prior-dominated |
| alphaq | 0.80× | PySR | clean |
| **heref** | **6.07×** | PySR | biggest miss |
| **herei** | **4.33×** | PySR | second-biggest miss |
| omegamh2 | 0.99× | GP-slice (gated) | x0 dropped from PySR eq |
| bhfeedback | 1.01× | GP-slice (gated) | x0 dropped from PySR eq |
| hireionz | 1.03× | GP-slice (gated) | x0 dropped from PySR eq |
| dtau0 | (fixed at 0) | — | Kim convention |

**Closure to σ_MCMC_simdat** (synthetic-data MCMC at θ_target_simdat
ind=15, the cleanest available benchmark — see §3 for why simdata not
real-data):

| param | σ_PySR / σ_MCMC | flag |
|---|---|---|
| ns | 1.0× | ✓ |
| tau0, hub, bhfeedback | 1.4–1.5× | OK |
| Ap | 0.77× | mildly overconfident |
| alphaq | 0.8× | mildly overconfident |
| omegamh2 | 0.6× | overconfident (priored, GP-slice) |
| **heref** | **14×** | **biggest miss** |
| **herei** | **3.4×** | second-biggest miss |
| hireionz | 3.7× | GP-slice fallback (PySR eq broken) |

**What the paper should claim**: per-1D + additive-Taylor closes
σ_MCMC_simdat within ~1.0–1.5× for cosmology + mean-flux + bhfeedback
(5/11 parameters) and is structurally rank-correct (Fisher full-rank by
construction since each θᵢ has its own 1D PySR eq). It under-performs
on the IGM thermal block (`heref`, `herei`) and produces overconfident σ
on prior-dominated weakly-coupled params (`omegamh2`).

**The paper should NOT claim Phase 1 alone is the final result** —
Phase 2 (per-pair coupling, see §2) is needed to recover σ_MCMC for the
IGM block and the cosmology degeneracies revealed by the simdata MCMC.

## 2. Phase 2 design: rank-additive ANOVA cross-coupling

> Full plan in `docs/PAIR_FIT_PLAN.md`. Summary here is paper-ready.

**Hybrid prediction structure** — additive functional ANOVA, truncated
at 2-way:

    P̂(θ) = P_GP^HF(fid)
          + Σ_i [P̂_i(θ_i) − P̂_i(fid_i)]                              ← Phase 1 main effects
          + Σ_{(i,j) ∈ pairs} cross_diff_{ij}(θ_i, θ_j)                ← Phase 2 interactions

where the per-pair cross-difference is:

    cross_diff_{ij}(θ_i, θ_j) = Ĝ_{ij}(θ_i, θ_j)
                              − Ĝ_{ij}(θ_i, fid_j)
                              − Ĝ_{ij}(fid_i, θ_j)
                              + Ĝ_{ij}(fid_i, fid_j)

Each cross-difference is **structurally zero** whenever either θ_i = fid_i
or θ_j = fid_j — the standard ANOVA pure-interaction term. Three
properties:

1. **Exact at fid** by construction (every bracketed term is 0).
2. **Each pair adds an independent gradient direction** in the (θ_i, θ_j)
   plane, orthogonal to all per-1D directions. Fisher rank stays full;
   contrasts directly with the failed multi-D approach (see §4a).
3. **Graceful degradation**: if pair signal is weak, Ĝ_{ij} fits to ≈ 0
   and the cross-difference is 0. Adding a redundant pair costs compute
   but cannot harm Fisher.

**Implementation**: per pair, fit a 5-D PySR equation
(θ_i_norm, θ_j_norm, k_norm, resolution, z_norm) on the residual
`P_GP(θ_i, θ_j, others=fid) − P̂_phase1`. Embarrassingly parallel via
SLURM. ~1 h SLURM per pair. Pair selection from simdata MCMC posterior
correlations (§3).

**Phase 1.5 architectural fix BEFORE pair fits** (per user direction
2026-05-04): refit `heref`, `herei`, `alphaq` per-1D with
(a) ANOVA dim-balanced loss (currently the per-1D path uses plain MSE),
(b) restricted unary operators {exp, log, square} (drop inv, sqrt for
smoother fid-curvature). Don't naively bump niter — these params are
weakly sensitive to P_F at fid; architectural fixes are the lever.

## 3. Findings: real-data vs simdata MCMC covariance (motivates §2)

Both MCMC chains live at:

- Real KODIAQ-SQUAD data: `chains/s-scalecovar1.0-kodiaq_squad_only-48-z2.6-4.2-loo-nodatacorr-noemuerror-optimiseGP-discardkbins-0.005-0.064-rescorr-datacorr4/`
- Synthetic-data closure (P_F drawn from GP at θ_target_simdat=ind15, fit
  back): `chains/simdat/s-simdat-ind15-48-z2.6-4.2-loo-nodatacorr-noemuerror-optimiseGP-discardkbins-0.005-0.064/`

**Top |ρ| ≥ 0.2 on the 11 forecast params**:

| pair | ρ_simdata | ρ_realdata | finding |
|---|---|---|---|
| tau0 × ns | **−0.92** | −0.70 | strongest cosmology degeneracy; real-data underestimates by 32% |
| Ap × alphaq | **+0.68** | +0.07 | hidden in real data |
| tau0 × Ap | **−0.66** | +0.28 | sign-flipped + 2.4× compressed in real data |
| ns × Ap | +0.55 | +0.03 | hidden |
| tau0 × alphaq | −0.55 | −0.28 | 2× compressed |
| ns × alphaq | +0.43 | +0.13 | hidden |
| heref × alphaq | +0.29 | +0.19 | matches |
| herei × alphaq | −0.22 | −0.26 | matches; Phase 5 IGM coupling |

**Why the difference**: real-data posteriors hit prior boundaries on
several parameters (especially `dtau0`, `tau0`, `Ap` in the KODIAQ-SQUAD
likelihood), which compresses off-diagonal correlations *as a posterior
phenomenon*, not a likelihood phenomenon. Simdata draws P_F from the GP
at a boundary-clean fiducial point and fits it back, so its correlations
reflect the actual likelihood.

**Implication**: simdata reveals that the
`(tau0, ns, Ap, alphaq)` cosmology+mean-flux+thermal-slope quartet is
heavily entangled (5 pairwise edges with |ρ| ≥ 0.4). Real-data MCMC
showed only `tau0 × ns` clearly. Phase 2 pair selection (must-have:
`tau0×ns` + `herei×alphaq`; should-have: `Ap×alphaq` + `tau0×Ap`) is
**driven by simdata, not real-data**.

**Paper figure**: side-by-side correlation matrices (real-data vs
simdata, 11×11) with prior-bounded params highlighted. PDFs already
exist at `chains/.../corr_matrix.pdf` for both directories. Cached
σ_MCMC + ρ_MCMC at `results/simdat_ind15_truth.npz`.

## 4. Failure story (briefly, just enough to rationalize choices)

> Mention these in the paper to rationalize the design choices in §1+§2,
> but don't dwell — they are the *why*, not headline results.

### 4a. Multi-D PySR over the 6-θ joint subset (abandoned)

We attempted a single PySR equation over
`{ns, Ap, herei, heref, alphaq, hireionz}` plus `(k, resolution, z)`.
Two configurations:

| run | procs | niter | complexity | flux_norm loss | Fisher outcome |
|---|---|---|---|---|---|
| login smoke | 4 | 50 | 24 | 0.554 | Ap/herei NaN; heref 5e6× |
| SLURM, stencil-safe filter | 15 | 100 | 25 | 0.585 | ns/herei/heref NaN; Ap 9.6e6×; alphaq, hireionz 6e19× |

Both Fisher matrices have eigenvalue spreads of 26 orders of magnitude
— numerical artifact of inverting a near-rank-deficient matrix. The
SLURM run is *worse* because more dimensions collapsed onto a shared
`exp(...)` group (PySR's Pareto rewards low loss per complexity → folds
features into shared nonlinear groups).

**Why per-1D + Taylor (Phase 1) avoids this**: each θ_i has its own 1D
equation, so each gradient direction is structurally distinct. Fisher
rank full by design.

PySR docs state symbolic regression works best on ≤4-feature problems
(github.com/MilesCranmer/PySR + Cranmer 2023, arXiv:2305.01582). With 9
features, multi-D was outside the recommended regime. **Lesson Phase 2
internalizes**: keep each PySR call low-D — per-pair (5 features) is in
the sweet spot.

### 4b. Three broken per-1D equations (PR #1 BLOCKER #1, fixed)

Three params produced PySR equations *technically using θ but
numerically meaningless*:

| param | failure | evidence |
|---|---|---|
| `dtau0` | literal `c=−8.1e13` constant | rel-err 1.5×10¹⁴%; Fisher → ∞ |
| `hireionz` | eq has no `x0` | σ_PySR/σ_GP = 1.09×10¹² before fix |
| `omegamh2`, `bhfeedback` | eq has no `x0` | σ-ratios near 1× by accident (priors dominate) |

All four are *weakly coupled* to P_F at fid (small ∂P_F/∂θ at fid). PySR
Pareto kept dropping x0 — the loss penalty for including a near-zero
feature exceeded the gain. Known PySR behavior, see PAPER_NOTES § D3.

**Architectural fix** is the ANOVA dim-balanced loss (penalizes
batch-level main effects on dropped features). Currently wired only into
the abandoned multi-D path; **wiring into per-1D is Phase 1.5** before
adding pair fits.

**Operational fix in PR #1**: aggregator gates broken refits and routes
them via GP-slice. Fisher contribution comes from the full GP for those
four params. Honest σ; the four don't claim "PySR matches GP" but
that's the truth.

**Lesson**: PySR's MSE Pareto is dimension-blind and tends to drop
weakly-sensitive features. ANOVA loss + per-feature complexity caps via
TemplateExpressionSpec are the architectural fixes.

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
