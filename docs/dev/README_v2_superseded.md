# README_v2 — Reproduce Phase 1.5 and Phase 2

This is a reproducibility quickstarter. Run the commands here and you
should regenerate the canonical Phase 1.5 and Phase 2 scorecards with
the headline numbers in `docs/PAPER_NOTES.md`.

If you want the math first, read `docs/ONBOARDING.md`. If you want to
score your own PySR CSVs interactively at one z, use `README.md`. This
doc is for **reproducing the production multi-z forecast**.

> ⚠️ **Are you on a fresh machine, off-Greatlakes, or hitting cryptic
> errors?** This doc is a *cheatsheet* tested on Greatlakes; it assumes
> the GP emulator pickles, simdat truth file, and a working PySR/Julia
> bootstrap are all already in place. If any of those is missing, or
> you need a verification checklist for each step, **read
> [`docs/REPRODUCE.md`](docs/REPRODUCE.md) first** — it covers data
> acquisition, cluster customization, step-by-step verification
> commands, and a recovery table for the dozen most common failures.

---

## 0. Prerequisites

- **Compute**: Greatlakes (or any SLURM cluster) for the parallel
  per-param/per-pair refits. Single-node sequential also works but
  takes ~15× longer for Phase 2.
- **Data**: upstream `lyaemu` GP emulator at
  `/home/mfho/student_projects/lya_emulator_full` (Greatlakes) or your
  own clone of `https://github.com/sbird/lya_emulator`. The KODIAQ
  emulator pickles are at
  `/nfs/turbo/umor-yueyingn/mfho/birdgroup/lya_xq100/kodiaq_2_2_4_6-48-48/`
  on Greatlakes — this is the default for `multi_z_aggregate.py
  --basedir` so you don't need to pass it explicitly.
- **`results/simdat_ind15_truth.npz`** — the simulated-data MCMC
  truth + chain at `θ_target_simdat`, used by both closure scripts
  (§ 1d, § 2f) as `--truth`. The repo ships this file
  (`results/simdat_ind15_truth.npz`); on a fresh clone it must come
  through git LFS or be regenerated separately. If you only care about
  the headline σ-table and hold-out rel-err numbers (§ 1c, § 1e, § 2d,
  § 2g), you can skip the closure step entirely.
- **Software**:

```bash
git clone <this repo> && cd lya1d_priya_forecast
pip install -e ".[forecast,pysr,gp,dev]"
export PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:$PWD/src
export PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env
export JULIA_DEPOT_PATH=$HOME/.julia
```

PySR / Julia bootstrap takes a few minutes the first time. Re-runs
hit the Julia depot cache.

---

## 1. Phase 1.5 — per-1D PySR + smart-kwargs for IGM block, no pair coupling

**What this produces**: 11 per-parameter PySR equations stitched into
a `MultiZAdditiveTaylorModel` with GP-slice fallback for params whose
fit fails the 5% rel-err gate or drops `x0`.

**Canonical output dir**: `results/refit_optionC_z2.6-4.2_phase1_5_ksdata/`

### 1a. Precompute payloads (one job, ~7 min wall)

Heavy GP work happens here ONCE; the per-param refits then load tiny
pickles instead of re-running thousands of GP predicts.

```bash
python scripts/precompute_payloads.py \
    --z-min 2.6 --z-max 4.2 --n-total 225 \
    --output results/refit_optionC_z2.6-4.2_phase1_5_ksdata/payloads
```

This writes `payloads/<param>.pkl` for each of the 11 PRIYA params.

### 1b. Per-param refits (11 SLURM tasks, ~3 min wall each)

The Phase 1.5 distinguishing feature: smart kwargs (option-B operators
+ ANOVA loss) for the **IGM thermal subset only** (`heref, herei,
alphaq, Ap`), default kwargs for the rest. The flag `--auto-smart`
does this automatically.

```bash
sbatch \
    --export=ALL,REPO=$PWD,\
PAYLOAD_DIR=results/refit_optionC_z2.6-4.2_phase1_5_ksdata/payloads,\
OUTPUT_DIR=results/refit_optionC_z2.6-4.2_phase1_5_ksdata \
    --array=0-10 \
    slurm/refit_array.slurm
```

The shipped SLURM template uses `--smart` (= smart for all 11) which is
the **Phase 2** behavior. To reproduce **Phase 1.5** exactly, copy the
template and flip the flag — *do not edit the shipped file in place*
(line numbers will drift across commits and you'll fight rebases):

```bash
cp slurm/refit_array.slurm slurm/refit_array_phase1_5.slurm
sed -i 's/--smart$/--auto-smart/' slurm/refit_array_phase1_5.slurm
# then sbatch ... slurm/refit_array_phase1_5.slurm
```

Or skip SLURM and run sequentially:

```bash
for p in dtau0 tau0 ns Ap herei heref alphaq hub omegamh2 hireionz bhfeedback; do
    python scripts/refit_one_param.py \
        --param "$p" \
        --payload-dir results/refit_optionC_z2.6-4.2_phase1_5_ksdata/payloads \
        --output-dir results/refit_optionC_z2.6-4.2_phase1_5_ksdata \
        --niter 50 --maxsize 20 --max-retries 2 --seed 42 \
        --auto-smart
done
```

Output: `results/refit_optionC_z2.6-4.2_phase1_5_ksdata/refits/<param>.pkl`.

### 1c. Aggregate to multi-z Fisher scorecard

```bash
python scripts/multi_z_aggregate.py \
    --refits-dir results/refit_optionC_z2.6-4.2_phase1_5_ksdata/refits \
    --output results/refit_optionC_z2.6-4.2_phase1_5_ksdata \
    --priors production --kim-tau0 --use-ksdata
```

`--use-ksdata` selects the real KODIAQ-SQUAD covariance (Karaçaylı+
2021); otherwise a synthetic diagonal cov is used and the σ-ratios
won't match the canonical numbers.

### 1d. Validation (optional but cheap)

**Multi-D Sobol hold-out** (the headline rel-err numbers):

```bash
python scripts/holdout_multid.py \
    --refits-dir results/refit_optionC_z2.6-4.2_phase1_5_ksdata/refits \
    --output results/holdout_multid_phase1_5 \
    --n-sobol 64 --z-eval 3.6
```

**Off-fid Fisher closure** at `θ_target_simdat` (σ_PySR vs σ_MCMC):

```bash
python scripts/closure_at_simdat_target.py \
    --refits-dir results/refit_optionC_z2.6-4.2_phase1_5_ksdata/refits \
    --truth results/simdat_ind15_truth.npz \
    --output results/closure_at_simdat_ind15_phase1_5_ksdata
```

### 1e. Expected results — Phase 1.5

**11-θ joint Sobol hold-out (n=64) at z=3.6**, KSData k-grid
(`results/holdout_multid_phase1_5/holdout_multid.md`):

| metric         | value   |
|----------------|---------|
| mean rel-err   | 3.27 %  |
| p99 rel-err    | 12.08 % |
| max rel-err    | 23.93 % |

**σ_PySR / σ_GP** (selected, `results/refit_optionC_z2.6-4.2_phase1_5_ksdata/scorecard.md`):

| param      | ratio  | route             |
|------------|--------|-------------------|
| ns         | 1.40×  | PySR              |
| Ap         | 0.79×  | PySR              |
| tau0       | 1.26×  | PySR              |
| hub        | 1.26×  | PySR              |
| omegamh2   | 0.99×  | GP-slice          |
| herei      | 5.90×  | PySR              |
| heref      | 0.88×  | GP-slice (gated)  |
| alphaq     | 1.30×  | PySR              |
| hireionz   | 1.01×  | GP-slice          |
| bhfeedback | 1.00×  | GP-slice (gated)  |

---

## 2. Phase 2 — Phase 1.5 base + 4 pair cross-couplings

**What this adds to Phase 1.5**: (a) extends smart kwargs to **all 11**
per-1D refits (not just IGM block), and (b) trains 4 per-pair PySR
equations on the residual `P_GP − P̂_phase1` over Sobol samples in each
2D plane. The pair contributions enter the prediction as ANOVA pure
2-way interactions (cross_diff = G(θ_i,θ_j) − G(θ_i,fid_j) −
G(fid_i,θ_j) + G(fid_i,fid_j)). Math: `docs/ONBOARDING.md` § 2 or
`src/priya_forecast/refit_pair.py`.

**Canonical output dir**: `results/refit_phase2_production_v2_ksdata/`

### 2a. Re-run per-1D refits with `--smart` (all 11 params)

You can reuse Phase 1.5's `payloads/` directly. Re-run the per-param
refits with `--smart` (not `--auto-smart`):

```bash
sbatch \
    --export=ALL,REPO=$PWD,\
PAYLOAD_DIR=results/refit_optionC_z2.6-4.2_phase1_5_ksdata/payloads,\
OUTPUT_DIR=results/refit_phase2_production_v2_ksdata \
    --array=0-10 \
    slurm/refit_array.slurm
```

The shipped `slurm/refit_array.slurm` already uses `--smart`. This is
why Phase 2's `bhfeedback` and `heref` get real PySR equations
(σ-ratios 1.02× and 3.25× respectively) where Phase 1.5 routed them
through GP-slice.

### 2b. Precompute pair payloads (one job)

For each of the 4 chosen pairs, sample 3D Sobol over `(θ_i, θ_j, z)`
with others=fid; query GP and Phase-1 hybrid; store the residual.

```bash
python scripts/precompute_payloads_pair.py \
    --phase1-refits-dir results/refit_phase2_production_v2_ksdata/refits \
    --pairs tau0,ns herei,alphaq Ap,alphaq tau0,Ap \
    --output results/refit_phase2_production/pair_payloads_v2
```

The 4 pairs are picked from the simdat MCMC posterior correlation
matrix (the strongest off-diagonal cross-coupling pairs); see
`docs/PAIR_FIT_PLAN.md` for the selection logic.

### 2c. Per-pair refits (4 SLURM tasks)

```bash
sbatch \
    --export=ALL,REPO=$PWD,\
PAYLOAD_DIR=results/refit_phase2_production/pair_payloads_v2,\
OUTPUT_DIR=results/refit_phase2_production/pair_v2,\
PAIRS=tau0_ns:herei_alphaq:Ap_alphaq:tau0_Ap \
    --array=0-3 \
    slurm/refit_pair_array.slurm
```

Sequential alternative:

```bash
for pair in tau0,ns herei,alphaq Ap,alphaq tau0,Ap; do
    python scripts/refit_one_pair.py \
        --pair "$pair" \
        --payload-dir results/refit_phase2_production/pair_payloads_v2 \
        --output-dir  results/refit_phase2_production/pair_v2 \
        --max-retries 2
done
```

This writes `results/refit_phase2_production/pair_v2/refits/<pair>.pkl`.

### 2d. Re-aggregate with pair refits

```bash
python scripts/multi_z_aggregate.py \
    --refits-dir results/refit_phase2_production_v2_ksdata/refits \
    --pair-refits-dir results/refit_phase2_production/pair_v2/refits \
    --output results/refit_phase2_production_v2_ksdata \
    --priors production --kim-tau0 --use-ksdata
```

### 2e. Multi-D hold-out (Phase 2)

```bash
python scripts/holdout_multid.py \
    --refits-dir results/refit_phase2_production_v2_ksdata/refits \
    --pair-refits-dir results/refit_phase2_production/pair_v2/refits \
    --output results/holdout_multid_phase2_production \
    --n-sobol 64 --z-eval 3.6
```

### 2f. Off-fid closure (Phase 2)

```bash
python scripts/closure_at_simdat_target.py \
    --refits-dir results/refit_phase2_production_v2_ksdata/refits \
    --pair-refits-dir results/refit_phase2_production/pair_v2/refits \
    --truth results/simdat_ind15_truth.npz \
    --output results/closure_at_simdat_ind15_phase2_v2_ksdata
```

### 2g. Expected results — Phase 2

**11-θ joint Sobol hold-out (n=64) at z=3.6**
(`results/holdout_multid_phase2_production/holdout_multid.md`):

| metric         | value   | vs Phase 1.5 |
|----------------|---------|--------------|
| mean rel-err   | 2.35 %  | 28% better   |
| p99 rel-err    | 7.05 %  | 42% better   |
| max rel-err    | 12.11 % | 49% better   |

**σ_PySR / σ_GP** (selected, `results/refit_phase2_production_v2_ksdata/scorecard.md`):

| param      | ratio  | route                       |
|------------|--------|-----------------------------|
| ns         | 1.39×  | PySR                        |
| Ap         | 2.62×  | PySR (known limitation)     |
| tau0       | 1.33×  | GP-slice (LF rel-err 5.14%) |
| hub        | 1.27×  | PySR                        |
| omegamh2   | 0.99×  | GP-slice                    |
| herei      | 5.43×  | PySR                        |
| heref      | 3.24×  | PySR (newly routed)         |
| alphaq     | 3.56×  | PySR                        |
| hireionz   | 1.10×  | GP-slice                    |
| bhfeedback | 1.02×  | PySR (newly routed)         |

The Ap σ-ratio = 2.62× is a known limitation — see
`docs/AP_REMEDIATION_PLAN.md` for Phase 3 plan.

---

## 3. How to verify your reproduction

1. After 1c, open `results/refit_optionC_z2.6-4.2_phase1_5_ksdata/scorecard.md`.
   Compare the σ-ratios to the table in § 1e.
2. After 1d, open `results/holdout_multid_phase1_5/holdout_multid.md`
   and confirm `mean rel-err = 3.27%`, `p99 = 12.08%`.
3. After 2d, repeat for Phase 2 (§ 2g table).
4. Per-1D PySR equations are saved as both pickles and a human-readable
   summary at `results/.../per_param_summary.md`. Comparing equations
   verbatim is fragile (PySR genetic search depends on random seed and
   thread schedule), but the **σ-ratios** should match within ~5%
   because option-B operators + the dim-balanced ANOVA loss + the
   at-fid LF anchor constrain the hypothesis space tightly.

---

## 4. Troubleshooting

- **`--use-ksdata` belongs only on `multi_z_aggregate.py`** (§ 1c, § 2d).
  The hold-out script (`holdout_multid.py`) doesn't use a covariance —
  it reports rel-err in P_F space — so it has no `--use-ksdata` flag.
  The closure script (`closure_at_simdat_target.py`) always uses
  KSData unconditionally. Don't add the flag elsewhere or argparse
  will reject it.
- **Byte-identical PySR equations require fixed thread count.** PySR's
  genetic search is multithreaded by default and the schedule is not
  deterministic across runs. The **σ-ratios are stable** (option-B
  operators + ANOVA loss + at-fid anchor constrain the hypothesis
  space tightly), but the verbatim equations may differ. If you need
  byte-identical equations across reruns, set `JULIA_NUM_THREADS=1`
  before invoking the refit scripts. See `memory/pysr_gp_gotchas.md`
  and `feedback_pysr_speed.md`.
- **PySR / Julia setup errors**: see `docs/PYSR_PERFORMANCE.md` and
  `memory/pysr_gp_gotchas.md`. Most common: stale `~/.julia` after a
  PySR upgrade — `rm -rf ~/.julia && rm -rf ~/.julia_env` then re-run.
- **`No module named 'lyaemu'`**: `PYTHONPATH` must include the
  upstream `lya_emulator_full` clone first, before `src/`.
- **Per-param SLURM tasks finish in seconds**: usually means
  `payloads/<param>.pkl` is missing — re-run § 1a.
- **σ-ratios differ by >10% from the tables**: most likely you forgot
  `--use-ksdata` in `multi_z_aggregate.py`. Without it, the script
  falls back to a synthetic diagonal covariance and σ_GP changes.
- **Pair fits fail with "no x0,x1 eq found"**: `--max-retries 2` is the
  default; bump to 3-4 for stubborn pairs. Phase 2's `tau0×Ap` fit
  drops `x0` after retries — this is a graceful no-op (the
  cross_difference returns 0 contribution by design); see
  `src/priya_forecast/refit_pair.py:185-224`.

---

## 5. References

- **`docs/REPRODUCE.md`** — full reproducibility guide (companion to
  this doc): data acquisition, Julia/PySR bootstrap, cluster
  customization, step-by-step verification, common-failure recovery.
- `docs/PAPER_NOTES.md` — full design log + canonical scorecard.
- `docs/ONBOARDING.md` — math walkthrough.
- `docs/PAIR_FIT_PLAN.md` — Phase 2 pair selection + ANOVA combine math.
- `docs/AP_REMEDIATION_PLAN.md` — Phase 3 plan for the Ap σ-ratio.
- `slurm/refit_array.slurm` — the per-1D SLURM template (default `--smart`).
- `slurm/refit_pair_array.slurm` — the per-pair SLURM template.
