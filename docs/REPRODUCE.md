# Full reproducibility guide (companion to `README_v2.md`)

`README_v2.md` is a tested-on-Greatlakes cheatsheet: if your environment
is already set up the same way mine is, the commands there work
verbatim. **This doc fills the gaps a fresh student would hit on a new
machine** — data acquisition, Julia/PySR bootstrap, cluster
customization, verification checkpoints, and recovery from common
failures.

Skip to whichever section bites you.

---

## 1. What this repo *does* and *does not* ship

✅ **Ships**:
- All code (`src/priya_forecast/`, `scripts/`, `slurm/`).
- The canonical reference scorecards under `results/refit_optionC_z2.6-4.2_phase1_5_ksdata/`,
  `results/refit_phase2_production_v2_ksdata/`,
  `results/holdout_multid_phase{1_5,2_production}/`,
  `results/closure_at_simdat_ind15_*/`. Use these to diff against your
  reproduction.
- The simdat-MCMC closure target `results/simdat_ind15_truth.npz`
  (committed plain to git as a regular `.npz` zip — not LFS, so a
  vanilla `git clone` gets the actual file).

❌ **Does NOT ship**:
- The KODIAQ GP emulator pickle library
  (`/nfs/turbo/umor-yueyingn/mfho/birdgroup/lya_xq100/kodiaq_2_2_4_6-48-48/`).
  Build it with the upstream `lyaemu` (sbird/lya_emulator) toolchain;
  see § 2 below.
- The simdat MCMC chains used to build `simdat_ind15_truth.npz`. The
  truth npz itself is shipped, so this only matters if you want to
  regenerate it.
- A `pyproject.toml` Julia-side spec — Julia + SymbolicRegression.jl
  install themselves via `juliacall` on first PySR import.

---

## 2. Data prerequisites

### 2a. GP emulator (KODIAQ-SQUAD + XQ-100)

The KODIAQ multi-fidelity GP is the *truth model* the forecast scores
PySR equations against. The forecast loads it through
`priya_forecast.models.gp_model.GPModel`, which adapts
`lyaemu.priya_explorer.PRIYAEmulatorExplorer`.

**On Greatlakes (umor-yueyingn group)**: pickles are at
`/nfs/turbo/umor-yueyingn/mfho/birdgroup/lya_xq100/kodiaq_2_2_4_6-48-48/`.
This is the `multi_z_aggregate.py --basedir` default — you don't need
to pass it explicitly.

**On a fresh machine** (not Greatlakes / no Turbo access): you need
either a copy of those pickles from the umor-yueyingn group, or to
rebuild them from scratch with `lyaemu`
(<https://github.com/sbird/lya_emulator>) from the underlying
simulation suite. The rebuild path is out of scope for this repo —
follow the upstream README + ask Simeon Bird (`sbird`) or Yueying Ni
for access to the trained KODIAQ pickles. Once you have a local copy,
pass `--basedir /path/to/your/kodiaq_2_2_4_6-48-48` to every
`multi_z_aggregate.py` / `holdout_multid.py` /
`closure_at_simdat_target.py` invocation.

### 2b. simdat MCMC closure target (`simdat_ind15_truth.npz`)

The closure step (§ 1d / § 2f of `README_v2.md`) compares
`σ_PySR_Fisher` to `σ_MCMC_simdat` at `θ_target_simdat`. The truth
file (`results/simdat_ind15_truth.npz`) is a 6-key npz holding
`(theta_target, mcmc_mean, mcmc_sigma, mcmc_cov, mcmc_corr,
param_names)`, all shape `(11,)` or `(11,11)`.

It was built **once** from MCMC chains at `chains/simdat/s-simdat-ind15-...`
(see `docs/PAPER_NOTES.md` line ~1018 for the chain path). The repo
ships the npz; **there's no in-repo script to regenerate it**, because
it depends on external chain data that isn't in this repo. Two options
on a fresh machine:

1. **Use the shipped npz as-is.** It's a plain `.npz` (not LFS), so a
   regular `git clone` already has it.
2. **Skip the closure step entirely**. The headline σ-table and
   hold-out rel-err numbers (§ 1c, § 1e, § 2d, § 2g of `README_v2.md`)
   do not depend on the closure target. Skipping § 1d and § 2f is a
   valid reproduction of the σ-ratios and the rel-err table.

If you need to regenerate it (e.g., new MCMC at different θ_target),
write your own builder that fills the 6 keys above from your chain
files. The format is documented at `scripts/closure_at_simdat_target.py`'s
`--truth` arg-parser.

---

## 3. Software setup

### 3a. Python + extras

```bash
git clone <this repo> && cd lya1d_priya_forecast
pip install -e ".[forecast,pysr,gp,dev]"
```

The four extras (`forecast,pysr,gp,dev`) pull in `numpy/scipy/emcee`
(forecast), `pysr` (PySR), `GPy/emukit` via lyaemu (gp), and
`pytest/hypothesis/ruff` (dev). All four are required for end-to-end.

### 3b. PYTHONPATH (order matters!)

```bash
export PYTHONPATH=/path/to/lya_emulator_full:$PWD/src
```

**The upstream `lya_emulator_full` clone MUST come BEFORE `$PWD/src`**
on `PYTHONPATH`. The forecast imports `lyaemu` first to construct the
GP, then imports `priya_forecast` for everything else. Reversing the
order silently breaks the GP wrapper; see `memory/pysr_gp_gotchas.md`.

### 3c. Julia + PySR bootstrap

```bash
export PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env
export JULIA_DEPOT_PATH=$HOME/.julia
```

PySR uses `juliacall` to spin up a Julia subprocess, which pulls
`SymbolicRegression.jl` into `$PYTHON_JULIAPKG_PROJECT`. **First run
takes 5–15 minutes** to download Julia + compile the package. Don't
panic if `precompute_payloads.py` looks frozen.

**If the bootstrap fails** with cryptic Julia stack traces (most
common: stale `juliapkg` env after a PySR upgrade), try the **least
destructive fix first** — just clear the project env, leave the depot:

```bash
rm -rf "$PYTHON_JULIAPKG_PROJECT"           # only the PySR project env
python -c "from pysr import PySRRegressor; print('ok')"
```

The depot at `$JULIA_DEPOT_PATH` ($HOME/.julia by default) holds
compiled artifacts shared across **all** your Julia projects (Pluto,
IJulia, other research code) — only nuke it as a last resort:

```bash
rm -rf "$JULIA_DEPOT_PATH" "$PYTHON_JULIAPKG_PROJECT"   # nuclear option
```

Re-bootstrap from a clean depot takes 5–15 minutes. After it works
once, subsequent runs hit the depot cache and start in ~10 seconds.

See `docs/PYSR_PERFORMANCE.md` for tuning notes (multithreading,
deterministic mode, etc.).

---

## 4. Cluster customization

The shipped `slurm/refit_array.slurm` and `slurm/refit_pair_array.slurm`
hardcode:

- `#SBATCH --account=cavestru0` — change to your account.
- `#SBATCH --partition=standard` — change if your cluster uses
  different partition names.
- `#SBATCH --mem=8G --time=1:00:00 --ntasks=4` — bump for slow
  per-param fits or many-thread PySR.
- Hardcoded shared mamba python: `PY=/sw/pkgs/arc/mamba/py3.11/bin/python`
  in `refit_array.slurm:57`. Replace with your interpreter path
  (`which python` on a login node usually works).

### Without a cluster (laptop / workstation)

The sequential fallbacks in `README_v2.md` § 1b and § 2c work fine.
Per-param refits are ~3 min each → 11 × 3 = ~35 min total for
Phase 1.5 per-1Ds, plus ~4 × 5 = 20 min for the four pair fits, plus
~10 min for aggregation and hold-out. Total ~1 hour wall-time for a
full Phase 2 reproduction on a single machine, dominated by PySR
genetic search.

---

## 5. Step-by-step verification

After each phase, run a quick sanity check before moving on. If a
step's check fails, see § 6 below.

### After § 1a (precompute payloads)

The exact directory you're verifying depends on which `--output` you
gave `precompute_payloads.py`. The canonical Phase 1.5/2 payloads in
this repo live at `results/refit_phase2_production/payloads/`; if you
ran with the README_v2.md examples, they're at
`results/refit_optionC_z2.6-4.2_phase1_5_ksdata/payloads/`. Replace the
path below as needed.

```bash
PAYLOADS=results/refit_phase2_production/payloads
ls "$PAYLOADS"/*.pkl | wc -l                    # expect 11
python -c "
import pickle
d = pickle.load(open('$PAYLOADS/ns.pkl','rb'))
print('top keys:', list(d.keys()))
inner = d['payload']
print('payload subkeys:', list(inner.keys()))
print('flux_lf_z shape:', inner['flux_lf_z'].shape, '(n_sobol_rows, n_k)')
print('z_per_row shape:', inner['z_per_row'].shape, '(z value per Sobol row)')
"
```

Expected output:

- `top keys` → `['param_name', 'payload', 'norm', 'k_grid', 'z_min', 'z_max', 'z_grid_in_range']`
- `payload subkeys` → `['params_lf', 'params_hf', 'kfkms_lf_z', 'kfkms_hf_z', 'flux_lf_z', 'flux_hf_z', 'z_per_row', 'kfkms_lf_min', 'kfkms_lf_max', 'sobol_seed']`
- `flux_lf_z shape` → `(225, 32)` (the 225 Sobol-(θ_i, z) rows, 32 k-bins; z is encoded in `z_per_row`, NOT a separate axis)
- `z_per_row shape` → `(225,)`

### After § 1b (per-1D refits)

```bash
ls results/refit_optionC_z2.6-4.2_phase1_5_ksdata/refits/*.pkl | wc -l
# Expect: 11
python -c "
import pickle
for p in ['ns','Ap','herei','heref','alphaq','bhfeedback']:
    r = pickle.load(open(f'results/refit_optionC_z2.6-4.2_phase1_5_ksdata/refits/{p}.pkl','rb'))
    has_x0 = 'x0' in r.equation_str
    print(f'{p}: x0={has_x0} eq={r.equation_str[:60]}...')
"
# Expect ns/Ap/herei/heref/alphaq to use x0; bhfeedback may drop x0
# (gets routed to GP-slice in 1c).
```

### After § 1c (aggregation)

```bash
head -25 results/refit_optionC_z2.6-4.2_phase1_5_ksdata/scorecard.md
# Expect a table with σ_GP, σ_PySR, σ_PySR/σ_GP per param.
# Compare to the table in README_v2.md § 1e — values should match
# within ~5%.
```

Compare against the canonical scorecard:

```bash
diff <(grep -E "^\| (param|tau0|ns|Ap|herei|heref|alphaq|hub|omegamh2|hireionz|bhfeedback)" \
        results/refit_optionC_z2.6-4.2_phase1_5_ksdata/scorecard.md) \
     <(git show HEAD:results/refit_optionC_z2.6-4.2_phase1_5_ksdata/scorecard.md \
        | grep -E "^\| (param|tau0|ns|Ap|herei|heref|alphaq|hub|omegamh2|hireionz|bhfeedback)")
```

This filters down to the σ-ratio rows and ignores everything else
(equation strings, complexity, run timestamps). PySR's genetic search
is non-deterministic across runs (§ 7), so the full unfiltered diff
will always show equation-string drift even when the σ-ratios are
correct.

### After § 1d (hold-out)

```bash
grep -E "^- (mean rel-err|p99|max):" \
    results/holdout_multid_phase1_5/holdout_multid.md
```

Expected output (Phase 1.5):

```
- mean rel-err: 3.27%
- p99:          12.08%
- max:          23.93%
```

(The `^- ...:` anchor avoids matching the table-header row that also
contains `mean rel-err` and `p99` as column labels.)

### After Phase 2 § 2c (pair refits)

```bash
ls results/refit_phase2_production/pair_v2/refits/*.pkl
# Expect 4 files: tau0_ns.pkl, herei_alphaq.pkl, Ap_alphaq.pkl, tau0_Ap.pkl.
```

### After Phase 2 § 2d (re-aggregate)

```bash
grep -E "ns|Ap|heref|alphaq|bhfeedback" \
    results/refit_phase2_production_v2_ksdata/scorecard.md | head -10
# Compare to README_v2.md § 2g table.
```

---

## 6. Common failure recovery

| Symptom | Cause | Fix |
|---|---|---|
| `precompute_payloads.py` hangs at "loading lyaemu" | First-run Julia bootstrap | Wait 10–15 min on first run; subsequent runs hit cache (§ 3c) |
| `ImportError: lyaemu` | `PYTHONPATH` ordering wrong | Put `lya_emulator_full` BEFORE `src` (§ 3b) |
| `refit_one_param.py` prints `WARNING: all N retries failed (x0=False, rel_err_ok=...)` and saves a no-x0 eq | Stubborn weak-coupling param (`bhfeedback`, `omegamh2`); 4 is the **default** retry count (`refit_one_param.py:144`) — bumping further sometimes helps | Either bump `--max-retries 6` and rerun, or accept the no-x0 fit (it gets GP-sliced in § 1c) |
| Scorecard shows `route = GP-slice (gated)` for several params + log lines `[gate] WARNING: dropping refit for 'X' (no x0 term)` | The aggregator's quality gate (`multi_z_aggregate.py:111-129`) drops refits whose eq has no `x0` or whose training-set mean rel-err exceeds 5% | Expected for some params (`heref`, `bhfeedback` in Phase 1.5; `dtau0` always). If MORE than 4 params get gated, your refits regressed — check the per-param `lf_train_mean_rel_err` field, then rerun with `--smart` or higher `--niter` |
| σ values shifted by ~50% from canonical (no params dropped, all routed PySR) | Forgot `--use-ksdata` → aggregator falls back to a synthetic-diagonal covariance set by `--cov-diag-frac` (default 5% of `P_F(fid, k)`) | Add `--use-ksdata` to `multi_z_aggregate.py` (§ 1c, § 2d) |
| σ-ratios off by 50%+ from canonical even with `--use-ksdata` | Forgot `--priors production --kim-tau0` | Add both flags |
| σ-ratios match but PySR equations differ verbatim | PySR thread non-determinism | Set `JULIA_NUM_THREADS=1` for byte-identical eqs (§ 4 of README_v2.md) |
| SLURM account error | `--account=cavestru0` hardcoded | `sed -i s/cavestru0/YOURACCOUNT/ slurm/*.slurm` |
| Pair fit log shows `no eq using both x0 and x1; saving best-anyway` for `tau0×Ap` (`refit_one_pair.py`) | Known: this pair has weak coupling and PySR can't find a 2-feature eq within the retry budget | Graceful no-op — `Refit2DPairResult.cross_difference` returns 0 contribution by design when the eq drops `x0` (`refit_pair.py:185-224`). The Phase 2 scorecard's Fisher numbers are byte-identical to a 3-pair run for this reason |

---

## 7. What "matches canonical" means

PySR's genetic search is non-deterministic across runs by default
(multithreaded scheduler). The numerical tolerances below are
**rough heuristics from session-to-session experience, not measured
run-to-run variance**. Treat them as "if you blow these by an order of
magnitude, definitely investigate; if you're inside them, probably
fine":

- ✅ **σ_PySR/σ_GP within ±5%** of the canonical table per parameter.
  The architectural levers (option-B operators, ANOVA loss, at-fid LF
  anchor, 5%-rel-err gate) constrain the hypothesis space tightly.
- ✅ **Hold-out rel-err mean/p99 within ±10%** of the canonical
  numbers (3.27%/12.08% Phase 1.5; 2.35%/7.05% Phase 2).
- ✅ **GP-slice routing identical** for `omegamh2`, `hireionz`, plus
  whichever weak-coupling params drop x0 in your run (typically
  `bhfeedback` for Phase 1.5, none for Phase 2).
- ⚠️ **Equation strings will differ** verbatim. Set
  `JULIA_NUM_THREADS=1` if you need byte-identical equations (slower).
- ⚠️ **Pareto-pick complexity may shift by ±2** between runs.

If your σ-ratios are off by more than ±5% per param, that's likely a
real reproduction failure — investigate (§ 6 above). If it's just the
equation strings that differ, you're fine.

---

## 8. Where to read more

- `README_v2.md` — the cheatsheet this guide expands.
- `docs/PAPER_NOTES.md` — full design log, especially § 5 (Fisher
  pipeline modifications) and § D7 (Phase 2 design).
- `docs/ONBOARDING.md` — math walkthrough.
- `docs/PAIR_FIT_PLAN.md` — Phase 2 pair selection rationale.
- `memory/pysr_gp_gotchas.md` — known PySR/GP integration bugs.
- `memory/feedback_pysr_speed.md` — multithreading vs determinism
  trade-off.
