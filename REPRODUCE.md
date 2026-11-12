# REPRODUCE — a step-by-step guide to every figure and table

This is the from-scratch tutorial for reproducing the paper *Knowledge
Distillation with PySR on the PRIYA suite*. Follow it top to bottom.

**The headline: every figure and table of the paper is redrawn in about a minute from
committed files, with only `numpy / pandas / matplotlib / scipy / sympy / pyyaml / h5py`
and a TeX install — no GP emulator, no PySR, no Julia.** The one committed production run
(`results/paper_production_20260630_perz_sobolev_z2.6-4.2/`) ships every PySR Pareto
front, a score file next to each one (`grad_faith_<param>.csv`: the derivative error of
every candidate equation), and the saved fits. The GP emulator is needed only to
*recompute* the multi-parameter table, or to refit.

The work is split into three tiers. Most readers only need **Step 1**.

| tier | you need | reproduces | time |
|------|----------|------------|------|
| **1** | 7 light pip deps (`requirements-figures.txt`) + a TeX install | every figure; Tables 1–5 | ~1 min |
| **2** | the full venv + the GP emulator (`GPy` + `lyaemu`) + `data/kodiaq_gp` | recomputes Table 5 and the per-k curves behind Figures 3 and 4 | ~15 min setup |
| **3** | Tier 2 + Julia/PySR + SLURM | re-train the PySR fits from scratch | hours (cluster) |

Jump to the [full figure/table → command → tier map](#full-figure--table--command--tier-map) at the end.

---

## Step 0 — clone

```bash
git clone https://github.com/jibanCat/lya1d_priya_forecast.git
cd lya1d_priya_forecast
```

Everything Tier-1 needs is already in the clone. Two large inputs are **not**:
- `data/kodiaq_gp/` (~43 MB, git-ignored) — the stripped GP emulator basedir,
  needed only for **Tier 2**. Built in Step 2.
- the trained MF-emulator pickles — they live inside the emulator repo you clone
  in Step 2.

What *is* committed and does the heavy lifting for Tier 1:

```
results/paper_production_20260630_perz_sobolev_z2.6-4.2/
├── RUN_MANIFEST.md                       # exact recipe + SLURM job graph
├── value/refit/z3.6/                      # value-loss fits (maxsize 20), 11 params
├── sobolev/refit/z{2.6,3.6,4.2}/          # Sobolev fits (the headline objective)
│   ├── pareto_<p>.csv                     #   PySR Pareto front (complexity, loss, equation)
│   ├── grad_faith_<p>.csv                 #   sidecar: grad_err + value_mse per candidate
│   └── z3.6/{payloads,refits}/            #   pickled artifacts (Tier-2 prediction figs)
├── seed_band/
│   ├── z3.6_seed0_budget/refit/z3.6/      #   ns budget control (maxsize 35)
│   └── seed_band_summary.json             #   aggregated 5-seed medians/ranges
├── sens_maxsize{30,40}_{value,sobolev}/   # maxsize sweep
└── figures/                               # table .txt/.tex/.csv of record (committed); figure PDFs are regenerable
```

---

## Step 1 — Tier 1: the main results, emulator-free (~2 min)

### 1a. Environment

```bash
python3 -m venv .venv-figures
source .venv-figures/bin/activate
pip install -r requirements-figures.txt
export PYTHONPATH=src          # the three tiny emulator-free helper modules
```

`requirements-figures.txt` pins the exact 7 working versions
(`numpy 1.26.4`, `scipy 1.12.0`, `pandas 2.3.3`, `matplotlib 3.10.9`,
`sympy 1.14.0`, `PyYAML 6.0.3`, `h5py 3.16.0`). Only
`priya_forecast.{parameters, grad_faith_io, pareto_diag}` are imported — no
GPy, PySR, juliacall, or lyaemu.

> **One non-pip prerequisite for the figures.** `scripts/make_figures.py` typesets
> labels with LaTeX (`matplotlib.rcParams["text.usetex"]=True`), so it needs a TeX
> install with `latex` on `PATH` (Debian/Ubuntu:
> `apt-get install texlive-latex-base texlive-latex-extra dvipng`; conda:
> `conda install -c conda-forge texlive-core`). **The tables (1b–1d) need no TeX.**

### 1b. Table 3 (`tab:faith_taxonomy`) — the derivative-faithfulness taxonomy (the paper's main result)

Recompute the knee-selected `grad_err` for the value vs Sobolev objective at
z=3.6, straight from the committed sidecars:

```bash
PROD=results/paper_production_20260630_perz_sobolev_z2.6-4.2
python - <<'PY'
from priya_forecast.parameters import PARAM_NAMES
from priya_forecast.grad_faith_io import read_grad_faith_sidecar, knee_row
from priya_forecast.pareto_diag import GATE_TOL
P = "results/paper_production_20260630_perz_sobolev_z2.6-4.2"
knee = lambda d, p: float(knee_row(read_grad_faith_sidecar(f"{P}/{d}/refit/z3.6/grad_faith_{p}.csv"))["grad_err"])
print(f"{'param':11s}{'value':>8s}{'sobolev':>9s}  class")
for p in PARAM_NAMES:
    v, s = knee("value", p), knee("sobolev", p)
    print(f"{p:11s}{v:8.3f}{s:9.3f}  {'RESISTANT' if s > GATE_TOL else 'faithful'}")
PY
```

**Expected output** (the load-bearing rows): `ns` value 0.365 → Sobolev **0.160**;
`Ap` 0.298 → 0.155; `omegamh2` 0.697 → 0.063; `hub` 0.986 → **1.000 RESISTANT**;
`bhfeedback` 1.418 → **0.771 RESISTANT**. Nine of eleven parameters clear the
0.25 gate under Sobolev; only `hub` and `bhfeedback` resist. Cross-check against
`$PROD/figures/taxonomy_table.tex` — the knee cut, regenerated at `f804916`, whose 33
cells and 11 class labels match the paper exactly. **Do not** cross-check against the
`.txt` sibling: it is the superseded best-loss pick (ns Sobolev 0.121, not 0.160).

### 1c. Table 4 (`tab:per1d_eqs`) — the per-parameter equations at the Pareto knee

```bash
python - <<'PY'
import pandas as pd
from priya_forecast.parameters import PARAM_NAMES
from priya_forecast.grad_faith_io import read_grad_faith_sidecar, knee_row
S = "results/paper_production_20260630_perz_sobolev_z2.6-4.2/sobolev/refit/z3.6"
print(f"{'param':11s}{'cmplx':>6s}{'loss':>11s}{'grad_err':>10s}")
for p in PARAM_NAMES:
    kr = knee_row(read_grad_faith_sidecar(f"{S}/grad_faith_{p}.csv"))
    cx = int(kr["Complexity"])
    eq = pd.read_csv(f"{S}/pareto_{p}.csv").query("Complexity == @cx")["Equation"].iloc[0]
    print(f"{p:11s}{cx:6d}{float(kr['Loss']):11.4g}{float(kr['grad_err']):10.3f}  {eq}")
PY
```

**Expected output**: `dtau0` cmplx 19 loss 0.0022 grad_err 0.003; `ns` cmplx 18
loss 9.75 grad_err **0.160**; `hub` cmplx 10 loss 422 grad_err 1.000. Matches the
`% sympy:` comments above each row of Table 4 in the paper, and the committed
`$PROD/figures/per_param_equations.tex` (the knee cut, regenerated at `f804916`).
The `.txt` sibling is the superseded best-loss pick — it does not match the paper.

### 1d. Table 1 — the 11 parameters + priors

```bash
python -c "from priya_forecast.parameters import PARAMS_11D; [print(f'{p.name:11s}{p.fid:9g}{p.prior[0]:9g}{p.prior[1]:9g}') for p in PARAMS_11D]"
```

**Expected**: `ns 0.983 0.8 1.05`, `Ap 1.46 1.2 2.6` (in units of 1e-9),
`hub 0.688 0.65 0.75`. This is the code's single source of truth; the committed
copy is `$PROD/figures/param_priors_table.txt`.

> Note (resolved 2026-07-29): this used to flag a Table 1 mismatch on `z_Hei` /
> `z_Hef`. The paper now lists `z_Hei ∈ [3.5, 4.5]` and `z_Hef ∈ [2.2, 3.2]`,
> matching `parameters.py` and the GP hypercube. No discrepancy remains;
> `param_priors_table.txt` is still the emulator-box truth.

### 1e. The paper's figures

One script draws them, exactly as they appear in the manuscript (needs TeX; see 1a):

```bash
python scripts/make_figures.py --out results/_repro_scratch/paper_figures
```

**Expected**: one `<name>: done` line per figure and one PDF each. Every input is checked
against tag `v1.2-paper` before it is plotted, so the script needs a full `git clone`.
On Great Lakes, with `requirements.lock.txt` and TeX Live 2026, the PDFs are byte-identical
to the committed copies in `$PROD/figures/` and to the manuscript's. Another TeX or
matplotlib version can change the bytes but not the plotted values, which `--dump DIR`
writes out.

Older generators remain for figures the paper no longer includes
(`make_diagnostic_figs.py`, `regen_maxsize_sensitivity.py`, `regen_rescorr.py`). Nothing
in the paper depends on them.

### 1f. One-command option: the notebook

`notebooks/reproduce_paper.ipynb` redraws every figure and prints every table, in the
paper's order, and says which selection rule each one uses. It needs no emulator. To run
it headlessly (needs `pip install nbconvert ipykernel`):

```bash
python -m ipykernel install --user --name priya-venv
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=priya-venv \
    notebooks/reproduce_paper.ipynb
```

Or just open it in Jupyter and **Run All**. Redrawn figures land in
`results/_repro_scratch/` (git-ignored). `notebooks/README.md` says what each notebook is for.

---

## Step 2 — Tier 2 (optional): recompute the multi-parameter table (needs the GP emulator)

Table 5 compares the additive model with the GP emulator at new test points, so
recomputing it needs the multi-fidelity GP emulator. **Julia/PySR are NOT needed here** —
the equations are already refit and saved in the production run.

### 2a. Full environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .            # or: pip install -r requirements.lock.txt  (exact, reproducible)
pip install GPy emukit matplotlib     # the emulator extras (the `gp` + `forecast` groups)
```

### 2b. Clone the GP emulator

The emulator is `github.com/sbird/lya_emulator` (the author works from a fork).
Clone it anywhere and point `$LYA_EMULATOR` at the repo root — the importable
package `lyaemu` sits at that root:

```bash
export LYA_EMULATOR=$PWD/../lya_emulator        # any path you like
git clone https://github.com/sbird/lya_emulator "$LYA_EMULATOR"
```

### 2c. Build `data/kodiaq_gp` (the stripped GP basedir)

`data/kodiaq_gp/` is the inference-time subset of a full KODIAQ-SQUAD GP training
directory. `scripts/prep_kodiaq_gp.py` strips a full basedir down to just the
files the forecast reads (all 13 z-bin pickles + the MF flux-vector HDF5 +
`emulator_params.json`), dropping alt-cut variants, leave-one-out diagnostics,
the temperature emulator, and the `kims_*` test subdir.

```bash
# --source is the FULL upstream basedir. The committed data was built from
# (on GreatLakes / Turbo):
#   /nfs/turbo/umor-yueyingn/mfho/birdgroup/lya_xq100/kodiaq_2_2_4_6-48-48
# Substitute your own copy of that directory (obtain it from the PRIYA / KODIAQ
# emulator authors — it is the multi-fidelity GP training set of Ho et al. 2025).
python scripts/prep_kodiaq_gp.py \
    --source /path/to/kodiaq_2_2_4_6-48-48 \
    --dest data/kodiaq_gp
```

**Expected**: `Wrote stripped GP basedir to data/kodiaq_gp (~43 MB).` and a
`data/kodiaq_gp/README.md` recording the source. Provenance of the committed
copy is in that README (`Source: .../kodiaq_2_2_4_6-48-48`).

### 2d. The four env vars, then the commands

```bash
export PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env    # (only used if Julia is present)
export JULIA_DEPOT_PATH=$HOME/.julia
export PYTHONPATH=src:$LYA_EMULATOR                 # src + the emulator repo root
PY=python                                           # the .venv python (GPy + lyaemu)
PROD=results/paper_production_20260630_perz_sobolev_z2.6-4.2
```

```bash
# --- Table 5 `tab:multid`: the additive model against the GP at 256 Sobol points ---
# regen_multid takes the PARENT sobolev dir and appends refit/z<z> itself:
$PY scripts/regen_multid.py --refit-dir $PROD/sobolev --z 3.6 \
    --basedir data/kodiaq_gp --out-dir results/_repro_scratch/multid --n-sobol 256
```

Check against the committed `$PROD/figures/multid_z3.6/multid_bestworst.csv`. The numbers
are already committed, so you can *read* Table 5 without the emulator (§1f) and only need
Tier 2 to *recompute* them.

The same command rewrites `multid_bestworst_perk.csv`, the per-k curves Figures 3 and 4 are
drawn from. A rerun reproduces Table 5's four-decimal values; the sixth decimal of a few
per-k values can differ between machines. It also needs the one-parameter sweeps in
`data/single_z_1pvar` (`scripts/regen_1pvar.py`): without them the script says so and
falls back to emulator slices, which is not the result in the paper.

---

## Step 3a — Tier 3 (optional): re-run and tweak the PySR pipeline (`rerun_paper.ipynb`)

`notebooks/rerun_paper.ipynb` is a collaborator-facing tutorial notebook that re-runs the **full** symbolic
regression pipeline end-to-end — one PySR fit per (parameter, arm, redshift), the run-local 1pvar regeneration,
and the derivative-faithfulness scoring — into an isolated output directory (`results/tutorial_reruns/`). It
lets you tweak the search budget, operators, Sobolev weight, or your own fiducial/prior and see where the fits
move, then regenerate the paper's taxonomy table and figures from *your* run. It writes to a fresh directory
and never touches the committed production run.

**Requirements**: the Tier-2/3 environment — the full venv, `GPy` + `emukit` (with `numpy < 2` for the GPy
ABI), the `lyaemu` package, a GP basedir, and Julia/PySR (the notebook retrains the equations from scratch, so
PySR runs).

**Provisioning the GP emulator (two public repos).** The `lyaemu` package is in
[`github.com/jibanCat/InferenceLyaData`](https://github.com/jibanCat/InferenceLyaData). The trained
emulator the paper uses (172 k-bins) is
[`github.com/jibanCat/kodiaq-squad-gp`](https://github.com/jibanCat/kodiaq-squad-gp); its clone root
is the basedir.

```bash
git clone https://github.com/jibanCat/InferenceLyaData ../InferenceLyaData    # the lyaemu package
git clone https://github.com/jibanCat/kodiaq-squad-gp data/kodiaq-squad-gp      # the paper's emulator
pip install GPy emukit                             # numpy must stay < 2
export LYA_EMULATOR=$PWD/../InferenceLyaData        # its lyaemu/ dir is the package
export GP_BASEDIR=data/kodiaq-squad-gp              # the clone root is the basedir
```

Do **not** point `GP_BASEDIR` at `InferenceLyaData/Emulator_Files_KS`. The file loaded from there has
329 k-bins: a different emulator, whose high-fidelity P1D differs from the paper's by a few percent.
The notebook's first cell prints the number of k-bins of the basedir it found.

The notebook's first cell reads `LYA_EMULATOR` and `GP_BASEDIR` and reports what is missing; if either is
absent it prints these commands and skips the run rather than erroring.

Run the notebook in Jupyter (or headlessly via `jupyter nbconvert --execute`). Output lands in
`results/tutorial_reruns/`, which is git-ignored for isolation.

The notebook's default setting is small. To run the same grid at the paper's search budget as one
cluster job (66 fits, about 10 h on four cores), and read the result back in the notebook:

```bash
REPO=$PWD bash slurm/rerun_grid.slurm --preset full --label full_seed0 --dry-run   # print the settings only
sbatch --export=ALL,REPO=$PWD slurm/rerun_grid.slurm --preset full --label full_seed0
```

---

## Step 3 — Tier 3 (optional): re-train the PySR fits (needs Julia/PySR + SLURM)

The whole production run (per-z Sobolev + value baseline + 5-seed band + ns budget
control + maxsize sweep) is one self-documenting submit script:

```bash
scripts/submit_paper_production.sh --dry-run          # print the sbatch lines only
SLURM_ACCOUNT=<acct> LYA_EMULATOR=/path/to/lya_emulator \
    scripts/submit_paper_production.sh                # submit to GreatLakes SLURM
```

It writes `RUN_MANIFEST.md` with the exact recipe (one PySR model per (param, z),
Sobolev λ=5 on a log-P target, maxsize=20, populations=48, niterations=200; z ∈
{2.6, 3.6, 4.2}; seeds 0–4 at z=3.6; ns budget control maxsize=35). PySR/Julia are
auto-installed by `juliacall` on first run. After the jobs finish, aggregate the
seed band and regenerate the figures:

```bash
scripts/aggregate_seed_band.py \
    --band-dir $PROD/seed_band --out $PROD/seed_band/seed_band_summary.json
# then the Step 1 (Tier-1) and Step 2 (Tier-2) commands.
```

---

## Selection rules and settings

Which equation is shown where, and where each choice is set. A fit returns a Pareto front
of candidate equations; the paper chooses from it in three ways (Sec. 2.2.2).

| choice | value | defined in | governs |
|--------|-------|------------|---------|
| valid candidate | contains the parameter; no constant above 100; stays finite when each input is shifted by ±0.05 and ±0.1 around the middle of its normalised range | `pareto_filters.py`, `single_z/forecast.py` (`_filter_fisher_safe`) | every rule below except lowest-loss |
| lowest-loss | smallest training loss on the front | `refit_1d_pysr.py` (`pareto["loss"].idxmin()`) | Figures 1, 2; Table 2 |
| lowest-loss valid | smallest training loss among valid candidates that also pass the threshold | `single_z/forecast.py` (`build_refit_from_pareto_gated`) | Table 5; Figures 3, 4 |
| Pareto knee | simplest valid candidate with loss within 10% of the best valid loss | `grad_faith_io.py` (`knee_row`, `rel_tol=0.1`) | Tables 3, 4; Figure 5 |
| faithful | derivative error `<= 0.25` | `paper_figures.GATE`; the same default in `derivative_gate.py`, `grad_faith_score.py`, `eval_grad_faithfulness.py --tol` | every faithful/unfaithful label |
| derivative error | median over k-bins; bins where the GP slope is below `1e-3` of its peak are skipped | `grad_faith_score.py` (`median_rel_error`) | Tables 3, 4; Figures 5–7 |
| search budget | `niterations=200`, `populations=48`, `maxsize=20` | `scripts/submit_paper_production.sh` | every production fit |
| Sobolev weight | `λ = 5` | same script | the Sobolev fits |
| seeds, redshifts | seeds 0–4 at z = 3.6; z = 2.6, 3.6, 4.2 | same script | Figure 5, Table 3 |
| complexity control | `maxsize = 35`, value-only loss, `n_P` | same script | Figures 6, 7 |
| retries | up to 5 attempts until the front has a valid candidate | `single_z/refit.py` (`max_retries=4`) | every fit |
| operators | `+ − × aq`; `exp log square` | `refit_1d_pysr.py` (`SMART_REFIT_PYSR_KWARGS`) | every 1D fit |
| test set | 256 Sobol points, seed 1234 | `scripts/regen_multid.py` defaults | Table 5; Figures 3, 4 |
| joint fit (Sec. 5.1) | lowest-loss; `+ − × /`, `log exp square`; rank tolerance `1e-8` | `scripts/run_multid_pysr.py`; `results/joint_multid_sweep/SWEEP_SUMMARY.md` | Sec. 5.1 |

The command-line defaults of `scripts/refit_one_param_single_z.py` are smaller than the
production budget (`niterations=50`, `populations=24`, `λ=1`); the production values are
the ones the submit script passes.

---

## Full figure / table → command map

`PROD=results/paper_production_20260630_perz_sobolev_z2.6-4.2`. Numbers follow the order
of the current draft; the `\label` is the stable key.

### Figures

All of the paper's figures are drawn by `scripts/make_figures.py` (§1e) from committed
files, with no emulator. The copies under `$PROD/figures/` are the manuscript's.

| paper figure | file | `make_figures.py --only` | selection rule |
|--------------|------|--------------------------|----------------|
| Figure 1, `fig:tau0_ap_pred` | `pysr_pred_tau0_Ap.pdf` | `prediction` | lowest-loss |
| Figure 2, `fig:param_error_bars` | `param_error_bars.pdf` | `param_errors` | lowest-loss |
| Figure 3, `fig:multid_bestworst` | `multid_bestworst.pdf` | `multid` | lowest-loss valid |
| Figure 4, `fig:multid_3d` | `multid_3d.pdf` | `multid` | lowest-loss valid |
| Figure 5, `fig:seed_band` | `seed_band.pdf` | `seed_band` | Pareto knee |
| Figure 6, `fig:pareto_faith` | `pareto_faithfulness.pdf` | `pareto` | none: every valid candidate |
| Figure 7, `fig:ns_budget` | `ns_budget_panel.pdf` | `ns_budget` | none: every valid candidate |

Figures 3 and 4 are drawn from `figures/multid_z3.6/multid_bestworst_perk.csv`: the mean and
90th-percentile error in each k-bin, for each of the ten parameter combinations of Table 5.

**No longer in the paper** (commented out in the `.tex`): `resolution_correction.pdf`,
`faithfulness_scorecard.pdf`, `crossz_faithfulness.pdf`, `maxsize_sensitivity.pdf`,
`pysr_graphs_3.6_dtau0.pdf`, `2d-denorm-Sobol_dtau0-Ap.pdf`. Their committed copies remain.
The paper's text still quotes the numbers behind the cross-redshift and maxsize ones.

### Tables

| paper table | label | tier | command / source of record |
|-------------|-------|:----:|----------------------------|
| Table 1 — parameters + priors | `tab:param_table` | 1 | `priya_forecast.parameters` (§1d); `figures/param_priors_table.txt` |
| Table 2 — per-param % error (LF/HF) | `tab:stats_table` | 1 read | committed `figures/table2_stats.tex` is the record — **see the gotcha below** |
| Table 3 — faithfulness taxonomy | `tab:faith_taxonomy` | 1 | `grad_faith_io.knee_row` on the score files (§1b); `figures/taxonomy_table.tex` |
| Table 4 — per-parameter equations | `tab:per1d_eqs` | 1 | knee row + Pareto CSV (§1c); `figures/per_param_equations.tex` |
| Table 5 — multi-D combine vs GP | `tab:multid` | 1 read / 2 recompute | committed `figures/multid_z3.6/multid_bestworst.csv` (read); `scripts/regen_multid.py` (§2d) |

---

## Known gotchas

- **Figures need TeX.** `make_figures.py` uses `text.usetex=True`. Without a `latex`
  install it raises a matplotlib LaTeX error. The **tables** do not need TeX.
- **Table 2 (`tab:stats_table`) cannot be recomputed from this repository.** Its generator
  lives in the manuscript's repository and needs the emulator. Treat the committed
  `figures/table2_stats.tex` as the record.
- **numpy < 2 / pandas < 3** in the pins is only for ABI compatibility with the
  Tier-2 GPy install; the Tier-1 code itself runs fine on numpy 2.x.
