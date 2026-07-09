# REPRODUCE — a step-by-step guide to every figure and table

This is the from-scratch tutorial for reproducing the paper *Knowledge
Distillation with PySR on the PRIYA suite*. Follow it top to bottom.

**The headline: the paper's main results reproduce in ~2 minutes with only
`numpy / pandas / matplotlib / scipy / sympy / pyyaml / h5py` — no GP emulator,
no PySR, no Julia.** The one committed production run
(`results/paper_production_20260630_perz_sobolev_z2.6-4.2/`) shipped, next to
every PySR Pareto CSV, a **grad-faith sidecar** carrying the two
emulator-grounded numbers each diagnostic needs. Replaying those CSVs
regenerates the derivative-faithfulness taxonomy (Table 6), the equation table
(Table 7), and all six diagnostic figures **emulator-free**. Only the four
*prediction* figures and the multi-D combine need the GP emulator.

The work is split into three tiers. Most readers only need **Step 1**.

| tier | you need | reproduces | time |
|------|----------|------------|------|
| **1** | 7 light pip deps (`requirements-figures.txt`) + a TeX install | 6 diagnostic figures + Tables 1, 2, 6, 7 | ~2 min |
| **2** | the full venv + the GP emulator (`GPy` + `lyaemu`) + `data/kodiaq_gp` | 4 prediction/multi-D figures + Table 2/3 regen | ~15 min setup |
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

> **One non-pip prerequisite for the four diagnostic figures.**
> `scripts/make_diagnostic_figs.py` renders labels with LaTeX
> (`matplotlib.rcParams["text.usetex"]=True`), so it needs a TeX install with the
> `latex` + `dvipng` binaries on `PATH` (Debian/Ubuntu:
> `apt-get install texlive-latex-base texlive-latex-extra dvipng`; conda:
> `conda install -c conda-forge texlive-core`). **The table reproductions
> (1b–1d) and the `seed_band`/`maxsize` figures need no TeX.** If you have no
> TeX and only want the numbers, skip 1e and run 1b–1d.

### 1b. Table 6 — the derivative-faithfulness taxonomy (the paper's main result)

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
`$PROD/figures/taxonomy_table.txt` (which tabulates the best-loss pick + seed band).

### 1c. Table 7 — the per-parameter equations at the Pareto knee

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
`% sympy:` comments above each row of Table 7 in the paper, and the committed
`$PROD/figures/per_param_equations.txt` (which shows the best-loss pick).

### 1d. Table 1 — the 11 parameters + priors

```bash
python -c "from priya_forecast.parameters import PARAMS_11D; [print(f'{p.name:11s}{p.fid:9g}{p.prior[0]:9g}{p.prior[1]:9g}') for p in PARAMS_11D]"
```

**Expected**: `ns 0.983 0.8 1.05`, `Ap 1.46 1.2 2.6` (in units of 1e-9),
`hub 0.688 0.65 0.75`. This is the code's single source of truth; the committed
copy is `$PROD/figures/param_priors_table.txt`.

> Note: the paper's Table 1 lists `z_Hei` max as 4.1 and `z_Hef` min as 2.6,
> whereas the GP hypercube (and `parameters.py`) uses `herei ∈ [3.5, 4.5]`,
> `heref ∈ [2.2, 3.2]`. This is a known open authoring discrepancy flagged with
> an `\mfho{}` note in the `.tex`; `param_priors_table.txt` is the emulator-box
> truth.

### 1e. The six diagnostic figures

Four come from **one** invocation (needs TeX; see the note in 1a):

```bash
PROD=results/paper_production_20260630_perz_sobolev_z2.6-4.2
python scripts/make_diagnostic_figs.py \
    --value-dir   $PROD/value/refit/z3.6 \
    --sobolev-dir $PROD/sobolev/refit/z3.6 \
    --budget-dir  $PROD/seed_band/z3.6_seed0_budget/refit/z3.6 \
    --crossz-dirs 2.6=$PROD/sobolev/refit/z2.6 \
                  3.6=$PROD/sobolev/refit/z3.6 \
                  4.2=$PROD/sobolev/refit/z4.2 \
    --out-dir     results/_repro_scratch/diagnostic
```

**Expected**: `wrote 4 figures (png+pdf) to results/_repro_scratch/diagnostic` —
`pareto_faithfulness`, `faithfulness_scorecard`, `ns_budget_panel`,
`crossz_faithfulness` (`.png` + `.pdf`).

The fifth, `maxsize_sensitivity` (no TeX needed):

```bash
python scripts/regen_maxsize_sensitivity.py --prod $PROD --z 3.6 \
    --out-dir results/_repro_scratch/maxsize
```

**Expected**: a `maxsize_sensitivity.{png,pdf,csv}`; the printed table ends with
`bhfeedback sobolev | 0.771 0.576 0.676 0.750`.

The sixth, `seed_band`, is a few lines of matplotlib over the committed JSON —
run it via the notebook (§1.3) or inline:

```bash
python - <<'PY'
import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from priya_forecast.parameters import PARAM_NAMES
s = json.load(open("results/paper_production_20260630_perz_sobolev_z2.6-4.2/seed_band/seed_band_summary.json"))
params = [p for p in PARAM_NAMES if p in s["params"]]
fig, ax = plt.subplots(figsize=(12, 5))
for i, p in enumerate(params):
    ax.plot(i - 0.1, min(s["params"][p]["value"][0], 1.25), "o", color="#d6604d")
    ax.plot(i + 0.1, min(s["params"][p]["sobolev"][0], 1.25), "s", color="#1a9850")
ax.axhline(s["gate"], ls="--", color="k"); ax.set_xticks(range(len(params)))
ax.set_xticklabels(params, rotation=30, ha="right")
fig.savefig("results/_repro_scratch/seed_band.png", dpi=120)
print("gate", s["gate"], "| ns sobolev band", [round(x, 3) for x in s["params"]["ns"]["sobolev"][:3]])
PY
```

**Expected**: `gate 0.25 | ns sobolev band [0.212, 0.123, 0.246]` (median, min, max
over 5 seeds) — derivative faithfulness is seed-dependent right at the gate.

### 1f. One-command option: the single whole-paper notebook

`notebooks/reproduce_paper.ipynb` is **the one notebook that reproduces every figure
and table in the paper**. Tier-1 (Tables 1/2/6/7 + the `pareto_faithfulness`,
`faithfulness_scorecard`, `ns_budget_panel`, `seed_band` figures) runs emulator-free;
the **Tier-2 GP-backed cells** (the `pysr_pred_tau0_Ap` / `pysr_graphs_3.6_dtau0` /
`multid_bestworst` prediction plots + Table 3) run automatically **if the GP emulator
is available** (Step 2), else print the exact command and skip — the notebook always
completes. To run it headlessly (needs `pip install nbconvert ipykernel`):

```bash
python -m ipykernel install --user --name priya-venv
# Tier-2 cells also need the Step-2 env vars (LYA_EMULATOR, FIGREPO, data/kodiaq_gp):
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=priya-venv \
    notebooks/reproduce_paper.ipynb
```

Or just open it in Jupyter and **Run All**. Reproduced figures land in
`results/_repro_scratch/` (git-ignored).

---

## Step 2 — Tier 2 (optional): the prediction + multi-D figures (needs the GP emulator)

The four prediction figures compare a PySR equation's P1D against the GP truth,
so they need the multi-fidelity GP emulator. **Julia/PySR are NOT needed here** —
the equations are already refit and pickled in the production run.

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
# The Fig 1/3/4 + Table 3 regen scripts live in the paper's LaTeX repo — clone it:
FIGREPO=$PWD/../Knowledge-Distillation-using-PySR-with-PRIYA-suite
git clone https://github.com/jibanCat/Knowledge-Distillation-using-PySR-with-PRIYA-suite.git "$FIGREPO"
```

```bash
# --- Fig 1: pysr_pred_tau0_Ap (tau0 & Ap 1D variation) ---
# regen_fig1.py takes --refit-dir (point it at the z-dir holding payloads/ + refits/):
$PY $FIGREPO/scripts/regen_fig1.py --refit-dir $PROD/sobolev/refit/z3.6 \
    --out-dir $PROD/figures                    # -> $PROD/figures/pysr_pred_tau0_Ap.pdf

# --- Fig 4: pysr_graphs_3.6_dtau0 (predicted-vs-true P1D, one param) ---
# --refit-dir must be the z-dir holding payloads/ + refits/ (NOT $PROD/sobolev):
$PY $FIGREPO/scripts/regen_fig3.py --param dtau0 --z 3.6 \
    --refit-dir $PROD/sobolev/refit/z3.6 --out-dir $PROD/figures

# --- 2D de-norm scatter: 2d-denorm-Sobol_dtau0-Ap ---
$PY $FIGREPO/scripts/regen_fig4.py --params dtau0 Ap --z 3.6 \
    --refit-dir $PROD/sobolev/refit/z3.6 --basedir data/kodiaq_gp --out-dir $PROD/figures

# --- Fig 3 (multi-D best/worst) + Table 2 backing CSV ---
# regen_multid takes the PARENT sobolev dir and appends refit/z<z> itself:
$PY scripts/regen_multid.py --refit-dir $PROD/sobolev --z 3.6 \
    --basedir data/kodiaq_gp --out-dir $PROD/figures/multid_z3.6 --n-sobol 256
```

Check against the committed outputs: `$PROD/figures/pysr_pred_tau0_Ap.pdf`,
`pysr_graphs_3.6_dtau0.pdf`, `2d-denorm-Sobol_dtau0-Ap.pdf`,
`multid_z3.6/multid_bestworst.csv`. The multi-D numbers are already committed, so
you can *read* Table 2 without the emulator — see 1f / notebook — and only need
Tier 2 to *regenerate* them.

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

**Provisioning the GP emulator (one public repo has both parts).** The package and the trained GP data both
live in [`github.com/jibanCat/InferenceLyaData`](https://github.com/jibanCat/InferenceLyaData):

```bash
git clone https://github.com/jibanCat/InferenceLyaData ../InferenceLyaData
pip install GPy emukit                             # numpy must stay < 2
export LYA_EMULATOR=$PWD/../InferenceLyaData        # its lyaemu/ dir is the package
export GP_BASEDIR=$LYA_EMULATOR/Emulator_Files_KS   # the KODIAQ-SQUAD GP basedir
# optional: strip the 104 MB basedir down to ~20 MB
# python scripts/prep_kodiaq_gp.py --source $GP_BASEDIR --dest data/kodiaq_gp && export GP_BASEDIR=data/kodiaq_gp
```

The notebook's first cell reads `LYA_EMULATOR` and `GP_BASEDIR` and reports what is missing; if either is
absent it prints these commands and skips the run rather than erroring.

Run the notebook in Jupyter (or headlessly via `jupyter nbconvert --execute`). Output lands in
`results/tutorial_reruns/`, which is git-ignored for isolation.

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

## Full figure / table → command → tier map

`PROD=results/paper_production_20260630_perz_sobolev_z2.6-4.2`.
Tier-1 commands are emulator-free (`PYTHONPATH=src python …`, TeX only for the
four diagnostic figures). Tier-2 commands need the Step-2 env; `FIGREPO` is the
paper's LaTeX repo.

### Figures

The paper's build loads **six** figures. Everything else below was dropped by the author
and is commented out in the `.tex` — the generators still work, but nothing in the PDF uses
them. The committed copies under `$PROD/figures/` are byte-identical to the paper's `figs/`.

**Active in the built PDF:**

| paper figure | file | tier | command |
|--------------|------|:----:|---------|
| Pareto faithfulness (`fig:pareto_faith`) | `pareto_faithfulness.pdf` | 1 | `make_diagnostic_figs.py` (§1e) |
| ns budget panel (`fig:ns_budget`) | `ns_budget_panel.pdf` | 1 | `make_diagnostic_figs.py` (§1e) |
| Resolution correction (`fig:rescorr_plot`) | `resolution_correction.pdf` | 1 | `scripts/regen_rescorr.py --out-dir <dir>` (no GP; needs TeX) |
| Across-seed band (`fig:seed_band`) | `seed_band.pdf` | 1 | `paper_figures.plot_seed_band` — see caveat below |
| tau0 & Ap prediction (`fig:tau0_ap_pred`) | `pysr_pred_tau0_Ap.pdf` | 2\* | `$FIGREPO/scripts/regen_fig1.py --refit-dir $PROD/sobolev/refit/z3.6` (§2d) |
| Multi-D best/worst (`fig:multid_bestworst`) | `multid_bestworst.pdf` | 2 | `scripts/regen_multid.py` (§2d) |

**Dropped by the author** (commented out in the `.tex`; not in the built PDF):
`faithfulness_scorecard.pdf`, `crossz_faithfulness.pdf` (both `make_diagnostic_figs.py`),
`maxsize_sensitivity.pdf` (`regen_maxsize_sensitivity.py`), `pysr_graphs_3.6_dtau0.pdf`
(`$FIGREPO/scripts/regen_fig3.py`), `2d-denorm-Sobol_dtau0-Ap.pdf` (`$FIGREPO/scripts/regen_fig4.py`),
`holdout_validation_{cosmo,astro}` (superseded by `multid_bestworst`).

Three caveats worth knowing before you trust a regenerated figure:

- **`pysr_pred_tau0_Ap` (tier 2\*)** needs no GP — `regen_fig1.py` reads committed pickles and runs
  in ~3 s — but the generator lives in the **paper** repo, not here, and `--refit-dir` is mandatory.
  Its default (`results/refit_phase2_production`) silently produces a *different* figure from an older
  refit set. With the refit-dir above, the output is pixel-identical to the published PDF.
- **`multid_bestworst`** requires the GP (`GPModel`, ~20 min load) and re-draws its Sobol sample, so a
  rerun agrees with the published figure only to ~4 significant figures, never byte- or pixel-exactly.
  The best/worst parameter combinations are stable.
- **`seed_band`** plots exactly the committed `seed_band/seed_band_summary.json`, but the published PDF
  was hand-tuned in an interactive session (title, legend text, canvas size) on top of
  `paper_figures.plot_seed_band`. `plot_seed_band` reproduces the *data and labels*; it does not
  reproduce the shipped canvas pixel-for-pixel.

Figures are compared by content, not bytes: matplotlib stamps a `CreationDate`, so rasterize first —
`gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r150 -sOutputFile=out.png in.pdf` — then compare the PNGs.

### Tables

| paper table | label | tier | command / source of record |
|-------------|-------|:----:|----------------------------|
| Table 1 — parameters + priors | `tab:param_table` | 1 | `priya_forecast.parameters` (§1d); `figures/param_priors_table.txt` |
| Table 2 — multi-D combine vs GP | `tab:multid` | 1 read / 2 regen | committed `figures/multid_z3.6/multid_bestworst.csv` (read); `scripts/regen_multid.py` (regen, §2d) |
| Table 3 — per-param % error (LF/HF) | `tab:stats_table` | 2/3 | committed `figures/table2_stats.tex` is the record; `$FIGREPO/scripts/regen_table2.py` regenerates from the refit/payload pickles — **see caveat below** |
| Table 4 — RMSE / % error 1D/2D/3D | `tab:rmse_pe_table` | — | **stale**: pre-reframe hand-picked subsets, not regenerated (author `\mfho` note in the `.tex`) |
| Table 5 — z=2.8 % error | `tab:stats_28_table` | — | **stale**: z=2.8 is not in the production grid (2.6/3.6/4.2); cannot be regenerated as-is |
| Table 6 — faithfulness taxonomy | `tab:faith_taxonomy` | 1 | `grad_faith_io.knee_row` on the sidecars (§1b); `figures/taxonomy_table.txt` |
| Table 7 — per-parameter equations | `tab:per1d_eqs` | 1 | knee row + Pareto CSV (§1c); `figures/per_param_equations.txt` |

---

## Known gotchas

- **Diagnostic figures need TeX.** `make_diagnostic_figs.py` uses
  `text.usetex=True`. Without a `latex`/`dvipng` install it raises a
  matplotlib LaTeX error. The **table** reproductions and the
  `seed_band`/`maxsize` figures do not.
- **`regen_fig1.py` *defaults* `--refit-dir` to `results/refit_phase2_production`**
  (the older refit set). Pass `--refit-dir $PROD/sobolev/refit/z3.6` explicitly — as
  §2d does, and as `regen_fig3.py`/`regen_fig4.py` require — to point it at the
  production Sobolev fits. (No symlink workaround is needed; the earlier hard-coded
  path was replaced by the `--refit-dir` flag on 2026-06-30.)
- **Table 3 (`table2_stats.tex`) does not cleanly re-run from the committed
  pickles.** `regen_table2.py` expects a payload schema (`pld['payload']`,
  `z_per_row`) that the production `sobolev/refit/z3.6/payloads/*.pkl` do **not**
  use (they are the flatter `--save-artifacts` schema), and the older
  `results/refit_phase2_production/payloads` it defaults to are a different refit
  set whose numbers don't match the paper. Treat the committed
  `figures/table2_stats.tex` as the artifact of record; a faithful regeneration
  needs the specific refit/payload pickles that produced it (re-run the fits,
  Tier 3, saving artifacts in the schema `regen_table2.py` expects).
- **numpy < 2 / pandas < 3** in the pins is only for ABI compatibility with the
  Tier-2 GPy install; the Tier-1 code itself runs fine on numpy 2.x.
