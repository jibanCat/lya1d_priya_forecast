# priya-forecast — symbolic distillation of the PRIYA Lyα P1D emulator

Distill the **PRIYA** multi-fidelity Gaussian-process (GP) emulator of the Lyman-α
1D flux power spectrum (P1D) into compact, per-parameter symbolic equations (via
**PySR**), then run a **derivative-faithfulness diagnostic** on them. The diagnostic
asks the question a Fisher forecast actually depends on: does an equation that
reproduces the GP's *values* also reproduce its *slopes* `∂P_F/∂θ`? An equation can
be value-accurate to a percent yet get the slope wrong — a **"Fisher's Mirage."**
We score all 11 PRIYA parameters as derivative-faithful or not, explain which resist
and why, and show what a **Sobolev** derivative-matching loss does and does not fix.

Companion code to the paper *Knowledge Distillation with PySR on the PRIYA suite*
(see **[Citation](#citation)**).

## Key results
- At **z = 3.6**, the Sobolev objective yields a derivative-faithful equation for
  **9 of 11** parameters — including the primordial amplitude and tilt — where the
  ordinary value loss leaves the slope biased.
- Only the Hubble parameter **h** and AGN feedback **ε_AGN** resist every method
  (their P1D response is weak/degenerate).
- The cure comes from the **training objective, not a bigger search**: value-loss
  faithfulness is budget- and seed-fragile; the Sobolev loss is faithful at the
  smallest budget.

Full diagnostic + metric definitions: **[`docs/PARETO_FAITHFULNESS_WALKTHROUGH.md`](docs/PARETO_FAITHFULNESS_WALKTHROUGH.md)**.
Where each result comes from in the code: **[`docs/CODE_REVIEW_MAP.md`](docs/CODE_REVIEW_MAP.md)**.

---

## Installation

Import name `priya_forecast`; requires **Python ≥ 3.11**. Pick the tier you need.

### Tier 1 — figures only (emulator-free, ~2 min)
Reproduces the paper's main tables + diagnostic figures from the committed sidecars.
**No GP, PySR, or Julia.**

```bash
git clone <repo-url> lya1d_priya_forecast && cd lya1d_priya_forecast
python3.11 -m venv .venv-figures && source .venv-figures/bin/activate
pip install -r requirements-figures.txt      # 7 light deps: numpy scipy pandas matplotlib sympy pyyaml h5py
export PYTHONPATH=src
```
> Rendering the four LaTeX-labelled diagnostic figures also needs a TeX install
> (`latex` + `dvipng` on `PATH`). The table reproductions and the seed-band/maxsize
> figures need no TeX.

### Tier 2 — full package + GP emulator (prediction figures)
Adds the multi-fidelity GP so the prediction figures and multi-D combine run.
Julia/PySR are **not** needed (equations are already refit + pickled).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt         # exact pinned env (numpy 1.26.4, GPy 1.13.2, …)
pip install -e ".[gp,forecast]"              # the priya_forecast package + GP/forecast extras
```
External prerequisites (Tier 2+):
1. **Upstream emulator** — `git clone https://github.com/sbird/lya_emulator "$LYA_EMULATOR"`,
   then `export PYTHONPATH=src:$LYA_EMULATOR` (the importable `lyaemu` package is at that root).
2. **GP basedir** `data/kodiaq_gp/` (~43 MB, gitignored) — build via
   `python scripts/prep_kodiaq_gp.py --source /path/to/kodiaq_2_2_4_6-48-48 --dest data/kodiaq_gp`
   (the Ho et al. 2025 MF-GP training set; obtain from the PRIYA/KODIAQ authors).

### Tier 3 — re-fit the PySR equations (Julia/PySR + SLURM)
```bash
pip install -e ".[gp,forecast,pysr,hpo]"
export PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia   # Julia/PySR auto-installed on first run
scripts/submit_paper_production.sh --dry-run                                    # then submit to SLURM
```

### Notes
- **numpy < 2 is required:** GPy 1.13.2's Cython extensions are built against numpy
  1.x's dtype ABI; numpy 2.x raises `numpy.dtype size changed`. The lockfile pins
  compatible versions (`pyproject.toml` caps `numpy<2`, `pandas<3`).
- Optional extras (`pyproject.toml`): `forecast` (emcee, getdist, matplotlib) ·
  `gp` (GPy, emukit) · `pysr` · `hpo` (optuna) · `dev` (pytest, hypothesis, ruff).
  Note `matplotlib` is only in `forecast` — the Tier-1 figure path uses
  `requirements-figures.txt`, not a bare `pip install -e .`.

### Sanity check (emulator-free)
```bash
PYTHONPATH=src pytest -q -k "not slow"       # ~436 passed; emulator/PySR tests skip on a bare clone
```

---

## Quickstart — reproduce the diagnostic figures (emulator-free)
Reads the committed Pareto fronts + grad-faithfulness sidecars and rebuilds the four
diagnostic figures with **no GP/PySR/Julia**:

```bash
PYTHONPATH=src python scripts/make_diagnostic_figs.py --out-dir results/_repro_scratch/diagnostic
```
Produces `pareto_faithfulness`, `faithfulness_scorecard`, `ns_budget_panel`,
`crossz_faithfulness` (PNG + PDF). Defaults read the committed production run
`results/paper_production_20260630_perz_sobolev_z2.6-4.2/`.

## Reproduce the whole paper
**One notebook — [`notebooks/reproduce_paper.ipynb`](notebooks/reproduce_paper.ipynb)**
regenerates **every figure and table**: Tier 1 (Tables 1/2/6/7 + the diagnostic
figures) runs emulator-free; Tier 2 (the PySR-vs-GP prediction plots + Table 3) runs
the GP-backed scripts if the emulator is available, else prints the command. Full
step-by-step guide: **[`REPRODUCE.md`](REPRODUCE.md)**.

## Use it as a library
The diagnostic is `priya_forecast.pareto_diag` / `priya_forecast.paper_figures`
(pure, emulator-free):

```python
from priya_forecast.pareto_diag import load_front, render_grid
P = "results/paper_production_20260630_perz_sobolev_z2.6-4.2/sobolev/refit/z3.6"
front = load_front(f"{P}/pareto_ns.csv", f"{P}/grad_faith_ns.csv")   # PySR front + grad-faith sidecar
best = front.dropna(subset=["grad_err"]).sort_values("Loss").iloc[0] # value-optimal faithful eq
print(best["Loss"], best["grad_err"], best["value_mse"])
```
Sidecar columns: `Complexity, Loss, grad_err, value_mse, n_keep, gate_pass, x0_enters`.
`grad_err` = median-over-k slope error vs the GP in **log-P_F** (`median_k |∂logP_eq/∂θ ÷ ∂logP_GP/∂θ − 1|`,
gate `0.25`); `value_mse` = log-P value error. See the walkthrough for the definitions.

---

## Repository layout
```
src/priya_forecast/   Installed package: GP + PySR models, the derivative gate,
                      Sobolev loss, grad-faith sidecar I/O, Pareto diagnostic,
                      single_z/ + multi_z/ pipelines, the `priya-forecast` CLI.
scripts/              Pipeline drivers: make_diagnostic_figs.py, eval_grad_faithfulness.py,
                      regen_*.py, submit_paper_production.sh, refit/aggregate entry points.
notebooks/            reproduce_paper.ipynb — the single all-figures-and-tables notebook.
tests/                pytest suite (-k "not slow"); emulator-touching tests skip on a bare clone.
docs/                 User docs: the diagnostic walkthrough, code-review map, onboarding,
                      method/perf notes.  docs/dev/ holds internal design/handoff history.
configs/              YAML run configs.  slurm/  cluster job scripts.
data/                 (gitignored) GP basedir + inputs — staged via scripts/prep_kodiaq_gp.py.
results/              The committed production run (paper_production_…) the figures replay.
```

## Citation
If you use this code, please cite the paper (companion LaTeX repo:
[github.com/jibanCat/Knowledge-Distillation-using-PySR-with-PRIYA-suite](https://github.com/jibanCat/Knowledge-Distillation-using-PySR-with-PRIYA-suite)):

```bibtex
@article{Ho_PySR_PRIYA,
  author  = {Ho, Ming-Feng and Avestruz, Camille},
  title   = {Knowledge Distillation with PySR on the PRIYA suite},
  year    = {2026},
  note    = {arXiv: TBD}
}
```
See [`CITATION.cff`](CITATION.cff) for the machine-readable citation.

## License
MIT — see [`LICENSE`](LICENSE).
