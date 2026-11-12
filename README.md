# priya-forecast — symbolic distillation of the PRIYA Lyα P1D emulator

Companion code for the paper. It distils the **PRIYA** multi-fidelity Gaussian-process
emulator of the Lyman-α 1D flux power spectrum into one symbolic equation per parameter
(with **PySR**), and tests whether each equation reproduces the emulator's parameter
derivatives, not only its values.

> **Student research project — original credit belongs to Aidan Behmer.** The modeling idea
> and the original implementation are Aidan's: see
> **[`aidanbehmer/priya_pysr`](https://github.com/aidanbehmer/priya_pysr)**. This repository
> re-runs that approach with more compute, refactored by **Ming-Feng Ho** for reproducibility.
> The companion paper is **in preparation**; see [Citation](#citation).

## Install

Python 3.11 or 3.12. This is all you need to reproduce the paper's figures and tables:

```bash
git clone https://github.com/jibanCat/lya1d_priya_forecast && cd lya1d_priya_forecast
python3.11 -m venv .venv-figures && source .venv-figures/bin/activate
pip install -r requirements-figures.txt
```

The figures also need LaTeX (`latex` on `PATH`) for their labels. Keep the full `git clone`:
the plotting script checks each input file against tag `v1.2-paper`.

## Reproduce the paper's figures and tables

No emulator and no PySR: everything is drawn from files committed in this repository.

```bash
python scripts/make_figures.py --out out_figs        # the seven figures, about 20 s

PYTHONPATH=src python - <<'EOF'                      # Tables 3, 4 and 5
import pandas as pd
from priya_forecast import paper_figures as pf
run = pf.load_run()
print(pf.taxonomy(run).round(3))                     # Table 3: derivative error at the Pareto knee
print(pf.equations(run, loss="sobolev").round(4))    # Table 4: the knee equations
print(pd.read_csv("results/paper_production_20260630_perz_sobolev_z2.6-4.2/"
                  "figures/multid_z3.6/multid_bestworst.csv", comment="#"))   # Table 5
EOF
```

Table 1 is `src/priya_forecast/parameters.py`; Table 2 is
`results/paper_production_20260630_perz_sobolev_z2.6-4.2/figures/table2_stats.tex`, which the
notebook below recomputes from the saved fits.
[`scripts/README.md`](scripts/README.md) shows the plotting function behind each figure.

## If you really need to rerun the fits

You do not need this to reproduce the paper. A rerun needs the GP emulator and PySR, and
because the search is not deterministic it reproduces each equation's faithful or
unfaithful class, not its digits.

```bash
# 1. the full environment (Julia is installed by PySR on first use)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt && pip install -e ".[gp,forecast,pysr]"

# 2. the emulator: two public repositories
git clone https://github.com/jibanCat/kodiaq-squad-gp data/kodiaq_gp          # the paper's trained emulator
git clone https://github.com/jibanCat/InferenceLyaData ../InferenceLyaData    # the lyaemu package
export LYA_EMULATOR=$PWD/../InferenceLyaData
export PYTHONPATH=src:$LYA_EMULATOR

# 3. one fit at the paper's budget (n_P, Sobolev loss; about 15 min on four cores)
python scripts/regen_1pvar.py --params ns            # the sweep file the scorer reads
python scripts/refit_one_param_single_z.py --param ns --z 3.6 --output-dir out/sobolev \
    --target-space log --maxsize 20 --niterations 200 --populations 48 --seed 0 \
    --use-sobolev --sobolev-lambda 5
python scripts/eval_grad_faithfulness.py --pareto out/sobolev/refit/z3.6/pareto_ns.csv \
    --param ns --z 3.6 --out out/sobolev/refit/z3.6/grad_faith_ns.csv

# or the whole grid in one command (66 fits, about 10 h on four cores)
python -m priya_forecast.rerun --preset full --label full_seed0
```

[`REPRODUCE.md`](REPRODUCE.md) has the details: every command, the SLURM scripts, and a
table of the selection rules and settings with the file that defines each.

## Tutorials

Two notebooks, cell by cell ([`notebooks/README.md`](notebooks/README.md)):

- [`notebooks/reproduce_paper.ipynb`](notebooks/reproduce_paper.ipynb) redraws every figure
  and prints every table of the paper, and says which selection rule each one uses. No emulator.
- [`notebooks/rerun_paper.ipynb`](notebooks/rerun_paper.ipynb) refits the equations yourself
  and compares your run with the paper's. Needs the emulator and PySR.

## Citation

This is a **student-led research project**. The original modeling idea and code are Aidan
Behmer's ([`aidanbehmer/priya_pysr`](https://github.com/aidanbehmer/priya_pysr)). The
companion paper is **in preparation** and its title and full author list are not yet
decided; please contact the authors before citing. Provisional entry:

```bibtex
@article{Behmer_PRIYA_PySR,
  author  = {Behmer, Aidan and Ho, Ming-Feng and Avestruz, Camille},
  title   = {Symbolic distillation of the PRIYA Lyman-$\alpha$ P1D emulator (working title, not final)},
  year    = {2026},
  note    = {Student project, in preparation; title and authors provisional; arXiv: TBD}
}
```

See [`CITATION.cff`](CITATION.cff) for the machine-readable citation. The emulator itself is
from Ho et al. (2026), JCAP 07 (2026) 094, arXiv:2509.18271; cite it when you use it.

## License

MIT — see [`LICENSE`](LICENSE).
