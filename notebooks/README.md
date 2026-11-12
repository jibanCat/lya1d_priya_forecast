# Notebooks

Two notebooks, for two different jobs.

| notebook | what it is for | needs | time |
|---|---|---|---|
| [`reproduce_paper.ipynb`](reproduce_paper.ipynb) | **Replot.** Redraws every figure and prints every table of the paper from the committed results, and says which selection rule each one uses. Nothing is refitted. | `requirements-figures.txt`, a full `git clone`, LaTeX | about a minute |
| [`rerun_paper.ipynb`](rerun_paper.ipynb) | **Rerun.** Refits the per-parameter equations yourself with PySR against the GP emulator, scores them, and compares your run with the paper's. For testing a different budget, seed or loss weight. | the GP emulator, PySR and Julia (see `REPRODUCE.md`, Step 3a) | minutes (quick setting) to cluster hours (full) |

Start with `reproduce_paper.ipynb`. Its figures are drawn by `scripts/make_figures.py`,
the script that builds the manuscript's figures, so they match the paper exactly.

A run from `rerun_paper.ipynb` is yours, on your budget and seed. It writes to
`results/tutorial_reruns/` and never changes the committed results. Its default setting
is much smaller than the paper's (`niterations = 30`, `populations = 8`, against 200 and
48), so treat its output as illustrative.

Selection rules and settings are listed in `REPRODUCE.md` ("Selection rules and settings").
