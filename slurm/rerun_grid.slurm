#!/bin/bash
# GreatLakes SLURM — run the rerun grid (priya_forecast.rerun.run_grid) as one job.
#
# This is the cluster form of notebooks/rerun_paper.ipynb. Everything after the script
# name is passed to `python -m priya_forecast.rerun`; `--help` lists the options.
#
# The paper's grid at the paper's search budget (11 parameters x z = 2.6, 3.6, 4.2 x
# both losses = 66 fits; niterations=200, populations=48, maxsize=20, lambda=5):
#   sbatch --export=ALL,REPO=$PWD slurm/rerun_grid.slurm --preset full --label full_seed0
# (LYA_EMULATOR must be set in the submitting shell; --export=ALL passes it on.)
# Another seed of the seed band, one redshift only:
#   sbatch --export=ALL,REPO=$PWD slurm/rerun_grid.slurm --preset full --zs 3.6 --seed 1 --label z36_seed1
# Print the settings without running anything:
#   REPO=$PWD bash slurm/rerun_grid.slurm --preset full --label full_seed0 --dry-run
#
# Output: results/tutorial_reruns/rerun_<label>/ (git-ignored), the same layout the
# notebook's Sections 7 and 8 read, plus compare_to_production.csv. Each label holds one
# run; a second run with other settings under the same label is refused.
#
# Wall time: the paper's fits took a median of 13 min (Sobolev) and 4 min (value-only) on
# four cores, so the 66 fits of `--preset full` need about 10 h. This script was checked
# with a small run, not with a full one.

#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=16:00:00
#SBATCH --job-name=rerun_grid
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO=${REPO:?REPO must be set (sbatch --export=ALL,REPO=\$PWD ...)}
module purge 2>/dev/null || true
module load gcc/10.3.0 2>/dev/null || true

# Project venv with the pinned numpy<2 stack (see README).
PY="${REPO}/.venv/bin/python"
[ -x "$PY" ] || { echo "ERROR: project venv missing at $PY — see README install."; exit 1; }
export PYTHON_JULIAPKG_PROJECT="${PYTHON_JULIAPKG_PROJECT:-${HOME}/.julia_env}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${HOME}/.julia}"
# LYA_EMULATOR: the directory that contains the lyaemu/ package (REPRODUCE.md, Step 2b).
LYA_EMULATOR=${LYA_EMULATOR:?LYA_EMULATOR must be set to the directory that contains lyaemu/}
export PYTHONPATH="${LYA_EMULATOR}:${REPO}/src"

cd "$REPO"
"$PY" -m priya_forecast.rerun "$@"
