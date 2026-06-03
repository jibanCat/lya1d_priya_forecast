#!/bin/bash
# GreatLakes SLURM array — multi-z PySR refit, 11 params over [Z_MIN, Z_MAX].
#   sbatch --export=ALL,REPO=$(pwd),BASEDIR=data/kodiaq_gp,\
#          OUTPUT_DIR=results/multi_z_stage7,Z_MIN=2.6,Z_MAX=4.2 \
#          --array=0-10%3 slurm/multi_z_refit.slurm   # %3 caps concurrent Julia inits
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1:30:00
#SBATCH --job-name=mz_refit
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
export PYTHONUNBUFFERED=1
REPO=${REPO:?REPO must be set via --export}
BASEDIR=${BASEDIR:?BASEDIR must be set via --export}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR must be set via --export}
Z_MIN=${Z_MIN:?Z_MIN must be set via --export}
Z_MAX=${Z_MAX:?Z_MAX must be set via --export}
PARAMS=(dtau0 tau0 ns Ap herei heref alphaq hub omegamh2 hireionz bhfeedback)
PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]:?Array index out of range}
echo "[slurm] task=${SLURM_ARRAY_TASK_ID} param=${PARAM} z in [${Z_MIN},${Z_MAX}]"
module purge
module load gcc/10.3.0 || true
# Project venv with the pinned numpy<2 stack (see README). Build once at the
# repo root before submitting:
#   /sw/pkgs/arc/mamba/py3.11/bin/python -m venv .venv
#   .venv/bin/pip install -r requirements.lock.txt && .venv/bin/pip install -e . --no-deps
PY="${REPO}/.venv/bin/python"
[ -x "$PY" ] || { echo "ERROR: project venv missing at $PY — see README install."; exit 1; }
export PYTHON_JULIAPKG_PROJECT="${HOME}/.julia_env"
export JULIA_DEPOT_PATH="${HOME}/.julia"
export PYTHONPATH="/home/mfho/student_projects/lya_emulator_full:${REPO}/src"
# Stagger Julia init: juliacall/juliapkg take an flock on the shared
# ~/.julia_env project + ~/.julia depot at startup, and concurrent array tasks
# can hit NFS lock failures (ENOLCK "No locks available" / ESTALE "Stale file
# handle"). Desynchronise starts; ALSO submit with a concurrency cap, e.g.
#   sbatch ... --array=0-10%3 slurm/multi_z_refit.slurm
sleep $(( (SLURM_ARRAY_TASK_ID % 12) * 12 ))
cd "$REPO"
"$PY" scripts/refit_one_param_multi_z.py --param "$PARAM" \
    --z-min "$Z_MIN" --z-max "$Z_MAX" \
    --basedir "$BASEDIR" --output-dir "$OUTPUT_DIR"
echo "[slurm] ${PARAM} z in [${Z_MIN},${Z_MAX}] done"
