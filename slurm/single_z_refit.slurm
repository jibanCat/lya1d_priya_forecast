#!/bin/bash
# GreatLakes SLURM array — single-z PySR refit, 11 params for one z-bin.
#
# Submit one array job per z-bin (the batch driver does this 13×):
#   sbatch --export=ALL,REPO=/home/mfho/lya1d_priya_forecast,\
#                    BASEDIR=data/kodiaq_gp,\
#                    OUTPUT_DIR=results/single_z_run,Z=3.6 \
#          --array=0-10 \
#          slurm/single_z_refit.slurm

#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=sz_refit
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO=${REPO:?REPO must be set via --export}
BASEDIR=${BASEDIR:?BASEDIR must be set via --export}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR must be set via --export}
Z=${Z:?Z must be set via --export}
TARGET_SPACE=${TARGET_SPACE:-linear}

PARAMS=(dtau0 tau0 ns Ap herei heref alphaq hub omegamh2 hireionz bhfeedback)
PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]:?Array index out of range}

echo "[slurm] task=${SLURM_ARRAY_TASK_ID} param=${PARAM} z=${Z}"

module purge
module load gcc/10.3.0 || true
# Project venv with the pinned numpy<2 stack (see README). Build once at the
# repo root before submitting (see slurm/multi_z_refit.slurm header / README).
PY="${REPO}/.venv/bin/python"
[ -x "$PY" ] || { echo "ERROR: project venv missing at $PY — see README install."; exit 1; }

export PYTHON_JULIAPKG_PROJECT="${HOME}/.julia_env"
export JULIA_DEPOT_PATH="${HOME}/.julia"
export PYTHONPATH="/home/mfho/student_projects/lya_emulator_full:${REPO}/src"

cd "$REPO"
"$PY" scripts/refit_one_param_single_z.py \
    --param "$PARAM" --z "$Z" \
    --basedir "$BASEDIR" --output-dir "$OUTPUT_DIR" \
    --target-space "$TARGET_SPACE"

echo "[slurm] ${PARAM} z=${Z} done"
