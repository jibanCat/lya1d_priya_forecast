#!/bin/bash
# GreatLakes SLURM array — single-z PySR refit, 11 params for one z-bin.
#
# Submit one array job per z-bin. Env knobs (all via --export):
#   Z TARGET_SPACE USE_SOBOLEV SOBOLEV_LAMBDA ANOVA_LOSS SAVE_ARTIFACTS SEED
#   MAXSIZE POPULATIONS NITERATIONS  (budget defaults: 24 / 50).
# NOTE: USE_SOBOLEV=1 requires TARGET_SPACE=log (the CLI enforces it).
# LYA_EMULATOR overrides the upstream emulator path (default = author's home).
#
# Production Sobolev fit (one z), higher budget:
#   sbatch --export=ALL,REPO=/home/mfho/lya1d_priya_forecast,\
#                    BASEDIR=data/kodiaq_gp,\
#                    OUTPUT_DIR=results/paper_production_20260630_perz_sobolev_z2.6-4.2/sobolev,\
#                    Z=3.6,TARGET_SPACE=log,USE_SOBOLEV=1,SOBOLEV_LAMBDA=5,\
#                    MAXSIZE=20,POPULATIONS=48,NITERATIONS=200,SAVE_ARTIFACTS=1,SEED=0 \
#          --array=0-10 slurm/single_z_refit.slurm
# Value (no-gradient, plain-MSE) baseline: drop USE_SOBOLEV/SOBOLEV_LAMBDA,
#   keep TARGET_SPACE=log, point OUTPUT_DIR at .../value.

#SBATCH --account=yueyingn0
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
USE_SOBOLEV=${USE_SOBOLEV:-0}
SOBOLEV_LAMBDA=${SOBOLEV_LAMBDA:-1.0}
SEED=${SEED:-0}
MAXSIZE=${MAXSIZE:-20}
POPULATIONS=${POPULATIONS:-24}
NITERATIONS=${NITERATIONS:-50}
ANOVA_LOSS=${ANOVA_LOSS:-0}
SAVE_ARTIFACTS=${SAVE_ARTIFACTS:-0}
SOBOLEV_ARGS=""
[ "$USE_SOBOLEV" = "1" ] && SOBOLEV_ARGS="--use-sobolev --sobolev-lambda ${SOBOLEV_LAMBDA}"
ANOVA_ARGS=""
[ "$ANOVA_LOSS" = "1" ] && ANOVA_ARGS="--anova-loss"
ARTIFACT_ARGS=""
[ "$SAVE_ARTIFACTS" = "1" ] && ARTIFACT_ARGS="--save-artifacts"

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
export PYTHONPATH="${LYA_EMULATOR:-/home/mfho/student_projects/lya_emulator_full}:${REPO}/src"

cd "$REPO"
"$PY" scripts/refit_one_param_single_z.py \
    --param "$PARAM" --z "$Z" \
    --basedir "$BASEDIR" --output-dir "$OUTPUT_DIR" \
    --target-space "$TARGET_SPACE" --seed "$SEED" --maxsize "$MAXSIZE" \
    --populations "$POPULATIONS" --niterations "$NITERATIONS" \
    $SOBOLEV_ARGS $ANOVA_ARGS $ARTIFACT_ARGS

echo "[slurm] ${PARAM} z=${Z} done"
