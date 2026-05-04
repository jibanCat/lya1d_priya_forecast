#!/bin/bash
# GreatLakes SLURM array template — option C parallel multi-z PySR refits.
#
# Each task in the array runs ONE param's PySR fit, loading a tiny
# precomputed payload (no GP emulator needed). 11 tasks → ~3 min wall.
#
# Usage:
#   # Phase 1: precompute payloads (single-node, ~7 min):
#   PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia \
#   PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \
#       python scripts/precompute_payloads.py \
#           --z-min 2.6 --z-max 4.2 --n-total 225 \
#           --output results/refit_kodiaq_multiz_2.6-4.2/payloads
#
#   # Phase 2: parallel refits via SLURM array:
#   sbatch --export=ALL,REPO=/home/mfho/lya1d_priya_forecast,\
#                    PAYLOAD_DIR=results/refit_kodiaq_multiz_2.6-4.2/payloads,\
#                    OUTPUT_DIR=results/refit_kodiaq_multiz_2.6-4.2 \
#          --array=0-10 \
#          slurm/refit_array.slurm
#
#   # Phase 3: aggregate (after all array tasks complete):
#   PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia \
#   PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \
#       python scripts/multi_z_aggregate.py \
#           --refits-dir results/refit_kodiaq_multiz_2.6-4.2/refits \
#           --output results/refit_kodiaq_multiz_2.6-4.2

#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=pysr_refit
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO=${REPO:?REPO must be set via --export}
PAYLOAD_DIR=${PAYLOAD_DIR:?PAYLOAD_DIR must be set via --export}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR must be set via --export}

# Map SLURM_ARRAY_TASK_ID 0..10 to the 11 PRIYA params (canonical order).
PARAMS=(dtau0 tau0 ns Ap herei heref alphaq hub omegamh2 hireionz bhfeedback)
PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]:?Array index out of range}

echo "[slurm] task=${SLURM_ARRAY_TASK_ID} param=${PARAM}"
echo "[slurm] host=$(hostname)"
echo "[slurm] payload=${PAYLOAD_DIR}/${PARAM}.pkl"

# Conda env (match the user's existing setup; mamba on Greatlakes).
module purge
module load gcc/10.3.0 || true   # not strictly needed for PySR-only
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh
conda activate emu-3.9 || true

# PySR / Julia environment paths.
export PYTHON_JULIAPKG_PROJECT="${HOME}/.julia_env"
export JULIA_DEPOT_PATH="${HOME}/.julia"
export PYTHONPATH="/home/mfho/student_projects/lya_emulator_full:${REPO}/src"

cd "$REPO"
python scripts/refit_one_param.py \
    --param "$PARAM" \
    --payload-dir "$PAYLOAD_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --niter 50 --maxsize 20 --max-retries 4 --seed 42

echo "[slurm] $PARAM done"
