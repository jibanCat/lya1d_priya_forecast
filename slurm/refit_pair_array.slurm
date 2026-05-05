#!/bin/bash
# SLURM array for per-pair PySR fits (Phase 2).
#
# Usage:
#   sbatch --export=ALL,REPO=...,PAYLOAD_DIR=...,OUTPUT_DIR=...,PAIRS=tau0,ns:herei,alphaq:Ap,alphaq:tau0,Ap \
#          --array=0-3 slurm/refit_pair_array.slurm

#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=pysr_pair
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1

cd "$REPO"
export PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env
export JULIA_DEPOT_PATH=$HOME/.julia
export PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src

# PAIRS is a colon-separated list of "name_i,name_j" entries.
IFS=':' read -ra PAIR_ARRAY <<< "$PAIRS"
PAIR="${PAIR_ARRAY[$SLURM_ARRAY_TASK_ID]}"
echo "Task $SLURM_ARRAY_TASK_ID → pair $PAIR"

/sw/pkgs/arc/mamba/py3.11/bin/python scripts/refit_one_pair.py \
    --pair "$PAIR" \
    --payload-dir "$PAYLOAD_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --max-retries 2
