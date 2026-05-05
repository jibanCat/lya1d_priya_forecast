#!/bin/bash
# GreatLakes SLURM single-job template — multi-D PySR cross-coupled refit.
#
# One PySR fit over the cross-coupled subset
# {ns, Ap, herei, heref, alphaq, hireionz} on a fat-cores compute node.
# Speed-up over the login-node multithread-procs=4 run is ~3-4× from
# `procs=16` on a 16-core node.
#
# Usage:
#   sbatch --export=ALL,REPO=/home/mfho/lya1d_priya_forecast,\
#                    OUTPUT_DIR=results/refit_multid_z2.6-4.2,\
#                    Z_MIN=2.6,Z_MAX=4.2,N_TOTAL=256,NITER=100,MAXSIZE=25 \
#          slurm/refit_multid.slurm
#
# After completion:
#   ls $OUTPUT_DIR/   # scorecard.md, multid_equation.md, fisher.npz, corner.{pdf,png}

#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=multid_pysr
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO=${REPO:?REPO must be set via --export}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR must be set via --export}
Z_MIN=${Z_MIN:-2.6}
Z_MAX=${Z_MAX:-4.2}
N_TOTAL=${N_TOTAL:-256}
NITER=${NITER:-100}
MAXSIZE=${MAXSIZE:-25}

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-interactive}"
echo "[slurm] cpus_per_task=${SLURM_CPUS_PER_TASK:-?}"
echo "[slurm] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[slurm] z range=[${Z_MIN}, ${Z_MAX}], n_total=${N_TOTAL}, niter=${NITER}"

# Use the Greatlakes shared mamba py3.11 base where pysr is installed
# via `pip install --user` (visible from any compute node since
# ~/.local/lib/... is on the home filesystem). Avoid `conda activate
# emu-3.9` which has no pysr.
module purge
module load gcc/10.3.0 || true
PY=/sw/pkgs/arc/mamba/py3.11/bin/python

# PySR / Julia env.
export PYTHON_JULIAPKG_PROJECT="${HOME}/.julia_env"
export JULIA_DEPOT_PATH="${HOME}/.julia"
export PYTHONPATH="/home/mfho/student_projects/lya_emulator_full:${REPO}/src"

cd "$REPO"
PROCS=$((${SLURM_CPUS_PER_TASK:-4} - 1))
[ "$PROCS" -lt 1 ] && PROCS=1

"$PY" scripts/refit_multid_subset.py \
    --z-min "$Z_MIN" --z-max "$Z_MAX" \
    --n-total "$N_TOTAL" --niter "$NITER" --maxsize "$MAXSIZE" \
    --procs "$PROCS" \
    --output "$OUTPUT_DIR"

echo "[slurm] multi-D refit done"
