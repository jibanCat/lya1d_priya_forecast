#!/bin/bash
# GreatLakes SLURM — §5.1 "joint multi-D PySR fit" VALIDATION run.
#
# Trains ONE PySR equation jointly over the cross-coupled subset
#   {ns, Ap, herei, heref, alphaq, hireionz}   (paper: n_S, A_P, z_Hei, z_Hef, alpha_q, z_Hi)
# via scripts/run_multid_pysr.py (combine='joint'), builds the PySRModel,
# computes the GP / perfect-1D / joint-PySR Fisher matrices, and DUMPS the
# rank diagnostic (joint_rank_diagnostic.json): eigen-spectrum, condition
# number, and numerical rank of the joint-PySR Fisher/Jacobian vs the GP.
# This backs the paper claim that the joint Jacobian dP_F/dtheta is numerically
# rank-deficient (parameters fold into one shared sub-expression).
#
# Submitted as a SEED-BAND job array (one seed per array task) for robustness.
# Each task is an independent single-z fit; PySR here is serial+deterministic
# (run_multid_pysr.py hardcodes parallelism='serial'), so cpus mainly serve
# BLAS/GP threads, not the PySR search.
#
# Usage (sbatch wrapper does this for you — see scripts/submit_joint_validation.sh):
#   sbatch --account=cavestru0 --array=0-4 \
#          --export=ALL,REPO=/home/mfho/lya1d_priya_forecast,\
#OUTPUT_ROOT=results/joint_multid_validation,PARAMS="ns Ap herei heref alphaq hireionz",\
#N_TRAIN=256,NITER=200,MAXSIZE=25,Z=3.6 \
#          slurm/joint_multid_validation.slurm
#
# After completion, per seed:
#   results/joint_multid_validation/seed<ID>/joint_rank_diagnostic.json  <-- rank numbers
#   results/joint_multid_validation/seed<ID>/hall_of_fame.csv            <-- Pareto front
#   results/joint_multid_validation/seed<ID>/scorecard.md, summary.md, *.png

#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --job-name=joint_multid_val
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO=${REPO:?REPO must be set via --export}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/joint_multid_validation}
PARAMS=${PARAMS:-ns Ap herei heref alphaq hireionz}
N_TRAIN=${N_TRAIN:-256}
NITER=${NITER:-200}
MAXSIZE=${MAXSIZE:-25}
Z=${Z:-3.6}
# Seed = array task id (seed band); falls back to SEED for a non-array run.
SEED=${SLURM_ARRAY_TASK_ID:-${SEED:-0}}
OUTPUT_DIR="${OUTPUT_ROOT}/seed${SEED}"

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-interactive} array_task=${SLURM_ARRAY_TASK_ID:-none}"
echo "[slurm] params=[${PARAMS}] seed=${SEED} z=${Z} n_train=${N_TRAIN} niter=${NITER} maxsize=${MAXSIZE}"
echo "[slurm] OUTPUT_DIR=${OUTPUT_DIR}"

# Greatlakes shared mamba py3.11 base (pysr installed via pip install --user).
module purge
module load gcc/10.3.0 || true
PY=${PY:-/sw/pkgs/arc/mamba/py3.11/bin/python}

# --- PySR / Julia env ---
export PYTHON_JULIAPKG_PROJECT="${HOME}/.julia_env"
export JULIA_DEPOT_PATH="${HOME}/.julia"

# --- numpy<2 shadow (REQUIRED) ---
# lyaemu's GP backend is GPy 1.13.2, whose compiled extension needs the numpy
# 1.x ABI; the base mamba numpy is 2.0.1. Prepend a numpy<2 shadow dir so GPy
# imports. Provisioned once on a login node with:
#   python -m pip install --target "$HOME/.local/np_gpy_shadow" --no-deps "numpy==1.26.4"
# (Alternatively the whole env can be pinned with `pip install --user "numpy<2"`,
#  in which case NUMPY_SHADOW can point at an empty/absent dir — the assert below
#  is the real guard.)
NUMPY_SHADOW="${NUMPY_SHADOW:-${HOME}/.local/np_gpy_shadow}"
export PYTHONPATH="${NUMPY_SHADOW}:/home/mfho/student_projects/lya_emulator_full:${REPO}/src"

cd "$REPO"

# Fail fast if numpy>=2 would be picked up (GPy import would crash mid-run).
"$PY" - <<'PYCHECK'
import numpy, sys
maj = int(numpy.__version__.split(".")[0])
print(f"[env] numpy {numpy.__version__} @ {numpy.__file__}")
if maj >= 2:
    sys.exit("[env][FATAL] numpy>=2 active; GPy needs numpy<2. "
             "Provision the shadow dir or `pip install --user 'numpy<2'`.")
import scipy, pandas, GPy
print(f"[env] scipy {scipy.__version__}, pandas {pandas.__version__}, GPy {GPy.__version__}")
PYCHECK

echo "[slurm] starting joint multi-D PySR fit..."
"$PY" scripts/run_multid_pysr.py \
    --params ${PARAMS} \
    --n-train "$N_TRAIN" \
    --niter "$NITER" --maxsize "$MAXSIZE" \
    --seed "$SEED" --z "$Z" \
    --output "$OUTPUT_DIR"

echo "[slurm] done. rank diagnostic -> ${OUTPUT_DIR}/joint_rank_diagnostic.json"
