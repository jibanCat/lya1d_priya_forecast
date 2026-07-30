#!/bin/bash
# GreatLakes SLURM — §5.1 joint-fit MAXSIZE SWEEP (pre-registered).
#
# Settles whether the joint-fit rank-deficiency is STRUCTURAL or a search-budget
# artifact by holding params + z + data budget fixed and varying ONLY maxsize.
#
#   subset : {ns, Ap, herei, heref, alphaq, hireionz}   (fixed)
#   z      : 3.6                                          (fixed)
#   budget : n_train=512, niter=400                       (fixed; both > the
#            original 256/200 so data-/iteration-starvation is not a confound)
#   grid   : maxsize in {25, 40, 60, 100} x seeds {0..4}  = 20 MSE runs
#   loss   : MSE only (joint-Sobolev is not well-defined; see SOBOLEV_ARM_NOTE
#            in scripts/run_multid_pysr.py and the sobolev_arm field in the JSON)
#
# 2-D array: task id encodes (maxsize, seed): ms_idx = id/5, seed = id%5.
#   #SBATCH --array=0-19
#
# Each run dumps results/joint_multid_sweep/ms<MS>_seed<SEED>/joint_rank_diagnostic.json
# with the upgraded discriminator fields: front_max_rank, pinned_at_cap,
# n_inputs_present, in-sample/off-fid MSE, plus the flat `preregistered` block.

#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --job-name=joint_ms_sweep
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO=${REPO:?REPO must be set via --export}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/joint_multid_sweep}
PARAMS=${PARAMS:-ns Ap herei heref alphaq hireionz}
N_TRAIN=${N_TRAIN:-512}
NITER=${NITER:-400}
Z=${Z:-3.6}

# --- decode (maxsize, seed) from the array task id ---
MAXSIZES=(25 40 60 100)
SEEDS_PER_MS=5
TID=${SLURM_ARRAY_TASK_ID:?this script must run as a job array (--array=0-19)}
MS_IDX=$(( TID / SEEDS_PER_MS ))
SEED=$(( TID % SEEDS_PER_MS ))
MAXSIZE=${MAXSIZES[$MS_IDX]}
OUTPUT_DIR="${OUTPUT_ROOT}/ms${MAXSIZE}_seed${SEED}"

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-?} array_task=${TID}"
echo "[slurm] maxsize=${MAXSIZE} seed=${SEED} params=[${PARAMS}] z=${Z} n_train=${N_TRAIN} niter=${NITER}"
echo "[slurm] OUTPUT_DIR=${OUTPUT_DIR}"

module purge
module load gcc/10.3.0 || true
PY=${PY:-/sw/pkgs/arc/mamba/py3.11/bin/python}

export PYTHON_JULIAPKG_PROJECT="${HOME}/.julia_env"
export JULIA_DEPOT_PATH="${HOME}/.julia"

# numpy<2 shadow (REQUIRED): GPy 1.13.2 needs the numpy 1.x ABI but the base
# mamba numpy is 2.0.1. Provisioned once on a login node with:
#   python -m pip install --target "$HOME/.local/np_gpy_shadow" --no-deps "numpy==1.26.4"
NUMPY_SHADOW="${NUMPY_SHADOW:-${HOME}/.local/np_gpy_shadow}"
export PYTHONPATH="${NUMPY_SHADOW}:/home/mfho/student_projects/lya_emulator_full:${REPO}/src"

cd "$REPO"

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

echo "[slurm] starting joint multi-D PySR fit (maxsize=${MAXSIZE}, seed=${SEED})..."
"$PY" scripts/run_multid_pysr.py \
    --params ${PARAMS} \
    --n-train "$N_TRAIN" \
    --niter "$NITER" --maxsize "$MAXSIZE" \
    --seed "$SEED" --z "$Z" \
    --output "$OUTPUT_DIR"

echo "[slurm] done -> ${OUTPUT_DIR}/joint_rank_diagnostic.json"
