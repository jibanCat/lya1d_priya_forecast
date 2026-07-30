#!/usr/bin/env bash
# Submit the §5.1 "joint multi-D PySR fit" VALIDATION seed band to GreatLakes.
#
# One PySR equation trained jointly over {ns, Ap, herei, heref, alphaq, hireionz},
# per seed, each dumping joint_rank_diagnostic.json (Fisher/Jacobian eigen-
# spectrum + condition number + numerical rank) to back the paper's
# rank-deficiency claim.  Runner: scripts/run_multid_pysr.py (combine='joint').
#
# Usage:
#   scripts/submit_joint_validation.sh              # real submit
#   scripts/submit_joint_validation.sh --dry-run    # print the sbatch, no queue
#   scripts/submit_joint_validation.sh --test-only  # sbatch --test-only (validate, no queue)
set -euo pipefail

MODE="submit"
case "${1:-}" in
  --dry-run)   MODE="dry" ;;
  --test-only) MODE="test" ;;
  "" ) ;;
  * ) echo "unknown arg: $1" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

ACCOUNT="${SLURM_ACCOUNT:-cavestru0}"     # yueyingn0 expired 2026-07-01
SLURM="slurm/joint_multid_validation.slurm"
OUTPUT_ROOT="results/joint_multid_validation"
PARAMS="ns Ap herei heref alphaq hireionz"
N_TRAIN="${N_TRAIN:-256}"
NITER="${NITER:-200}"
MAXSIZE="${MAXSIZE:-25}"
Z="${Z:-3.6}"
ARRAY="${ARRAY:-0-4}"                      # 5-seed band (seeds 0..4)
WALLT="${WALLT:-2:00:00}"
NUMPY_SHADOW="${NUMPY_SHADOW:-$HOME/.local/np_gpy_shadow}"

# NOTE: PARAMS is NOT exported here — spaces in an sbatch --export value are
# fragile. The 6-param subset is the hardcoded default in the .slurm file
# (PARAMS=${PARAMS:-ns Ap herei heref alphaq hireionz}); keep the two in sync.
EXPORTS="ALL,REPO=$REPO,OUTPUT_ROOT=$OUTPUT_ROOT,N_TRAIN=$N_TRAIN,NITER=$NITER,MAXSIZE=$MAXSIZE,Z=$Z,NUMPY_SHADOW=$NUMPY_SHADOW"

# sbatch OPTIONS only (no script path). Any sbatch flag (e.g. --test-only)
# MUST precede the script path — flags placed AFTER the script are passed as
# arguments to the batch script and are silently ignored by sbatch (so
# `sbatch ... script.slurm --test-only` submits for real!).
SBATCH_OPTS=(--account="$ACCOUNT" --time="$WALLT" --array="$ARRAY" --export="$EXPORTS")

echo "REPO         = $REPO"
echo "account      = $ACCOUNT"
echo "array (seeds)= $ARRAY"
echo "params       = $PARAMS"
echo "budget       = n_train=$N_TRAIN niter=$NITER maxsize=$MAXSIZE z=$Z"
echo "walltime     = $WALLT   (cpus=4, mem=24G per task; see $SLURM)"
echo "output       = $OUTPUT_ROOT/seed<ID>/  (rank JSON: joint_rank_diagnostic.json)"
echo "numpy shadow = $NUMPY_SHADOW"
echo
printf 'sbatch cmd   : sbatch %s %s\n' "${SBATCH_OPTS[*]}" "$SLURM"

case "$MODE" in
  dry)  echo "[--dry-run] not submitted." ;;
  test) echo "[--test-only] validating with the scheduler (no queue)..."
        sbatch "${SBATCH_OPTS[@]}" --test-only "$SLURM" ;;
  submit) mkdir -p "$OUTPUT_ROOT"; sbatch "${SBATCH_OPTS[@]}" "$SLURM"
        echo "Submitted. Watch: squeue -u $USER -A $ACCOUNT" ;;
esac
