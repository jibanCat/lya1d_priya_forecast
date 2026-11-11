#!/usr/bin/env bash
# Submit the §5.1 joint-fit MAXSIZE SWEEP to GreatLakes, as the committed results ran.
#
# 20 MSE runs = maxsize {25,40,60,100} x seeds {0..4}, encoded in a 0-19 job
# array (ms_idx = id/5, seed = id%5). Each dumps joint_rank_diagnostic.json with
# the discriminator fields (front_max_rank, pinned_at_cap, n_inputs_present,
# in-sample/off-fid MSE) to results/joint_multid_sweep/ms<MS>_seed<SEED>/.
#
# Two submissions, because the iteration count is not uniform: maxsize 25/40/60
# (tasks 0-14) run 400 iterations; maxsize 100 (tasks 15-19) runs 250, since 400
# overran the wall time there. Emulator: data/kodiaq_gp (the paper's, nk=172).
#
# Usage:
#   scripts/submit_joint_sweep.sh              # real submit
#   scripts/submit_joint_sweep.sh --dry-run    # print sbatch, no queue
#   scripts/submit_joint_sweep.sh --test-only  # sbatch --test-only (validate, no queue)
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
SLURM="slurm/joint_multid_sweep_kodiaq.slurm"
OUTPUT_ROOT="results/joint_multid_sweep"
N_TRAIN="${N_TRAIN:-512}"
BASEDIR="${BASEDIR:-data/kodiaq_gp}"
Z="${Z:-3.6}"
# array tasks, iterations, wall time: one line per submission
LEGS=("0-14 400 6:00:00" "15-19 250 9:00:00")
NUMPY_SHADOW="${NUMPY_SHADOW:-$HOME/.local/np_gpy_shadow}"

# PARAMS is NOT exported (spaces break sbatch --export comma-parsing); the
# 6-param default lives in the .slurm file. Keep the two in sync.
echo "REPO         = $REPO"
echo "account      = $ACCOUNT"
echo "array        = id/5 -> maxsize {25,40,60,100}; id%5 -> seed 0..4"
echo "params       = ns Ap herei heref alphaq hireionz  (default in .slurm)"
echo "emulator     = $BASEDIR"
echo "budget       = n_train=$N_TRAIN z=$Z   (maxsize swept; niter per submission below)"
echo "output       = $OUTPUT_ROOT/ms<MS>_seed<SEED>/  (rank JSON: joint_rank_diagnostic.json)"
echo "numpy shadow = $NUMPY_SHADOW"
echo

for leg in "${LEGS[@]}"; do
  read -r ARRAY NITER WALLT <<<"$leg"
  EXPORTS="ALL,REPO=$REPO,OUTPUT_ROOT=$OUTPUT_ROOT,N_TRAIN=$N_TRAIN,NITER=$NITER,Z=$Z,BASEDIR=$BASEDIR,NUMPY_SHADOW=$NUMPY_SHADOW"
  # sbatch OPTIONS only — any flag (e.g. --test-only) MUST precede the script path,
  # else sbatch passes it to the batch script and submits for real.
  SBATCH_OPTS=(--account="$ACCOUNT" --time="$WALLT" --array="$ARRAY" --export="$EXPORTS")
  printf 'sbatch cmd   : sbatch %s %s\n' "${SBATCH_OPTS[*]}" "$SLURM"
  case "$MODE" in
    dry)  echo "[--dry-run] not submitted." ;;
    test) echo "[--test-only] validating with the scheduler (no queue)..."
          sbatch "${SBATCH_OPTS[@]}" --test-only "$SLURM" ;;
    submit) mkdir -p "$OUTPUT_ROOT"; sbatch "${SBATCH_OPTS[@]}" "$SLURM" ;;
  esac
done
[ "$MODE" = submit ] && echo "Watch: squeue -u $USER -A $ACCOUNT" || true
