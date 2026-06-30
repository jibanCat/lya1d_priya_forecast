#!/usr/bin/env bash
# Submit the paper-production per-z Sobolev run (+ value baseline + seed band)
# to GreatLakes SLURM. Self-documenting: writes RUN_MANIFEST.md with the git
# stamp, budget, and every submitted job id, so the paper's %ref comments can
# trace each figure/table back to this run.
#
#   run-id : prod-20260630-perz-sobolev
#   recipe : one PySR model per param per z; Sobolev (lambda=5, log target)
#            vs plain-MSE value baseline (no ANOVA); z in {2.6, 3.6, 4.2}.
#
# Usage:  scripts/submit_paper_production.sh [--dry-run]
set -euo pipefail

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

RUN_ID="prod-20260630-perz-sobolev"
PROD_DIR="results/paper_production_20260630_perz_sobolev_z2.6-4.2"
ACCOUNT="${SLURM_ACCOUNT:-yueyingn0}"   # override with SLURM_ACCOUNT=...
LYA_EMULATOR="${LYA_EMULATOR:-/home/mfho/student_projects/lya_emulator_full}"
ZS=(2.6 3.6 4.2)
SEEDS=(0 1 2 3 4)
LAMBDA=5
MAXSIZE=20
BUDGET_MAXSIZE=35
POPULATIONS=48
NITERATIONS=200
NS_INDEX=2          # PARAMS=(dtau0 tau0 ns ...) -> ns is array index 2
WALLT="2:00:00"
SLURM="slurm/single_z_refit.slurm"

GIT_HASH="$(git rev-parse --short HEAD)"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
GIT_DIRTY=""; git diff --quiet || GIT_DIRTY=" (+uncommitted)"
STAMP="$(date '+%Y-%m-%d %H:%M:%S %Z')"

# Dry mode must be side-effect-free: don't create the dir or clobber the manifest.
if [ "$DRY" = "1" ]; then
  MANIFEST="/dev/null"
else
  mkdir -p "$PROD_DIR"
  MANIFEST="$PROD_DIR/RUN_MANIFEST.md"
fi

# ---- helper: submit one array job, echo the job id -------------------------
submit() {  # $1=name $2=array_spec $3=outdir  rest=ENV assignments
  local name="$1" array="$2" outdir="$3"; shift 3
  local exports="ALL,REPO=$REPO,BASEDIR=data/kodiaq_gp,OUTPUT_DIR=$outdir,$*"
  if [ "$DRY" = "1" ]; then
    echo "DRY sbatch --account=$ACCOUNT --time=$WALLT --job-name=$name --array=$array --export=$exports $SLURM" >&2
    echo "DRYID_$name"
    return
  fi
  sbatch --parsable --account="$ACCOUNT" --time="$WALLT" \
         --job-name="$name" --array="$array" \
         --export="$exports" "$SLURM"
}

# ---- helper: dependent grad-faith sidecar job (afterany) --------------------
sidecar() {  # $1=name $2=depjobid $3=refit_dir $4=z
  local name="$1" dep="$2" dir="$3" z="$4"
  local cmd="cd $REPO && export PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia PYTHONPATH=${LYA_EMULATOR}:$REPO/src && scripts/make_grad_faith_sidecars.sh $dir $z --log-space"
  if [ "$DRY" = "1" ]; then echo "DRY sidecar $name afterany:$dep on $dir" >&2; echo "DRYSC_$name"; return; fi
  sbatch --parsable --account="$ACCOUNT" --time="$WALLT" --mem=8G --cpus-per-task=4 \
         --partition=standard --job-name="$name" --dependency="afterany:$dep" \
         --wrap "$cmd"
}

JOBS=()

echo "# RUN MANIFEST — $RUN_ID" > "$MANIFEST"
{
  echo
  echo "- **git:** \`$GIT_HASH\` @ \`$GIT_BRANCH\`$GIT_DIRTY"
  echo "- **submitted:** $STAMP"
  echo "- **account:** $ACCOUNT (expires 2026-07-01)"
  echo "- **recipe:** one PySR model per (param, z); Sobolev lambda=$LAMBDA, log target, no ANOVA."
  echo "- **budget:** maxsize=$MAXSIZE, populations=$POPULATIONS, niterations=$NITERATIONS (value baseline = plain MSE, same operators/budget)."
  echo "- **grid:** z = ${ZS[*]}; seed band seeds = ${SEEDS[*]} at z=3.6; ns budget control maxsize=$BUDGET_MAXSIZE."
  echo "- **layout:** \`sobolev/refit/z<z>/\`, \`value/refit/z<z>/\`, \`seed_band/z3.6_seed<S>_{value,sobolev,budget}/refit/z3.6/\`."
  echo
  echo "## Submitted jobs"
  echo "| job | id | array | output |"
  echo "|---|---|---|---|"
} >> "$MANIFEST"

log_job() { JOBS+=("$2"); echo "| $1 | $2 | $3 | $4 |" >> "$MANIFEST"; }

COMMON="TARGET_SPACE=log,MAXSIZE=$MAXSIZE,POPULATIONS=$POPULATIONS,NITERATIONS=$NITERATIONS"

# ===== main per-z fits: Sobolev + value baseline ===========================
for z in "${ZS[@]}"; do
  jid=$(submit "sob_z${z}" 0-10 "$PROD_DIR/sobolev" \
        "Z=$z,USE_SOBOLEV=1,SOBOLEV_LAMBDA=$LAMBDA,SEED=0,SAVE_ARTIFACTS=1,$COMMON")
  log_job "sobolev z=$z" "$jid" "0-10" "$PROD_DIR/sobolev/refit/z$z"
  sc=$(sidecar "sc_sob_z${z}" "$jid" "$PROD_DIR/sobolev/refit/z$z" "$z")
  log_job "  sidecar sobolev z=$z" "$sc" "afterany:$jid" "grad_faith"

  jid=$(submit "val_z${z}" 0-10 "$PROD_DIR/value" \
        "Z=$z,USE_SOBOLEV=0,SEED=0,$COMMON")
  log_job "value z=$z" "$jid" "0-10" "$PROD_DIR/value/refit/z$z"
  sc=$(sidecar "sc_val_z${z}" "$jid" "$PROD_DIR/value/refit/z$z" "$z")
  log_job "  sidecar value z=$z" "$sc" "afterany:$jid" "grad_faith"
done

# ===== seed band at z=3.6: value + sobolev + ns budget control =============
for S in "${SEEDS[@]}"; do
  jid=$(submit "sb_sob_s${S}" 0-10 "$PROD_DIR/seed_band/z3.6_seed${S}_sobolev" \
        "Z=3.6,USE_SOBOLEV=1,SOBOLEV_LAMBDA=$LAMBDA,SEED=$S,$COMMON")
  log_job "seedband sobolev seed=$S" "$jid" "0-10" "seed_band/z3.6_seed${S}_sobolev"

  jid=$(submit "sb_val_s${S}" 0-10 "$PROD_DIR/seed_band/z3.6_seed${S}_value" \
        "Z=3.6,USE_SOBOLEV=0,SEED=$S,$COMMON")
  log_job "seedband value seed=$S" "$jid" "0-10" "seed_band/z3.6_seed${S}_value"

  jid=$(submit "sb_bud_s${S}" "$NS_INDEX" "$PROD_DIR/seed_band/z3.6_seed${S}_budget" \
        "Z=3.6,USE_SOBOLEV=0,SEED=$S,TARGET_SPACE=log,MAXSIZE=$BUDGET_MAXSIZE,POPULATIONS=$POPULATIONS,NITERATIONS=$NITERATIONS")
  log_job "seedband ns-budget seed=$S" "$jid" "$NS_INDEX" "seed_band/z3.6_seed${S}_budget"
  if [ "$S" = "0" ]; then  # the ns_budget panel reads seed0's budget dir
    sc=$(sidecar "sc_bud_s0" "$jid" "$PROD_DIR/seed_band/z3.6_seed0_budget/refit/z3.6" "3.6")
    log_job "  sidecar ns-budget seed=0" "$sc" "afterany:$jid" "grad_faith_ns"
  fi
done

{
  echo
  echo "## Next (after all jobs finish)"
  echo "- aggregate seed band: \`scripts/aggregate_seed_band.py --band-dir $PROD_DIR/seed_band --out $PROD_DIR/seed_band/seed_band_summary.json\`"
  echo "- diagnostic figs (Phase C) read: value=\`$PROD_DIR/value/refit/z3.6\`, sobolev=\`$PROD_DIR/sobolev/refit/z3.6\`, budget=\`$PROD_DIR/seed_band/z3.6_seed0_budget/refit/z3.6\`."
} >> "$MANIFEST"

echo
echo "Submitted ${#JOBS[@]} jobs. Manifest: $MANIFEST"
echo "Watch: squeue -u $USER -A $ACCOUNT"
printf '%s\n' "${JOBS[@]}" | paste -sd, -
