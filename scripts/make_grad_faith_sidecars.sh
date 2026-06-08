#!/usr/bin/env bash
# Produce grad-faith sidecars for every param in one (or more) Pareto dirs.
#
# Each sidecar (grad_faith_<param>.csv) sits beside its pareto_<param>.csv and
# records the production derivative-gate metric per Fisher-safe candidate. Needs
# the GP emulator (read-only), so run on a GP-capable node with the project .venv.
#
# Usage: scripts/make_grad_faith_sidecars.sh <pareto_dir> <z> [extra eval args...]
set -euo pipefail
DIR="$1"; Z="$2"; shift 2
export PYTHON_JULIAPKG_PROJECT="$HOME/.julia_env"
export JULIA_DEPOT_PATH="$HOME/.julia"
export PYTHONPATH="src:/home/mfho/student_projects/lya_emulator_full"
PY="${PYTHON:-.venv/bin/python}"
PARAMS="dtau0 tau0 ns Ap herei heref alphaq hub omegamh2 hireionz bhfeedback"
for p in $PARAMS; do
  csv="$DIR/pareto_${p}.csv"
  [ -f "$csv" ] || { echo "skip $p (no $csv)"; continue; }
  echo "=== $p ==="
  # Continue the sweep even if one param fails (diagnostic run: collect what we can).
  "$PY" scripts/eval_grad_faithfulness.py \
    --pareto "$csv" --param "$p" --z "$Z" --basedir data/kodiaq_gp \
    --log-space --out "$DIR/grad_faith_${p}.csv" "$@" \
    || echo "FAILED $p (continuing)"
done
