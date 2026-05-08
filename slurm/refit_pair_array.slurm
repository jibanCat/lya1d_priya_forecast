#!/bin/bash
# SLURM array for per-pair PySR fits (Phase 2).
#
# Usage (note underscore inside pair, colon between pairs — see comment
# below; sbatch's --export uses comma as env-var separator so PAIRS
# cannot contain commas):
#   sbatch --export=ALL,REPO=...,PAYLOAD_DIR=...,OUTPUT_DIR=...,PAIRS=tau0_ns:herei_alphaq:Ap_alphaq:tau0_Ap \
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

# PAIRS is a colon-separated list of "name_i_name_j" entries (underscore
# inside the pair, since `--export=...,PAIRS=...` treats `,` as an env-var
# separator). The script translates `_` back to `,` for refit_one_pair.py.
IFS=':' read -ra PAIR_ARRAY <<< "$PAIRS"
RAW="${PAIR_ARRAY[$SLURM_ARRAY_TASK_ID]}"
# Translate first `_` to `,` (e.g. tau0_ns → tau0,ns).
PAIR="${RAW/_/,}"
# Guard: refit_one_pair.py expects exactly one comma. If a future PARAM_NAME
# contains `_`, the first-`_`-to-`,` translation above would silently
# mis-split the pair (e.g. "foo_bar_baz" → "foo,bar_baz"). Fail loudly
# instead — change separator (e.g. to `__`) before extending PARAM_NAMES.
COMMA_COUNT="$(echo -n "$PAIR" | tr -cd ',' | wc -c)"
if [[ "$COMMA_COUNT" -ne 1 ]]; then
    echo "ERROR: PAIR='$PAIR' must contain exactly one comma after the" >&2
    echo "  '_'->',' translation (raw='$RAW'). Update the slurm template" >&2
    echo "  separator if PARAM_NAMES now contains '_' characters." >&2
    exit 1
fi
echo "Task $SLURM_ARRAY_TASK_ID → pair $PAIR (from raw $RAW)"

/sw/pkgs/arc/mamba/py3.11/bin/python scripts/refit_one_pair.py \
    --pair "$PAIR" \
    --payload-dir "$PAYLOAD_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --max-retries 2
