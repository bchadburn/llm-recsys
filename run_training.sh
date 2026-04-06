#!/usr/bin/env bash
# run_training.sh — background training runner
# Logs all output to training.log; appends a summary line at the end.
# Usage: bash run_training.sh [--n-users N] [--n-items N] [--n-interactions N]
#
# Default: 2000 users, 5000 items, 50000 interactions (fast, ~5 min on CPU)
# Higher quality: --n-users 5000 --n-items 10000 --n-interactions 200000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/training.log"
DATA_DIR="$SCRIPT_DIR/data/instacart"

N_USERS="${N_USERS:-2000}"
N_ITEMS="${N_ITEMS:-5000}"
N_INTERACTIONS="${N_INTERACTIONS:-50000}"

# Allow CLI overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-users)         N_USERS="$2";         shift 2 ;;
        --n-items)         N_ITEMS="$2";          shift 2 ;;
        --n-interactions)  N_INTERACTIONS="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "======================================================================" > "$LOG"
echo "  Training run started: $(date)" >> "$LOG"
echo "  n_users=$N_USERS  n_items=$N_ITEMS  n_interactions=$N_INTERACTIONS" >> "$LOG"
echo "======================================================================" >> "$LOG"

cd "$SCRIPT_DIR"

uv run --with xgboost python main.py \
    --data-dir "$DATA_DIR" \
    --n-users "$N_USERS" \
    --n-items "$N_ITEMS" \
    --n-interactions "$N_INTERACTIONS" \
    >> "$LOG" 2>&1

EXIT_CODE="$?"

echo "" >> "$LOG"
if [[ "$EXIT_CODE" -eq 0 ]]; then
    echo "  FINISHED OK: $(date)" >> "$LOG"
else
    echo "  FAILED (exit $EXIT_CODE): $(date)" >> "$LOG"
fi

exit "$EXIT_CODE"
