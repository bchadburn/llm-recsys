#!/usr/bin/env bash
# Runs all 5 LLM experiments sequentially, logs each independently,
# then generates overnight_report.md.
# Safe to re-run: Exp 1 caches API results; Exps 2-4 are deterministic.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

PYTHON_DEPS="anthropic sentence-transformers faiss-cpu torch numpy pandas lightgbm xgboost python-dotenv"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_experiment() {
    local num="$1"
    local script="$2"
    local logfile="$3"
    log "Starting Experiment $num: $script"
    # Build --with flags from space-separated deps
    local with_flags
    with_flags=$(printf -- '--with %s ' $PYTHON_DEPS)
    if uv run $with_flags python "$script" > "$logfile" 2>&1; then
        log "Experiment $num COMPLETE"
    else
        log "Experiment $num FAILED — check $logfile for details"
    fi
}

log "======================================================"
log "  OVERNIGHT LLM EXPERIMENTS"
log "  Started: $(date)"
log "======================================================"

# Verify API key present
if ! grep -q "ANTHROPIC_API_KEY" "$HOME/.env" 2>/dev/null; then
    log "ERROR: ANTHROPIC_API_KEY not found in ~/.env — aborting"
    exit 1
fi
log "API key found in ~/.env"

run_experiment 1 "llm_item_enrichment.py" "$LOG_DIR/exp1_enrichment.log"
run_experiment 2 "llm_user_narration.py"  "$LOG_DIR/exp2_narration.log"
run_experiment 3 "llm_reranker.py"        "$LOG_DIR/exp3_reranker.log"
run_experiment 4 "synthetic_context.py"   "$LOG_DIR/exp4_synthetic.log"
run_experiment 5 "exp5_dual_head.py"      "$LOG_DIR/exp5_dual_head.log"

log "Generating overnight_report.md..."
uv run --with python-dotenv python generate_report.py

log "======================================================"
log "  ALL EXPERIMENTS DONE"
log "  Results: overnight_report.md"
log "  Full logs: logs/"
log "  Finished: $(date)"
log "======================================================"
