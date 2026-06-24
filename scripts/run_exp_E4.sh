#!/usr/bin/env bash
# run_exp_E4.sh — E4 Rank-Based vs Dice-Based Resilience
# Usage: bash run_exp_E4.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SCRIPT_DIR}/../../../.venv"
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
fi
export PYTHONUNBUFFERED=1
LOG_FILE="${SCRIPT_DIR}/../logs/e4_run.log"

mkdir -p "${SCRIPT_DIR}/../logs"
mkdir -p "${SCRIPT_DIR}/../results"
mkdir -p "${SCRIPT_DIR}/../tables"
mkdir -p "${SCRIPT_DIR}/../figures"

DRY_RUN=0
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN=1
    fi
done

echo "======================================"
echo "E4 — Rank-Based vs Dice-Based Resilience"
echo "Started: $(date)"
echo "======================================"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY-RUN] Would execute: python ${SCRIPT_DIR}/exp_E4_rankbased.py"
    echo "[DRY-RUN] Would plot:    python ${SCRIPT_DIR}/plot_E4_threshold.py"
    exit 0
fi

python "${SCRIPT_DIR}/exp_E4_rankbased.py" 2>&1 | tee "${LOG_FILE}"

echo ""
echo "--- Generating E4 plots ---"
python "${SCRIPT_DIR}/plot_E4_threshold.py" 2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "======================================"
echo "E4 complete. Log: ${LOG_FILE}"
echo "Finished: $(date)"
echo "======================================"
