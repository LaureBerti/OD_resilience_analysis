#!/usr/bin/env bash
# run_exp_E6.sh — E6: Stratified vs. SRSWOR Sampling
# Usage: bash run_exp_E6.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SCRIPT_DIR}/../../../.venv"
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
fi
LOG_FILE="${SCRIPT_DIR}/../logs/e6_run.log"

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

# Preflight: verify required packages
python -c "import pyod, numpy, pandas, scipy, sklearn, matplotlib" 2>/dev/null || {
    echo "ERROR: Missing Python dependencies (pyod not found)."
    echo "Run: source .venv/bin/activate && pip install pyod"
    exit 1
}

echo "======================================"
echo "E6 — Stratified vs. SRSWOR Sampling"
echo "Started: $(date)"
echo "======================================"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY-RUN] Would execute: python ${SCRIPT_DIR}/exp_E6_stratified.py"
    exit 0
fi

python "${SCRIPT_DIR}/exp_E6_stratified.py" 2>&1 | tee "${LOG_FILE}"

echo ""
echo "======================================"
echo "E6 complete. Log: ${LOG_FILE}"
echo "Finished: $(date)"
echo "======================================"
