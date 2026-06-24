#!/usr/bin/env bash
# run_exp_E1.sh — E1 Class Partition Validation
# Usage: bash run_exp_E1.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SCRIPT_DIR}/../../../.venv"
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
fi
LOG_FILE="${SCRIPT_DIR}/../logs/e1_run.log"

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
echo "E1 — Class Partition Validation"
echo "Started: $(date)"
echo "======================================"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY-RUN] Would execute: python ${SCRIPT_DIR}/exp_E1_class_partition.py"
    echo "[DRY-RUN] Would plot:    python ${SCRIPT_DIR}/plot_E1_violin.py"
    exit 0
fi

# Run experiment
python "${SCRIPT_DIR}/exp_E1_class_partition.py" 2>&1 | tee "${LOG_FILE}"

echo ""
echo "--- Generating E1 plots ---"
python "${SCRIPT_DIR}/plot_E1_violin.py" 2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "======================================"
echo "E1 complete. Log: ${LOG_FILE}"
echo "Finished: $(date)"
echo "======================================"
