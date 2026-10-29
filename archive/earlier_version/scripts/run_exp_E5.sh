#!/usr/bin/env bash
export PYTHONUNBUFFERED=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SCRIPT_DIR}/../../../.venv"
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
fi
LOG_FILE="${SCRIPT_DIR}/../logs/e5_run.log"

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
echo "E5 — Large-Scale Generalization and Runtime"
echo "WARNING: Long-running experiment (4-12 hrs)"
echo "Started: $(date)"
echo "======================================"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY-RUN] Would execute: python ${SCRIPT_DIR}/exp_E5_largescale.py"
    exit 0
fi

python "${SCRIPT_DIR}/exp_E5_largescale.py" 2>&1 | tee "${LOG_FILE}"

echo ""
echo "======================================"
echo "E5 complete. Log: ${LOG_FILE}"
echo "Finished: $(date)"
echo "======================================"
