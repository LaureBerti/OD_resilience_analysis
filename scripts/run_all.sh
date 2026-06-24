#!/usr/bin/env bash
# run_all.sh — Run all experiments E1 through E5 sequentially.
#
# Usage:
#   bash run_all.sh              # run all experiments
#   bash run_all.sh --skip-e5    # skip large-scale experiment
#   bash run_all.sh --dry-run    # print what would be done
#
# Expected total runtime: 8-24 hours (E5 dominates).
# E1 + E2 + E3 + E4 typically complete in 2-4 hours.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERALL_LOG="${SCRIPT_DIR}/../logs/run_all.log"

mkdir -p "${SCRIPT_DIR}/../logs"

SKIP_E5=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --skip-e5)  SKIP_E5=1  ;;
        --dry-run)  DRY_RUN=1  ;;
    esac
done

# Activate virtual environment if present
VENV_CANDIDATES=(
    "${SCRIPT_DIR}/../../../../.venv/bin/activate"
    "${SCRIPT_DIR}/../../../../venv/bin/activate"
    "${SCRIPT_DIR}/../../../.venv/bin/activate"
)
for candidate in "${VENV_CANDIDATES[@]}"; do
    if [ -f "$candidate" ]; then
        echo "Activating venv: $candidate"
        source "$candidate"
        break
    fi
done

# Check Python dependencies
echo "Checking Python environment..."
python -c "import pyod, numpy, pandas, scipy, sklearn, matplotlib, seaborn, tqdm" 2>/dev/null || {
    echo "ERROR: Missing Python dependencies."
    echo "Run: pip install -r requirements.txt"
    exit 1
}
echo "Dependencies OK."

{
echo "======================================"
echo "FULL EXPERIMENT PIPELINE"
echo "Started: $(date)"
echo "Skip E5: ${SKIP_E5}"
echo "Dry run: ${DRY_RUN}"
echo "======================================"

T_TOTAL_START=$(python -c "import time; print(time.perf_counter())")

# ---- E1 ----
echo ""
echo "[$(date)] Starting E1..."
if [ "$DRY_RUN" = "1" ]; then
    bash "${SCRIPT_DIR}/run_exp_E1.sh" --dry-run
else
    bash "${SCRIPT_DIR}/run_exp_E1.sh"
fi
echo "[$(date)] E1 done."

# ---- E2 ----
echo ""
echo "[$(date)] Starting E2..."
if [ "$DRY_RUN" = "1" ]; then
    bash "${SCRIPT_DIR}/run_exp_E2.sh" --dry-run
else
    bash "${SCRIPT_DIR}/run_exp_E2.sh"
fi
echo "[$(date)] E2 done."

# ---- E3 ----
echo ""
echo "[$(date)] Starting E3..."
if [ "$DRY_RUN" = "1" ]; then
    bash "${SCRIPT_DIR}/run_exp_E3.sh" --dry-run
else
    bash "${SCRIPT_DIR}/run_exp_E3.sh"
fi
echo "[$(date)] E3 done."

# ---- E4 ----
echo ""
echo "[$(date)] Starting E4..."
if [ "$DRY_RUN" = "1" ]; then
    bash "${SCRIPT_DIR}/run_exp_E4.sh" --dry-run
else
    bash "${SCRIPT_DIR}/run_exp_E4.sh"
fi
echo "[$(date)] E4 done."

# ---- E5 ----
if [ "$SKIP_E5" = "0" ]; then
    echo ""
    echo "[$(date)] Starting E5 (large-scale, may take several hours)..."
    if [ "$DRY_RUN" = "1" ]; then
        bash "${SCRIPT_DIR}/run_exp_E5.sh" --dry-run
    else
        bash "${SCRIPT_DIR}/run_exp_E5.sh"
    fi
    echo "[$(date)] E5 done."
else
    echo ""
    echo "[$(date)] E5 SKIPPED (--skip-e5)."
fi

# ---- E6 ----
echo ""
echo "[$(date)] Starting E6 (stratified vs. SRSWOR, ~30-60 min)..."
if [ "$DRY_RUN" = "1" ]; then
    python -c "print('DRY RUN: exp_E6_stratified.py would run here')"
else
    python "${SCRIPT_DIR}/exp_E6_stratified.py"
fi
echo "[$(date)] E6 done."

# ---- E7 ----
echo ""
echo "[$(date)] Starting E7 (cluster sampling, ~60-120 min)..."
if [ "$DRY_RUN" = "1" ]; then
    python -c "print('DRY RUN: exp_E7_cluster.py would run here')"
else
    python "${SCRIPT_DIR}/exp_E7_cluster.py"
fi
echo "[$(date)] E7 done."

echo ""
echo "======================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "Finished: $(date)"
echo "Results: ${SCRIPT_DIR}/../results/"
echo "Tables:  ${SCRIPT_DIR}/../tables/"
echo "Figures: ${SCRIPT_DIR}/../figures/"
echo "======================================"

} 2>&1 | tee "${OVERALL_LOG}"
