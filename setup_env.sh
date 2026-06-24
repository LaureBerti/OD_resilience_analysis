#!/usr/bin/env bash
# setup_env.sh — cmake-free install for Python 3.13 on macOS
#
# All pyod versions require numba → llvmlite → cmake.
# No pre-built llvmlite wheel exists for macOS + Python 3.13.
# This script installs pyod WITHOUT its optional numba dependency,
# which is only needed for LUNAR/DevNet (not used in this benchmark).
#
# COPOD, ECOD, HBOS, LOF, KNN, OCSVM, IsolationForest all work without numba.
# AutoEncoder is excluded (needs torch; re-enable by uncommenting below).
#
# Alternative: brew install cmake && pip install -r requirements.txt
set -e

cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || { echo "ERROR: run 'python3.13 -m venv .venv' first"; exit 1; }

echo "[setup_env] $(date '+%Y-%m-%d %H:%M %Z') Installing core scientific stack..."
pip install --upgrade pip
pip install \
    "numpy>=1.24" \
    "pandas>=2.0" \
    "scipy>=1.11" \
    "scikit-learn>=1.3" \
    "matplotlib>=3.7" \
    "seaborn>=0.12" \
    "tqdm>=4.65" \
    "scikit-posthocs>=0.7.0"

echo "[setup_env] $(date '+%Y-%m-%d %H:%M %Z') Installing pyod 0.9.9 WITHOUT numba dependency..."
# pyod 3.x imports numba unconditionally in pyod/utils/stat_models.py.
# pyod 0.9.9 (Oct 2022) predates that change and has no numba in core utils.
# COPOD (0.9.0+), ECOD (0.9.7+), HBOS, LOF, KNN, OCSVM, IForest all present.
pip install "joblib>=1.0" "Pillow>=7.0"
pip install "pyod==0.9.9" --no-deps

# Optional: uncomment to enable AutoEncoder (requires cmake to build llvmlite first)
# pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "[setup_env] $(date '+%Y-%m-%d %H:%M %Z') Installing numba stub (macOS Tahoe has no llvmlite wheel yet)..."
SITE=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
mkdir -p "$SITE/numba"
cat > "$SITE/numba/__init__.py" << 'STUB'
# Numba stub — @njit is a no-op decorator; functions run as plain Python.
# Replace with real numba once llvmlite ships a macOS-26 pre-built wheel.
def njit(*args, **kwargs):
    if len(args) == 1 and callable(args[0]):
        return args[0]
    def decorator(func):
        return func
    return decorator
jit = njit
vectorize = njit
guvectorize = njit
STUB

echo "[setup_env] $(date '+%Y-%m-%d %H:%M %Z') Done."
echo ""
echo "NOTE: numba stub installed — KNN/LOF distance functions run as plain Python (slower, correct)."
echo "NOTE: AutoEncoder detector is disabled (needs torch)."
echo "To upgrade to real numba: brew install llvm@14 && LLVM_DIR=\$(brew --prefix llvm@14)/lib/cmake/llvm pip install numba"
