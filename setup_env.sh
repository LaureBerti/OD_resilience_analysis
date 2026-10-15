#!/usr/bin/env bash
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
pip install "joblib>=1.0" "Pillow>=7.0"
pip install "pyod==0.9.9" --no-deps


echo "[setup_env] $(date '+%Y-%m-%d %H:%M %Z') Installing numba stub (macOS Tahoe has no llvmlite wheel yet)..."
SITE=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
mkdir -p "$SITE/numba"
cat > "$SITE/numba/__init__.py" << 'STUB'
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
