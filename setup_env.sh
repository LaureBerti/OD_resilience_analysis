#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || { echo "ERROR: run 'python3.13 -m venv .venv' first"; exit 1; }

echo "[setup_env] $(date '+%Y-%m-%d %H:%M %Z') Installing the pinned scientific stack (requirements.txt)..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[setup_env] $(date '+%Y-%m-%d %H:%M %Z') Installing pyod 0.9.9 and combo 0.1.3 without their dependencies..."
pip install --no-deps "pyod==0.9.9" "combo==0.1.3"

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
if python -c "import numba" 2>/dev/null; then
    echo "[setup_env] numba is importable; no stub installed."
else
    echo "[setup_env] $(date '+%Y-%m-%d %H:%M %Z') Installing a numba stub (no llvmlite wheel for this platform)..."
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
fi

echo "[setup_env] $(date '+%Y-%m-%d %H:%M %Z') Patching pyod/models/rod.py for NumPy 2 (np.array(..., copy=False) -> np.asarray)..."
python - "$SITE/pyod/models/rod.py" << 'PATCH'
import sys
p = sys.argv[1]
t = open(p).read()
old, new = "np.array(res, copy=False)", "np.asarray(res)"
if old in t:
    open(p, "w").write(t.replace(old, new))
    print("patched", p)
elif new in t:
    print("already patched", p)
else:
    sys.exit("unexpected rod.py: patch not applied")
PATCH

echo "[setup_env] $(date '+%Y-%m-%d %H:%M %Z') Checking that ROD runs..."
python -c "import numpy as np; from pyod.models.rod import ROD; ROD().fit(np.random.default_rng(0).normal(size=(50, 4))); print('ROD ok')"

echo "[setup_env] $(date '+%Y-%m-%d %H:%M %Z') Done."
