"""
data_loader.py
--------------
Unified ADBench dataset loader.

Strategy:
1. Try `from adbench.datasets import ...` (package install).
2. Fall back to loading from ./data/{dataset_name}.npz (manual download).

Dataset names follow the ADBench repository naming convention.
Returns (X, y) tuples where y is binary (0=inlier, 1=outlier).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# ADBench canonical name → fallback file name mappings
# Some datasets have slightly different names between the ADBench Python
# package and the raw .npz files in the GitHub repo.
# ---------------------------------------------------------------------------
# Maps our canonical dataset name → the .npz filename saved in ./data/
# by download_datasets.py (all saved without the ADBench numeric prefix).
# Substitutions vs. the original benchmark plan:
#   arrhythmia → vertebral  (ADBench: 39_vertebral; N≈267, Tier 1)
#   kddcup99   → smtp       (ADBench: 34_smtp;      N≈95K, network intrusion, Tier 3)
#   forestcov  → skin       (ADBench: 33_skin;      N≈245K, large tabular, Tier 3)
_NPZ_FILENAME_MAP: dict[str, str] = {
    "vertebral":  "vertebral",
    "breastw":    "breastw",
    "glass":      "glass",
    "lympho":     "lympho",
    "annthyroid": "annthyroid",
    "ionosphere": "ionosphere",
    "letter":     "letter",
    "musk":       "musk",
    "optdigits":  "optdigits",
    "pendigits":  "pendigits",
    "satellite":  "satellite",
    "thyroid":    "thyroid",
    "covertype":  "covertype",
    "shuttle":    "shuttle",
    "smtp":       "smtp",
    "skin":       "skin",
    "http":       "http",
    "mammography":"mammography",
}

# Legacy alias kept for backward compat (adbench package not installed)
_ADBENCH_NAME_MAP = _NPZ_FILENAME_MAP

# Default data directory: Outlier_resilience/data/
# scripts/ -> github_repo/ -> Outlier_resilience/ -> data/
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _load_from_adbench_package(dataset_name: str) -> Optional[tuple]:
    """
    Attempt to load via the adbench Python package.
    Returns (X, y) or None on failure.
    """
    try:
        from adbench.myutils import Utils  # type: ignore

        utils = Utils()
        data = utils.load_data(dataset=dataset_name)
        if data is None:
            return None
        X, y = data["X"], data["y"]
        y = np.asarray(y, dtype=int)
        return np.asarray(X, dtype=float), y
    except Exception:
        pass

    # Second attempt: try adbench.datasets if available
    try:
        import importlib

        mod = importlib.import_module("adbench.datasets")
        load_fn = getattr(mod, "load_dataset", None)
        if load_fn is not None:
            result = load_fn(dataset_name)
            if result is not None:
                X, y = result
                return np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    except Exception:
        pass

    return None


def _load_from_npz(dataset_name: str, data_dir: Path) -> Optional[tuple]:
    """
    Load from a .npz file in data_dir.
    Expects keys 'X' and 'y' (or 'data' and 'labels' as fallback).
    Returns (X, y) or None on failure.
    """
    fname = _NPZ_FILENAME_MAP.get(dataset_name, dataset_name)
    candidates = [
        data_dir / f"{fname}.npz",
        data_dir / f"{dataset_name}.npz",
        data_dir / fname / f"{fname}.npz",
    ]

    for path in candidates:
        if path.exists():
            try:
                npz = np.load(path, allow_pickle=False)
                # Try standard keys
                if "X" in npz and "y" in npz:
                    X = np.asarray(npz["X"], dtype=float)
                    y = np.asarray(npz["y"], dtype=int).ravel()
                    return X, y
                # Fallback key names
                if "data" in npz and "labels" in npz:
                    X = np.asarray(npz["data"], dtype=float)
                    y = np.asarray(npz["labels"], dtype=int).ravel()
                    return X, y
                if "X" in npz and "label" in npz:
                    X = np.asarray(npz["X"], dtype=float)
                    y = np.asarray(npz["label"], dtype=int).ravel()
                    return X, y
            except Exception as e:
                warnings.warn(f"Failed to load {path}: {e}")

    return None


def load_dataset(
    dataset_name: str,
    data_dir: Optional[str] = None,
) -> tuple:
    """
    Load an ADBench dataset by name.

    Parameters
    ----------
    dataset_name : str
        Canonical dataset name (e.g., 'arrhythmia', 'http').
    data_dir : str or None
        Path to directory containing .npz files.
        Defaults to ./data/ relative to project root.

    Returns
    -------
    (X, y) : tuple of np.ndarray
        X: float array of shape (N, d)
        y: int array of shape (N,), values in {0, 1}

    Raises
    ------
    FileNotFoundError
        If the dataset cannot be found by any loading strategy.
    """
    dataset_name = dataset_name.lower().strip()

    if data_dir is None:
        resolved_dir = _DEFAULT_DATA_DIR
    else:
        resolved_dir = Path(data_dir)

    # Strategy 1: adbench package
    result = _load_from_adbench_package(dataset_name)
    if result is not None:
        X, y = result
        _validate(X, y, dataset_name)
        return X, y

    # Strategy 2: .npz file
    result = _load_from_npz(dataset_name, resolved_dir)
    if result is not None:
        X, y = result
        _validate(X, y, dataset_name)
        return X, y

    raise FileNotFoundError(
        f"Dataset '{dataset_name}' not found. "
        f"Install adbench (`pip install adbench`) or place "
        f"'{dataset_name}.npz' in '{resolved_dir}'.\n"
        f"Download from: https://github.com/Minqi824/ADBench/tree/main/datasets"
    )


def _validate(X: np.ndarray, y: np.ndarray, name: str) -> None:
    """Basic sanity checks on loaded dataset."""
    if X.ndim != 2:
        raise ValueError(f"Dataset '{name}': X must be 2D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Dataset '{name}': y must be 1D, got shape {y.shape}")
    if len(X) != len(y):
        raise ValueError(
            f"Dataset '{name}': X has {len(X)} rows but y has {len(y)} elements"
        )
    unique_vals = set(np.unique(y).tolist())
    if not unique_vals.issubset({0, 1}):
        raise ValueError(
            f"Dataset '{name}': y contains non-binary values {unique_vals}"
        )
    n_outliers = int(np.sum(y))
    if n_outliers == 0:
        warnings.warn(
            f"Dataset '{name}': no labelled outliers (n_outliers=0). "
            "Dice resilience will be undefined."
        )


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

TIER1_DATASETS = ["vertebral", "breastw", "glass", "lympho"]   # N < 1000
TIER2_DATASETS = [
    "annthyroid", "ionosphere", "letter", "musk",
    "optdigits", "pendigits", "satellite", "thyroid",
]   # 1000 ≤ N ≤ 10000
TIER3_DATASETS = ["smtp", "shuttle", "covertype", "skin"]       # N > 10000
TIER4_DATASETS = ["http", "mammography"]                        # N > 100000 (unavailable — use covertype/skin for E5)

ALL_DATASETS = TIER1_DATASETS + TIER2_DATASETS + TIER3_DATASETS + TIER4_DATASETS

DATASET_TIERS: dict[str, str] = {}
for ds in TIER1_DATASETS:
    DATASET_TIERS[ds] = "T1"
for ds in TIER2_DATASETS:
    DATASET_TIERS[ds] = "T2"
for ds in TIER3_DATASETS:
    DATASET_TIERS[ds] = "T3"
for ds in TIER4_DATASETS:
    DATASET_TIERS[ds] = "T4"


def load_multiple(
    dataset_names: list,
    data_dir: Optional[str] = None,
    skip_on_error: bool = True,
) -> dict:
    """
    Load multiple datasets, optionally skipping ones that fail.

    Returns
    -------
    dict mapping dataset_name -> (X, y)
    """
    results = {}
    for name in dataset_names:
        try:
            X, y = load_dataset(name, data_dir=data_dir)
            results[name] = (X, y)
            print(
                f"  Loaded {name}: N={len(X)}, d={X.shape[1]}, "
                f"outliers={int(np.sum(y))} ({100*np.mean(y):.1f}%)"
            )
        except Exception as e:
            if skip_on_error:
                warnings.warn(f"Skipping '{name}': {e}")
            else:
                raise
    return results


if __name__ == "__main__":
    print("Testing data_loader.py with Tier 1 datasets...")
    for ds in TIER1_DATASETS:
        try:
            X, y = load_dataset(ds)
            print(f"  OK: {ds} — shape={X.shape}, n_outliers={np.sum(y)}")
        except FileNotFoundError as e:
            print(f"  MISSING: {ds} — {e}")
