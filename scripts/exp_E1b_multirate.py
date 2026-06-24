"""
exp_E1b_multirate.py
---------------------
E1b — Multi-rate Class Partition Validation

Validates the class ordering (I > II > III > IV in resilience) at
p = 1%, 5%, and 10% to show convergence behaviour across sampling rates.

Setup:
  - 18 methods (ALL_METHODS, AutoEncoder excluded — torch not available)
  - All Tier 1–3 datasets
  - p in {0.01, 0.05, 0.10}
  - N_REPS = 20 per (dataset, method, p) combination
  - Tier 3 reference capped at N_MAX_FULL = 5000

Output files:
  results/e1b_raw_results.csv
  tables/e1b_multirate_by_class.csv
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation")

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allow running from any directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = OUTPUT_DIR / "results"
TABLES_DIR = OUTPUT_DIR / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))

from data_loader import (
    TIER1_DATASETS, TIER2_DATASETS, TIER3_DATASETS, load_multiple, DATASET_TIERS
)
from method_wrappers import make_detector, METHOD_CLASS_MAP, ALL_METHODS
from resilience_metrics import compute_resilience_pair


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
P_VALUES = [0.01, 0.05, 0.10]
N_REPS = 20
CONTAMINATION = 0.10
BASE_SEED = 42
N_MAX_FULL = 5000

E1B_DATASETS = TIER1_DATASETS + TIER2_DATASETS + TIER3_DATASETS

# Exclude AutoEncoder (torch not available)
E1B_METHODS = [m for m in ALL_METHODS if m != "AutoEncoder"]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def run_single_condition(
    X: np.ndarray,
    y: np.ndarray,
    method_name: str,
    p: float,
    rep: int,
    contamination: float = CONTAMINATION,
) -> dict:
    """
    Run one (method, p, rep) condition on dataset (X, y).
    Returns a dict with rho and rho_tilde fields.
    """
    rng = np.random.default_rng(BASE_SEED + rep)
    n_full = len(X)
    n_sample = max(2, int(np.floor(p * n_full)))
    idx_sample = rng.choice(n_full, size=n_sample, replace=False)
    idx_sample_sorted = np.sort(idx_sample)

    X_sample = X[idx_sample_sorted]
    X_full = X

    result = {
        "method": method_name,
        "p": p,
        "rep": rep,
        "rho": np.nan,
        "rho_tilde": np.nan,
    }

    try:
        det_full = make_detector(method_name, contamination=contamination, random_state=BASE_SEED + rep)
        det_full.fit(X_full)
        scores_full_all = det_full.decision_function(X_full)
        scores_full_restricted = scores_full_all[idx_sample_sorted]

        det_sample = make_detector(method_name, contamination=contamination, random_state=BASE_SEED + rep)
        det_sample.fit(X_sample)
        scores_sample = det_sample.decision_function(X_sample)

        metrics = compute_resilience_pair(scores_sample, scores_full_restricted, contamination)
        result["rho"] = metrics["rho"]
        result["rho_tilde"] = metrics["rho_tilde"]

    except Exception as e:
        warnings.warn(f"Method {method_name} p={p} rep={rep} failed: {e}")

    return result


def run_experiment(datasets: dict) -> pd.DataFrame:
    """Run E1b for all method × p × dataset × rep combinations."""
    all_records = []
    t_script_start = time.perf_counter()

    total_conditions = len(E1B_METHODS) * len(P_VALUES) * len(datasets) * N_REPS
    done = 0

    for ds_name, (X, y) in datasets.items():
        tier = DATASET_TIERS.get(ds_name, "?")
        t_dataset_start = time.perf_counter()

        # Cap large datasets for computational feasibility
        n_orig = len(X)
        if n_orig > N_MAX_FULL:
            rng_cap = np.random.default_rng(BASE_SEED)
            cap_idx = rng_cap.choice(n_orig, size=N_MAX_FULL, replace=False)
            cap_idx = np.sort(cap_idx)
            X, y = X[cap_idx], y[cap_idx]
            print(f"\nDataset: {ds_name} (N={n_orig}→{N_MAX_FULL} capped, d={X.shape[1]}, tier={tier})")
        else:
            print(f"\nDataset: {ds_name} (N={len(X)}, d={X.shape[1]}, tier={tier})")

        for method_name in E1B_METHODS:
            cls = METHOD_CLASS_MAP[method_name]

            for p in P_VALUES:
                t_cond_start = time.perf_counter()
                reps_records = []

                for rep in range(N_REPS):
                    row = run_single_condition(X, y, method_name, p, rep)
                    row["dataset"] = ds_name
                    row["tier"] = tier
                    row["method_class"] = cls
                    reps_records.append(row)
                    done += 1

                all_records.extend(reps_records)
                t_cond_end = time.perf_counter()
                mean_rho = np.nanmean([r["rho"] for r in reps_records])
                print(
                    f"  [{cls}] {method_name} p={p:.2f}: mean_rho={mean_rho:.3f}  "
                    f"({t_cond_end - t_cond_start:.1f}s, {done}/{total_conditions})"
                )

        t_dataset_end = time.perf_counter()
        print(f"  Dataset {ds_name} done in {t_dataset_end - t_dataset_start:.1f}s")

    t_script_end = time.perf_counter()
    elapsed = t_script_end - t_script_start
    print(f"\nE1b total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return pd.DataFrame(all_records)


def save_results(df: pd.DataFrame) -> None:
    """Save raw results and aggregated summary table."""
    # Raw results
    raw_path = RESULTS_DIR / "e1b_raw_results.csv"
    df.to_csv(raw_path, index=False)
    print(f"Raw results saved: {raw_path}")

    # Summary: mean rho by method_class × p
    summary = (
        df.groupby(["method_class", "p"])
        .agg(
            mean_rho=("rho", "mean"),
            std_rho=("rho", "std"),
            mean_rho_tilde=("rho_tilde", "mean"),
        )
        .reset_index()
        .sort_values(["method_class", "p"])
    )
    summary_path = TABLES_DIR / "e1b_multirate_by_class.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary table saved: {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("E1b — Multi-rate Class Partition Validation")
    print(f"  Methods: {len(E1B_METHODS)} (AutoEncoder excluded)")
    print(f"  Datasets: {len(E1B_DATASETS)} (Tier 1-3)")
    print(f"  p values: {P_VALUES}")
    print(f"  Reps per condition: {N_REPS}")
    print("=" * 60)

    t0 = time.perf_counter()

    print("\nLoading datasets...")
    datasets = load_multiple(E1B_DATASETS, skip_on_error=True)
    if not datasets:
        print("ERROR: No datasets loaded. Check data_loader.py and data/ directory.")
        sys.exit(1)
    print(f"Loaded {len(datasets)} / {len(E1B_DATASETS)} datasets.")

    df = run_experiment(datasets)
    summary = save_results(df)

    # Print 4×3 summary table (class × p)
    print("\n" + "=" * 60)
    print("Summary: mean rho by class × p")
    print("=" * 60)
    pivot = summary.pivot(index="method_class", columns="p", values="mean_rho")
    pivot.columns = [f"p={c:.2f}" for c in pivot.columns]
    pivot.index.name = "class"
    print(pivot.to_string(float_format="{:.4f}".format))

    t1 = time.perf_counter()
    elapsed = t1 - t0
    print("\n" + "=" * 60)
    print(f"E1b complete. Total wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Results in: {RESULTS_DIR}")
    print(f"Tables in:  {TABLES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
