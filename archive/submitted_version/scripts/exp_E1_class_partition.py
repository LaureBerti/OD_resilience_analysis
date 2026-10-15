"""
exp_E1_class_partition.py
--------------------------
E1 — Class Partition Validation

Hypothesis H1: Classes I/II/III have significantly higher rho than Class IV
across all datasets.

Setup:
  - All 18 methods × 16 datasets × p=10% × 50 replications
  - Compute both rho (Dice) and rho_tilde (Spearman) per rep
  - Statistical test: Kruskal-Wallis + Dunn post-hoc (Bonferroni)

Output files:
  results/e1_raw_results.csv
  tables/e1_mean_rho_by_class.csv
  tables/e1_kruskal_wallis.csv
  timing_per_condition.csv (appended)
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

# Suppress scipy skewness precision warnings on near-constant subsamples (harmless)
warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation")

import numpy as np
import pandas as pd
from scipy import stats

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
P_SAMPLE = 0.10
N_REPS = 50
CONTAMINATION = 0.10
BASE_SEED = 42

# For datasets with N > N_MAX_FULL, the full-dataset reference is capped at
# N_MAX_FULL samples (drawn once per dataset, shared across all methods/reps).
# This makes LOF/KNN/OCSVM feasible on Tier 3 datasets (otherwise O(N^2)).
# Noted in the paper: "Tier 3 datasets use N_ref = 5000 stratified reference."
N_MAX_FULL = 5000

E1_DATASETS = TIER1_DATASETS + TIER2_DATASETS + TIER3_DATASETS
E1_METHODS = ALL_METHODS


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
    Returns a dict with rho, rho_tilde, and timing fields.
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
        "n_full": n_full,
        "n_sample": n_sample,
        "rho": np.nan,
        "rho_tilde": np.nan,
        "time_fit_full_s": np.nan,
        "time_fit_sample_s": np.nan,
        "time_score_s": np.nan,
        "time_total_s": np.nan,
    }

    t0 = time.perf_counter()

    try:
        # Fit on full dataset
        t_fit_full_start = time.perf_counter()
        det_full = make_detector(method_name, contamination=contamination, random_state=BASE_SEED + rep)
        det_full.fit(X_full)
        t_fit_full_end = time.perf_counter()
        result["time_fit_full_s"] = t_fit_full_end - t_fit_full_start

        # Scores from full model, restricted to sample indices
        t_score_start = time.perf_counter()
        scores_full_all = det_full.decision_function(X_full)
        scores_full_restricted = scores_full_all[idx_sample_sorted]
        t_score_end = time.perf_counter()
        result["time_score_s"] = t_score_end - t_score_start

        # Fit on sample
        t_fit_sample_start = time.perf_counter()
        det_sample = make_detector(method_name, contamination=contamination, random_state=BASE_SEED + rep)
        det_sample.fit(X_sample)
        t_fit_sample_end = time.perf_counter()
        result["time_fit_sample_s"] = t_fit_sample_end - t_fit_sample_start

        # Scores from sample model on sample
        scores_sample = det_sample.decision_function(X_sample)

        # Compute resilience
        metrics = compute_resilience_pair(scores_sample, scores_full_restricted, contamination)
        result["rho"] = metrics["rho"]
        result["rho_tilde"] = metrics["rho_tilde"]

    except Exception as e:
        warnings.warn(f"Method {method_name} rep {rep} failed: {e}")

    t1 = time.perf_counter()
    result["time_total_s"] = t1 - t0
    return result


def run_experiment(datasets: dict) -> pd.DataFrame:
    """Run E1 for all method × dataset × rep combinations."""
    all_records = []
    t_script_start = time.perf_counter()

    total_conditions = len(E1_METHODS) * len(datasets) * N_REPS
    done = 0

    for ds_name, (X, y) in datasets.items():
        tier = DATASET_TIERS.get(ds_name, "?")
        t_dataset_start = time.perf_counter()

        # Cap large datasets for computational feasibility (avoids O(N^2) LOF/KNN)
        n_orig = len(X)
        if n_orig > N_MAX_FULL:
            rng_cap = np.random.default_rng(BASE_SEED)
            cap_idx = rng_cap.choice(n_orig, size=N_MAX_FULL, replace=False)
            cap_idx = np.sort(cap_idx)
            X, y = X[cap_idx], y[cap_idx]
            print(f"\nDataset: {ds_name} (N={n_orig}→{N_MAX_FULL} capped, d={X.shape[1]}, tier={tier})")
        else:
            print(f"\nDataset: {ds_name} (N={len(X)}, d={X.shape[1]}, tier={tier})")

        for method_name in E1_METHODS:
            cls = METHOD_CLASS_MAP[method_name]
            t_method_start = time.perf_counter()

            reps_records = []
            for rep in range(N_REPS):
                row = run_single_condition(X, y, method_name, P_SAMPLE, rep)
                row["dataset"] = ds_name
                row["tier"] = tier
                row["method_class"] = cls
                reps_records.append(row)
                done += 1

            all_records.extend(reps_records)
            t_method_end = time.perf_counter()
            mean_rho = np.nanmean([r["rho"] for r in reps_records])
            print(
                f"  [{cls}] {method_name}: mean_rho={mean_rho:.3f}  "
                f"({t_method_end - t_method_start:.1f}s, {done}/{total_conditions})"
            )

        t_dataset_end = time.perf_counter()
        print(f"  Dataset {ds_name} done in {t_dataset_end - t_dataset_start:.1f}s")

    t_script_end = time.perf_counter()
    elapsed = t_script_end - t_script_start
    print(f"\nE1 total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    df = pd.DataFrame(all_records)
    return df


def save_results(df: pd.DataFrame) -> None:
    """Save raw results and aggregated tables."""
    # Raw results
    raw_path = RESULTS_DIR / "e1_raw_results.csv"
    df.to_csv(raw_path, index=False)
    print(f"Raw results saved: {raw_path}")

    # Timing CSV (append compatible)
    timing_cols = ["dataset", "method", "p", "rep",
                   "time_fit_full_s", "time_fit_sample_s", "time_score_s", "time_total_s"]
    timing_path = RESULTS_DIR / "timing_per_condition.csv"
    df_timing = df[timing_cols].copy()
    df_timing["experiment"] = "E1"
    if timing_path.exists():
        df_timing.to_csv(timing_path, mode="a", header=False, index=False)
    else:
        df_timing.to_csv(timing_path, index=False)

    # Aggregated: mean±std rho by class × tier
    agg = (
        df.groupby(["tier", "method_class", "method"])
        .agg(
            mean_rho=("rho", "mean"),
            std_rho=("rho", "std"),
            mean_rho_tilde=("rho_tilde", "mean"),
            std_rho_tilde=("rho_tilde", "std"),
            n_valid=("rho", lambda x: x.notna().sum()),
        )
        .reset_index()
    )
    agg_path = TABLES_DIR / "e1_mean_rho_by_class.csv"
    agg.to_csv(agg_path, index=False)
    print(f"Aggregated table saved: {agg_path}")


def run_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kruskal-Wallis H-test across classes, followed by Dunn post-hoc
    with Bonferroni correction.
    """
    try:
        import scikit_posthocs as sp  # type: ignore
    except ImportError:
        warnings.warn(
            "scikit_posthocs not installed. Skipping Dunn test. "
            "Install with: pip install scikit-posthocs"
        )
        sp = None

    # Collect per-class vectors of rho
    classes = ["I", "II", "III", "IV"]
    class_rho = {c: df[df["method_class"] == c]["rho"].dropna().values for c in classes}

    groups = [class_rho[c] for c in classes]
    h_stat, p_value = stats.kruskal(*groups)
    print(f"\nKruskal-Wallis: H={h_stat:.4f}, p={p_value:.2e}")

    kw_records = [{"test": "Kruskal-Wallis", "H": h_stat, "p": p_value}]

    if sp is not None:
        # Build long-form DataFrame for Dunn test
        rho_long = df[["method_class", "rho"]].dropna()
        dunn = sp.posthoc_dunn(
            rho_long, val_col="rho", group_col="method_class", p_adjust="bonferroni"
        )
        dunn_path = TABLES_DIR / "e1_dunn_posthoc.csv"
        dunn.to_csv(dunn_path)
        print(f"Dunn post-hoc saved: {dunn_path}")

    kw_df = pd.DataFrame(kw_records)
    kw_path = TABLES_DIR / "e1_kruskal_wallis.csv"
    kw_df.to_csv(kw_path, index=False)
    print(f"Kruskal-Wallis results saved: {kw_path}")
    return kw_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("E1 — Class Partition Validation")
    print(f"  Methods: {len(E1_METHODS)}")
    print(f"  Datasets: {len(E1_DATASETS)} (Tier 1-3)")
    print(f"  p = {P_SAMPLE}, reps = {N_REPS}")
    print("=" * 60)

    t0 = time.perf_counter()

    # Load datasets
    print("\nLoading datasets...")
    datasets = load_multiple(E1_DATASETS, skip_on_error=True)
    if not datasets:
        print("ERROR: No datasets loaded. Check data_loader.py and data/ directory.")
        sys.exit(1)
    print(f"Loaded {len(datasets)} / {len(E1_DATASETS)} datasets.")

    # Run experiment
    df = run_experiment(datasets)

    # Save
    save_results(df)

    # Statistical tests
    run_statistical_tests(df)

    t1 = time.perf_counter()
    elapsed = t1 - t0
    print("\n" + "=" * 60)
    print(f"E1 complete. Total wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Results in: {RESULTS_DIR}")
    print(f"Tables in:  {TABLES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
