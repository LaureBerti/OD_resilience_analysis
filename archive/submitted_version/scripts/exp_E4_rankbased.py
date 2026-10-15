"""
exp_E4_rankbased.py
-------------------
E4 — Rank-Based vs Dice-Based Resilience

Hypothesis H4:
  - rho_tilde >= rho universally (Wilcoxon signed-rank p < 0.001)
  - rho_tilde is stable under threshold variation while rho varies substantially

Setup:
  Part A: All 18 methods × 12 datasets × p=10% × 50 reps
          Compute rho (top-10% outliers as threshold) and rho_tilde
          Test rho_tilde >= rho with Wilcoxon signed-rank test

  Part B: IForest + LOF × 5 datasets × p=10% × 50 reps
          × threshold k in {5, 8, 10, 12, 15}% of N
          Plot rho as function of k, overlay constant rho_tilde

Output:
  results/e4_raw_results.csv
  tables/e4_rho_vs_rhotilde.csv
  tables/e4_threshold_sensitivity.csv
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation")

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = OUTPUT_DIR / "results"
TABLES_DIR = OUTPUT_DIR / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))

from data_loader import load_multiple
from method_wrappers import make_detector, ALL_METHODS
from resilience_metrics import dice_resilience, rank_resilience, scores_to_labels

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_SEED = 42
N_REPS = 50
P_SAMPLE = 0.10

# Part A: 12 datasets
PART_A_DATASETS = [
    "vertebral", "breastw", "glass", "lympho",         # T1 (vertebral substitutes arrhythmia)
    "annthyroid", "ionosphere", "pendigits", "thyroid", # T2
    "smtp", "shuttle", "covertype", "skin",             # T3 (smtp/skin substitute kddcup99/forestcov)
]

# Part B: 5 datasets, 2 methods
PART_B_DATASETS = [
    "annthyroid", "satellite", "smtp", "shuttle", "skin"
]
PART_B_METHODS = ["IForest", "LOF"]
THRESHOLD_PCT_VALUES = [0.05, 0.08, 0.10, 0.12, 0.15]


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run_single_rep_part_a(
    X: np.ndarray,
    method_name: str,
    p: float,
    rep: int,
    contamination: float = 0.10,
) -> dict:
    rng = np.random.default_rng(BASE_SEED + rep)
    n_full = len(X)
    n_sample = max(2, int(np.floor(p * n_full)))
    idx = rng.choice(n_full, size=n_sample, replace=False)
    idx = np.sort(idx)

    row = {
        "method": method_name,
        "p": p,
        "rep": rep,
        "threshold_pct": contamination,
        "n_full": n_full,
        "n_sample": n_sample,
        "rho": np.nan,
        "rho_tilde": np.nan,
        "time_total_s": np.nan,
    }

    t0 = time.perf_counter()
    try:
        det_full = make_detector(method_name, contamination=contamination, random_state=BASE_SEED + rep)
        det_full.fit(X)
        scores_full_all = det_full.decision_function(X)
        scores_full_restricted = scores_full_all[idx]

        det_s = make_detector(method_name, contamination=contamination, random_state=BASE_SEED + rep)
        det_s.fit(X[idx])
        scores_sample = det_s.decision_function(X[idx])

        labels_s = scores_to_labels(scores_sample, contamination)
        labels_f = scores_to_labels(scores_full_restricted, contamination)

        row["rho"] = dice_resilience(labels_s, labels_f)
        row["rho_tilde"] = rank_resilience(scores_sample, scores_full_restricted)
    except Exception as e:
        warnings.warn(f"  {method_name} rep={rep} failed: {e}")

    row["time_total_s"] = time.perf_counter() - t0
    return row


def run_single_rep_part_b(
    X: np.ndarray,
    method_name: str,
    p: float,
    rep: int,
    threshold_pct: float,
) -> dict:
    """
    Run one rep and compute rho at a specific threshold_pct,
    plus rho_tilde (threshold-free).
    """
    rng = np.random.default_rng(BASE_SEED + rep)
    n_full = len(X)
    n_sample = max(2, int(np.floor(p * n_full)))
    idx = rng.choice(n_full, size=n_sample, replace=False)
    idx = np.sort(idx)

    row = {
        "method": method_name,
        "p": p,
        "rep": rep,
        "threshold_pct": threshold_pct,
        "n_full": n_full,
        "n_sample": n_sample,
        "rho": np.nan,
        "rho_tilde": np.nan,
        "time_total_s": np.nan,
    }

    t0 = time.perf_counter()
    try:
        det_full = make_detector(method_name, contamination=threshold_pct, random_state=BASE_SEED + rep)
        det_full.fit(X)
        scores_full_all = det_full.decision_function(X)
        scores_full_restricted = scores_full_all[idx]

        det_s = make_detector(method_name, contamination=threshold_pct, random_state=BASE_SEED + rep)
        det_s.fit(X[idx])
        scores_sample = det_s.decision_function(X[idx])

        labels_s = scores_to_labels(scores_sample, threshold_pct)
        labels_f = scores_to_labels(scores_full_restricted, threshold_pct)

        row["rho"] = dice_resilience(labels_s, labels_f)
        row["rho_tilde"] = rank_resilience(scores_sample, scores_full_restricted)
    except Exception as e:
        warnings.warn(f"  {method_name} threshold={threshold_pct} rep={rep} failed: {e}")

    row["time_total_s"] = time.perf_counter() - t0
    return row


def run_part_a(datasets: dict) -> pd.DataFrame:
    print("\n--- Part A: rho vs rho_tilde across all methods ---")
    N_MAX_FULL = 5000
    records = []
    for ds_name, (X, _y) in datasets.items():
        if ds_name not in PART_A_DATASETS:
            continue
        if len(X) > N_MAX_FULL:
            rng_cap = np.random.default_rng(42)
            idx = np.sort(rng_cap.choice(len(X), size=N_MAX_FULL, replace=False))
            X = X[idx]
            print(f"  Dataset: {ds_name} (N→{N_MAX_FULL} capped)")
        else:
            print(f"  Dataset: {ds_name} (N={len(X)})")
        for method_name in ALL_METHODS:
            for rep in range(N_REPS):
                row = run_single_rep_part_a(X, method_name, P_SAMPLE, rep)
                row["dataset"] = ds_name
                row["part"] = "A"
                records.append(row)
            subset = [r for r in records if r["dataset"] == ds_name and r["method"] == method_name]
            mean_rho = np.nanmean([r["rho"] for r in subset])
            mean_rho_t = np.nanmean([r["rho_tilde"] for r in subset])
            print(f"    {method_name}: rho={mean_rho:.3f}, rho_tilde={mean_rho_t:.3f}")
    return pd.DataFrame(records)


def run_part_b(datasets: dict) -> pd.DataFrame:
    print("\n--- Part B: Threshold sensitivity (rho vs threshold_pct) ---")
    N_MAX_FULL = 5000
    records = []
    for ds_name, (X, _y) in datasets.items():
        if ds_name not in PART_B_DATASETS:
            continue
        if len(X) > N_MAX_FULL:
            rng_cap = np.random.default_rng(42)
            idx = np.sort(rng_cap.choice(len(X), size=N_MAX_FULL, replace=False))
            X = X[idx]
            print(f"  Dataset: {ds_name} (N→{N_MAX_FULL} capped)")
        else:
            print(f"  Dataset: {ds_name} (N={len(X)})")
        for method_name in PART_B_METHODS:
            for threshold_pct in THRESHOLD_PCT_VALUES:
                for rep in range(N_REPS):
                    row = run_single_rep_part_b(X, method_name, P_SAMPLE, rep, threshold_pct)
                    row["dataset"] = ds_name
                    row["part"] = "B"
                    records.append(row)
            # Summary by threshold
            for threshold_pct in THRESHOLD_PCT_VALUES:
                subset = [
                    r for r in records
                    if r["dataset"] == ds_name
                    and r["method"] == method_name
                    and r["threshold_pct"] == threshold_pct
                    and r["part"] == "B"
                ]
                mean_rho = np.nanmean([r["rho"] for r in subset])
                mean_rho_t = np.nanmean([r["rho_tilde"] for r in subset])
                print(
                    f"    {method_name} k={threshold_pct:.2f}: "
                    f"rho={mean_rho:.3f}, rho_tilde={mean_rho_t:.3f}"
                )
    return pd.DataFrame(records)


def run_wilcoxon_test(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wilcoxon signed-rank test: rho_tilde >= rho per method.
    """
    records = []
    for method_name, grp in df.groupby("method"):
        valid = grp.dropna(subset=["rho", "rho_tilde"])
        if len(valid) < 10:
            continue
        rho_vals = valid["rho"].values
        rho_tilde_vals = valid["rho_tilde"].values
        diff = rho_tilde_vals - rho_vals
        mean_diff = float(np.nanmean(diff))

        try:
            stat, p_val = stats.wilcoxon(
                rho_tilde_vals, rho_vals, alternative="greater"
            )
        except Exception as e:
            warnings.warn(f"Wilcoxon failed for {method_name}: {e}")
            stat, p_val = np.nan, np.nan

        records.append({
            "method": method_name,
            "mean_rho": float(np.nanmean(rho_vals)),
            "mean_rho_tilde": float(np.nanmean(rho_tilde_vals)),
            "mean_diff": mean_diff,
            "wilcoxon_statistic": stat,
            "wilcoxon_p": p_val,
            "n_pairs": len(valid),
        })

    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("E4 — Rank-Based vs Dice-Based Resilience")
    print(f"  Part A: {len(ALL_METHODS)} methods × {len(PART_A_DATASETS)} datasets × {N_REPS} reps")
    print(f"  Part B: {PART_B_METHODS} × {len(PART_B_DATASETS)} datasets × {THRESHOLD_PCT_VALUES}")
    print("=" * 60)

    t0 = time.perf_counter()

    all_datasets = list(set(PART_A_DATASETS + PART_B_DATASETS))
    print("\nLoading datasets...")
    datasets = load_multiple(all_datasets, skip_on_error=True)
    print(f"Loaded {len(datasets)}/{len(all_datasets)} datasets.")

    df_a = run_part_a(datasets)
    df_b = run_part_b(datasets)

    df_all = pd.concat([df_a, df_b], ignore_index=True)
    raw_path = RESULTS_DIR / "e4_raw_results.csv"
    df_all.to_csv(raw_path, index=False)
    print(f"\nRaw results saved: {raw_path}")

    # Timing
    timing_path = RESULTS_DIR / "timing_per_condition.csv"
    df_timing = df_all[["dataset", "method", "p", "rep", "time_total_s"]].copy()
    df_timing["experiment"] = "E4"
    if timing_path.exists():
        df_timing.to_csv(timing_path, mode="a", header=False, index=False)
    else:
        df_timing.to_csv(timing_path, index=False)

    # Wilcoxon test (Part A)
    print("\nRunning Wilcoxon signed-rank tests...")
    wilcoxon_df = run_wilcoxon_test(df_a)
    wilcoxon_path = TABLES_DIR / "e4_rho_vs_rhotilde.csv"
    wilcoxon_df.to_csv(wilcoxon_path, index=False)
    print(f"Wilcoxon results saved: {wilcoxon_path}")
    print("\nWilcoxon summary:")
    print(wilcoxon_df[["method", "mean_rho", "mean_rho_tilde", "mean_diff", "wilcoxon_p"]].to_string(index=False))

    # Threshold sensitivity summary (Part B)
    thresh_agg = (
        df_b.groupby(["method", "dataset", "threshold_pct"])
        .agg(mean_rho=("rho", "mean"), mean_rho_tilde=("rho_tilde", "mean"))
        .reset_index()
    )
    thresh_path = TABLES_DIR / "e4_threshold_sensitivity.csv"
    thresh_agg.to_csv(thresh_path, index=False)
    print(f"\nThreshold sensitivity saved: {thresh_path}")

    t1 = time.perf_counter()
    elapsed = t1 - t0
    print("\n" + "=" * 60)
    print(f"E4 complete. Total wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
