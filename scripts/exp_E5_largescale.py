"""
exp_E5_largescale.py
--------------------
E5 — Large-Scale Generalization and Runtime

Hypothesis H5:
  - Resilience class ordering holds on Tier 4 datasets (N >= 100K)
  - IForest remains rho >= 0.95 at p=1% (n=pN >> psi)
  - Runtime scales as expected (quadratic for LOF/KNN, linear for others)

Setup:
  - Class I/II/III methods (9, incl. EIF) × 3 largest datasets × p in {1, 5, 10}% × 10 reps
  - Track per-(method, sample) timing

Output:
  results/e5_raw_results.csv
  tables/e5_large_scale_rho.csv
  tables/e5_runtime.csv
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation")

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = OUTPUT_DIR / "results"
TABLES_DIR = OUTPUT_DIR / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))

from data_loader import load_multiple
from method_wrappers import make_detector, METHOD_CLASS_MAP, CLASS_I_METHODS, CLASS_II_METHODS, CLASS_III_METHODS
from resilience_metrics import compute_resilience_pair

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_SEED = 42
N_REPS = 10
P_VALUES = [0.01, 0.05, 0.10]
CONTAMINATION = 0.10

# Class IV (LOF, KNN) excluded: O(N²) complexity at N>100K is infeasible.
# AutoEncoder excluded: torch not available in benchmark environment.
E5_METHODS = CLASS_I_METHODS + CLASS_II_METHODS + CLASS_III_METHODS

# HTTP/Census/CreditCard unavailable in ADBench Classical.
# Covertype (N=286K, d=10) and skin (N=245K, d=3) are used as substitutes.
E5_DATASETS = ["covertype", "skin", "smtp"]
RUNTIME_DATASET = "covertype"   # substitute for HTTP as speedup baseline
RUNTIME_REPS = 5


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run_single_condition(
    X: np.ndarray,
    method_name: str,
    p: float,
    rep: int,
    contamination: float = CONTAMINATION,
) -> dict:
    rng = np.random.default_rng(BASE_SEED + rep)
    n_full = len(X)
    n_sample = max(2, int(np.floor(p * n_full)))
    idx = rng.choice(n_full, size=n_sample, replace=False)
    idx = np.sort(idx)

    row = {
        "method": method_name,
        "method_class": METHOD_CLASS_MAP.get(method_name, "?"),
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
        # Full model
        t_fit_full = time.perf_counter()
        det_full = make_detector(method_name, contamination=contamination, random_state=BASE_SEED + rep)
        det_full.fit(X)
        scores_full = det_full.decision_function(X)
        scores_full_restricted = scores_full[idx]
        row["time_fit_full_s"] = time.perf_counter() - t_fit_full

        # Sample model
        t_fit_sample = time.perf_counter()
        det_s = make_detector(method_name, contamination=contamination, random_state=BASE_SEED + rep)
        det_s.fit(X[idx])
        scores_sample = det_s.decision_function(X[idx])
        row["time_fit_sample_s"] = time.perf_counter() - t_fit_sample

        row["time_score_s"] = row["time_fit_full_s"]  # dominated by fit

        metrics = compute_resilience_pair(scores_sample, scores_full_restricted, contamination)
        row["rho"] = metrics["rho"]
        row["rho_tilde"] = metrics["rho_tilde"]
    except Exception as e:
        warnings.warn(f"  {method_name} p={p} rep={rep} failed: {e}")

    row["time_total_s"] = time.perf_counter() - t0
    return row


def run_full_dataset_timing(X: np.ndarray, method_name: str, n_reps: int = RUNTIME_REPS) -> dict:
    """
    Time the method on the full dataset (no sampling) for runtime comparison.
    """
    times = []
    for rep in range(n_reps):
        t0 = time.perf_counter()
        try:
            det = make_detector(method_name, contamination=CONTAMINATION, random_state=BASE_SEED + rep)
            det.fit(X)
            det.decision_function(X)
        except Exception as e:
            warnings.warn(f"Full timing failed for {method_name} rep {rep}: {e}")
            times.append(np.nan)
            continue
        times.append(time.perf_counter() - t0)

    return {
        "method": method_name,
        "n": len(X),
        "p": 1.0,
        "time_mean_s": float(np.nanmean(times)),
        "time_std_s": float(np.nanstd(times)),
        "n_reps": n_reps,
    }


def run_experiment(datasets: dict) -> pd.DataFrame:
    records = []
    total = len(datasets) * len(E5_METHODS) * len(P_VALUES) * N_REPS
    done = 0

    for ds_name, (X, _y) in datasets.items():
        print(f"\nDataset: {ds_name} (N={len(X)}, d={X.shape[1]})")
        t_ds = time.perf_counter()

        for method_name in E5_METHODS:
            cls = METHOD_CLASS_MAP.get(method_name, "?")
            for p in P_VALUES:
                for rep in range(N_REPS):
                    row = run_single_condition(X, method_name, p, rep)
                    row["dataset"] = ds_name
                    records.append(row)
                    done += 1

                # Summary per (method, p)
                subset = [r for r in records
                          if r["dataset"] == ds_name and r["method"] == method_name and r["p"] == p]
                mean_rho = np.nanmean([r["rho"] for r in subset])
                mean_t = np.nanmean([r["time_total_s"] for r in subset])
                print(
                    f"  [{cls}] {method_name} p={p:.2f}: "
                    f"rho={mean_rho:.3f}, t={mean_t:.1f}s  ({done}/{total})"
                )

        print(f"  Dataset {ds_name} done in {time.perf_counter()-t_ds:.1f}s")

    return pd.DataFrame(records)


def run_runtime_benchmark(datasets: dict) -> pd.DataFrame:
    """
    Time all methods on full HTTP dataset and at p=10% for speedup comparison.
    """
    if RUNTIME_DATASET not in datasets:
        print(f"\nRuntime benchmark skipped: '{RUNTIME_DATASET}' not loaded.")
        return pd.DataFrame()

    X, _y = datasets[RUNTIME_DATASET]
    print(f"\nRuntime benchmark on {RUNTIME_DATASET} (N={len(X)})...")

    runtime_records = []

    # Full dataset timing
    for method_name in E5_METHODS:
        print(f"  Full: {method_name}...")
        rec = run_full_dataset_timing(X, method_name)
        rec["dataset"] = RUNTIME_DATASET
        rec["sampling"] = "full"
        runtime_records.append(rec)

    # p=10% timing
    p = 0.10
    n_sample = int(np.floor(p * len(X)))
    rng = np.random.default_rng(BASE_SEED)
    idx = np.sort(rng.choice(len(X), size=n_sample, replace=False))
    X_sample = X[idx]

    for method_name in E5_METHODS:
        print(f"  p=10%: {method_name}...")
        times = []
        for rep in range(RUNTIME_REPS):
            t0 = time.perf_counter()
            try:
                det = make_detector(method_name, contamination=CONTAMINATION, random_state=BASE_SEED + rep)
                det.fit(X_sample)
                det.decision_function(X_sample)
            except Exception as e:
                warnings.warn(f"  Sampling timing failed {method_name} rep {rep}: {e}")
                times.append(np.nan)
                continue
            times.append(time.perf_counter() - t0)

        runtime_records.append({
            "method": method_name,
            "dataset": RUNTIME_DATASET,
            "n": n_sample,
            "p": p,
            "time_mean_s": float(np.nanmean(times)),
            "time_std_s": float(np.nanstd(times)),
            "n_reps": RUNTIME_REPS,
            "sampling": f"p={p:.0%}",
        })

    df_rt = pd.DataFrame(runtime_records)

    # Compute speedup
    full_times = df_rt[df_rt["sampling"] == "full"].set_index("method")["time_mean_s"]
    sample_times = df_rt[df_rt["sampling"] == "p=10%"].set_index("method")["time_mean_s"]
    speedup = full_times / sample_times
    speedup_df = speedup.reset_index()
    speedup_df.columns = ["method", "speedup"]
    speedup_df["time_full_s"] = full_times.values
    speedup_df["time_p10_s"] = sample_times.values
    speedup_df["dataset"] = RUNTIME_DATASET

    rt_path = TABLES_DIR / "e5_runtime.csv"
    speedup_df.to_csv(rt_path, index=False)
    print(f"Runtime table saved: {rt_path}")
    print("\nSpeedup summary:")
    print(speedup_df[["method", "time_full_s", "time_p10_s", "speedup"]].to_string(index=False))

    return df_rt


def main():
    print("=" * 60)
    print("E5 — Large-Scale Generalization and Runtime")
    print(f"  Methods: {len(E5_METHODS)}")
    print(f"  Datasets: {E5_DATASETS}")
    print(f"  p values: {P_VALUES}")
    print(f"  Reps: {N_REPS}")
    print("=" * 60)

    t0 = time.perf_counter()

    print("\nLoading Tier 4 datasets (this may take a while)...")
    datasets = load_multiple(E5_DATASETS, skip_on_error=True)
    print(f"Loaded {len(datasets)}/{len(E5_DATASETS)} datasets.")

    if not datasets:
        print("ERROR: No Tier 4 datasets available. Check data_loader.py.")
        sys.exit(1)

    # Main experiment
    df = run_experiment(datasets)

    # Save raw
    raw_path = RESULTS_DIR / "e5_raw_results.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nRaw results saved: {raw_path}")

    # Timing CSV
    timing_path = RESULTS_DIR / "timing_per_condition.csv"
    df_timing = df[["dataset", "method", "p", "rep",
                    "time_fit_full_s", "time_fit_sample_s", "time_score_s", "time_total_s"]].copy()
    df_timing["experiment"] = "E5"
    if timing_path.exists():
        df_timing.to_csv(timing_path, mode="a", header=False, index=False)
    else:
        df_timing.to_csv(timing_path, index=False)

    # Aggregated resilience table
    agg = (
        df.groupby(["method", "method_class", "p"])
        .agg(
            mean_rho=("rho", "mean"),
            std_rho=("rho", "std"),
            mean_rho_tilde=("rho_tilde", "mean"),
        )
        .reset_index()
    )
    agg_path = TABLES_DIR / "e5_large_scale_rho.csv"
    agg.to_csv(agg_path, index=False)
    print(f"Resilience table saved: {agg_path}")

    # Runtime benchmark
    run_runtime_benchmark(datasets)

    t1 = time.perf_counter()
    elapsed = t1 - t0
    print("\n" + "=" * 60)
    print(f"E5 complete. Total wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
