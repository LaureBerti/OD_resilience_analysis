"""
exp_E2_convergence.py
---------------------
E2 — Convergence Rate Fitting

Hypothesis H2:
  Part A: For Class I (MAD), 1-rho decreases as (pN)^{-alpha} with alpha~0.5;
          for Class II (ECOD), alpha faster than 0.5;
          for Class IV (LOF), alpha << 0.5.
  Part B: IForest shows phase-transition at n=pN=psi=256 (sharp increase in rho).

Setup:
  Part A: MAD, ECOD, LOF × 8 medium/large datasets × p in {2,5,10,15,20,30,50}% × 30 reps
          Fit 1-rho_bar = A*(pN)^{-alpha}: report alpha_hat per method
  Part B: IForest × 6 datasets of varying N × p in {0.5,1,2,5,10,20,50}% × 30 reps
          Plot rho vs n=pN with vertical line at n=psi=256

Output:
  results/e2_raw_results.csv
  tables/e2_alpha_estimates.csv
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
from method_wrappers import make_detector
from resilience_metrics import compute_resilience_pair

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_SEED = 42
N_REPS = 30
CONTAMINATION = 0.10
PSI = 256  # IForest subsampling size

PART_A_METHODS = ["MAD", "ECOD", "LOF"]
PART_A_P_VALUES = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

# 8 medium/large datasets for Part A
PART_A_DATASETS = [
    "annthyroid", "musk", "pendigits", "satellite", "thyroid",
    "smtp", "shuttle", "skin",
]

PART_B_P_VALUES = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
PART_B_DATASETS = [
    "vertebral",    # N~267  — below psi for small p (substituting arrhythmia)
    "breastw",      # N~683
    "annthyroid",   # N~7200
    "satellite",    # N~6435
    "smtp",         # N~95156 (substituting kddcup99)
    "shuttle",      # N~49097
]


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def compute_rho_at_condition(
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
        "p": p,
        "rep": rep,
        "n_full": n_full,
        "n_sample": n_sample,
        "rho": np.nan,
        "rho_tilde": np.nan,
        "time_fit_s": np.nan,
        "time_score_s": np.nan,
        "time_total_s": np.nan,
    }

    t0 = time.perf_counter()
    try:
        t_fit = time.perf_counter()
        det_full = make_detector(method_name, contamination=contamination, random_state=BASE_SEED + rep)
        det_full.fit(X)
        scores_full = det_full.decision_function(X)
        scores_full_restricted = scores_full[idx]
        row["time_fit_s"] = time.perf_counter() - t_fit

        t_score = time.perf_counter()
        det_s = make_detector(method_name, contamination=contamination, random_state=BASE_SEED + rep)
        det_s.fit(X[idx])
        scores_sample = det_s.decision_function(X[idx])
        row["time_score_s"] = time.perf_counter() - t_score

        metrics = compute_resilience_pair(scores_sample, scores_full_restricted, contamination)
        row["rho"] = metrics["rho"]
        row["rho_tilde"] = metrics["rho_tilde"]
    except Exception as e:
        warnings.warn(f"  {method_name} p={p:.3f} rep={rep} failed: {e}")

    row["time_total_s"] = time.perf_counter() - t0
    return row


def run_part_a(datasets: dict) -> pd.DataFrame:
    print("\n--- Part A: Convergence Rate Fitting (MAD, ECOD, LOF) ---")
    records = []
    N_MAX_FULL = 5000
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
        for method_name in PART_A_METHODS:
            for p in PART_A_P_VALUES:
                n_s = int(np.floor(p * len(X)))
                for rep in range(N_REPS):
                    row = compute_rho_at_condition(X, method_name, p, rep)
                    row["dataset"] = ds_name
                    row["part"] = "A"
                    records.append(row)
            rhos = [r["rho"] for r in records if r["dataset"] == ds_name and r["method"] == method_name]
            print(f"    {method_name}: mean_rho={np.nanmean(rhos):.3f}")
    return pd.DataFrame(records)


def run_part_b(datasets: dict) -> pd.DataFrame:
    print("\n--- Part B: IForest Phase Transition ---")
    records = []
    N_MAX_FULL = 5000
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
        for p in PART_B_P_VALUES:
            n_s = max(2, int(np.floor(p * len(X))))
            for rep in range(N_REPS):
                row = compute_rho_at_condition(X, "IForest", p, rep)
                row["dataset"] = ds_name
                row["part"] = "B"
                row["n_vs_psi"] = "above" if n_s >= PSI else "below"
                records.append(row)
        by_p = pd.DataFrame([r for r in records if r["dataset"] == ds_name and r["part"] == "B"])
        for p_val in PART_B_P_VALUES:
            subset = by_p[by_p["p"] == p_val]["rho"]
            n_s = max(2, int(np.floor(p_val * len(X))))
            marker = "*" if n_s >= PSI else " "
            print(f"    {marker} p={p_val:.3f} n={n_s:5d}  mean_rho={np.nanmean(subset):.3f}")
    return pd.DataFrame(records)


def fit_power_law(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (method, dataset) fit: log(1-rho_bar) = log(A) + alpha*log(n)
    Returns DataFrame with columns: method, dataset, alpha_hat, alpha_ci_low,
    alpha_ci_high, R2, n_points.
    """
    records = []
    for (method, ds), grp in df.groupby(["method", "dataset"]):
        # Average over reps
        mean_by_n = grp.groupby("n_sample")["rho"].mean().reset_index()
        mean_by_n.columns = ["n", "rho_bar"]
        mean_by_n = mean_by_n[mean_by_n["rho_bar"] < 1.0]  # avoid log(0)
        mean_by_n = mean_by_n[mean_by_n["rho_bar"] > 0.0]
        mean_by_n["one_minus_rho"] = 1.0 - mean_by_n["rho_bar"]
        mean_by_n = mean_by_n[mean_by_n["one_minus_rho"] > 0]

        if len(mean_by_n) < 3:
            continue

        log_n = np.log(mean_by_n["n"].values)
        log_res = np.log(mean_by_n["one_minus_rho"].values)

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_res)
        r2 = r_value ** 2
        ci_low = slope - 1.96 * std_err
        ci_high = slope + 1.96 * std_err

        records.append({
            "method": method,
            "dataset": ds,
            "alpha_hat": slope,
            "alpha_ci_low": ci_low,
            "alpha_ci_high": ci_high,
            "R2": r2,
            "p_value": p_value,
            "n_points": len(mean_by_n),
        })

    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("E2 — Convergence Rate Fitting")
    print(f"  Part A methods: {PART_A_METHODS}")
    print(f"  Part A p values: {PART_A_P_VALUES}")
    print(f"  Part B: IForest phase transition at psi={PSI}")
    print(f"  Reps: {N_REPS}")
    print("=" * 60)

    t0 = time.perf_counter()

    all_datasets_needed = list(set(PART_A_DATASETS + PART_B_DATASETS))
    print("\nLoading datasets...")
    datasets = load_multiple(all_datasets_needed, skip_on_error=True)
    print(f"Loaded {len(datasets)}/{len(all_datasets_needed)} datasets.")

    df_a = run_part_a(datasets)
    df_b = run_part_b(datasets)

    # Combine and save raw
    df_b_iforest = df_b.copy()
    df_b_iforest["method"] = "IForest"
    df_all = pd.concat([df_a, df_b_iforest], ignore_index=True)
    raw_path = RESULTS_DIR / "e2_raw_results.csv"
    df_all.to_csv(raw_path, index=False)
    print(f"\nRaw results saved: {raw_path}")

    # Timing
    timing_path = RESULTS_DIR / "timing_per_condition.csv"
    df_timing = df_all[["dataset", "method", "p", "rep",
                         "time_fit_s", "time_score_s", "time_total_s"]].copy()
    df_timing["experiment"] = "E2"
    if timing_path.exists():
        df_timing.to_csv(timing_path, mode="a", header=False, index=False)
    else:
        df_timing.to_csv(timing_path, index=False)

    # Fit power law for Part A
    print("\nFitting power-law convergence rates...")
    alpha_df = fit_power_law(df_a)
    alpha_path = TABLES_DIR / "e2_alpha_estimates.csv"
    alpha_df.to_csv(alpha_path, index=False)
    print(f"Alpha estimates saved: {alpha_path}")

    # Summary
    print("\nAlpha estimates summary:")
    summary = alpha_df.groupby("method").agg(
        mean_alpha=("alpha_hat", "mean"),
        std_alpha=("alpha_hat", "std"),
    )
    print(summary.to_string())

    # IForest phase transition summary
    print("\nIForest phase transition (Part B):")
    pt = df_b.groupby(["dataset", "n_sample"])["rho"].mean().reset_index()
    for ds in PART_B_DATASETS:
        sub = pt[pt["dataset"] == ds].sort_values("n_sample")
        if len(sub) == 0:
            continue
        print(f"  {ds}:")
        for _, row in sub.iterrows():
            marker = ">>> TRANSITION <<<" if abs(row["n_sample"] - PSI) < 50 else ""
            print(f"    n={row['n_sample']:6d}  rho={row['rho']:.3f}  {marker}")

    t1 = time.perf_counter()
    elapsed = t1 - t0
    print("\n" + "=" * 60)
    print(f"E2 complete. Total wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
