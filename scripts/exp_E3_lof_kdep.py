"""
exp_E3_lof_kdep.py
------------------
E3 — LOF k-Dependence Validation

Hypothesis H3: For LOF, 1-rho scales linearly with 1/sqrt(kp)
(from the hypergeometric neighborhood bound in Theorem 1).

Setup:
  - LOF × k in {3, 5, 10, 15, 20} × p in {5, 10, 20}% × 8 datasets × 30 reps
  - KNN as control (same k parameter)
  - Regress 1-rho on 1/sqrt(kp): report R^2 and slope

Output:
  results/e3_raw_results.csv
  tables/e3_regression_results.csv
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

K_VALUES = [3, 5, 10, 15, 20]
P_VALUES = [0.05, 0.10, 0.20]

E3_METHODS = ["LOF", "KNN"]  # KNN as control

E3_DATASETS = [
    "glass", "lympho",         # Tier 1
    "annthyroid", "thyroid",   # Tier 2
    "letter", "optdigits",     # Tier 2
    "smtp", "skin",             # Tier 3 (substituting kddcup99, forestcov)
]


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def compute_rho_k_condition(
    X: np.ndarray,
    method_name: str,
    k: int,
    p: float,
    rep: int,
    contamination: float = CONTAMINATION,
) -> dict:
    rng = np.random.default_rng(BASE_SEED + rep)
    n_full = len(X)
    n_sample = max(k + 2, int(np.floor(p * n_full)))
    idx = rng.choice(n_full, size=n_sample, replace=False)
    idx = np.sort(idx)

    row = {
        "method": method_name,
        "k": k,
        "p": p,
        "kp_inv_sqrt": 1.0 / np.sqrt(k * p),
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
        # Fit on full
        t_fit = time.perf_counter()
        det_full = make_detector(
            method_name, contamination=contamination,
            random_state=BASE_SEED + rep, n_neighbors=k
        )
        det_full.fit(X)
        scores_full = det_full.decision_function(X)
        scores_full_restricted = scores_full[idx]
        row["time_fit_s"] = time.perf_counter() - t_fit

        # Fit on sample
        t_score = time.perf_counter()
        effective_k = min(k, n_sample - 1)
        det_s = make_detector(
            method_name, contamination=contamination,
            random_state=BASE_SEED + rep, n_neighbors=effective_k
        )
        det_s.fit(X[idx])
        scores_sample = det_s.decision_function(X[idx])
        row["time_score_s"] = time.perf_counter() - t_score

        metrics = compute_resilience_pair(scores_sample, scores_full_restricted, contamination)
        row["rho"] = metrics["rho"]
        row["rho_tilde"] = metrics["rho_tilde"]
    except Exception as e:
        warnings.warn(f"  {method_name} k={k} p={p:.2f} rep={rep} failed: {e}")

    row["time_total_s"] = time.perf_counter() - t0
    return row


def run_experiment(datasets: dict) -> pd.DataFrame:
    records = []
    total = len(E3_DATASETS) * len(E3_METHODS) * len(K_VALUES) * len(P_VALUES) * N_REPS
    done = 0

    N_MAX_FULL = 5000
    for ds_name, (X, _y) in datasets.items():
        if ds_name not in E3_DATASETS:
            continue
        if len(X) > N_MAX_FULL:
            rng_cap = np.random.default_rng(42)
            idx = np.sort(rng_cap.choice(len(X), size=N_MAX_FULL, replace=False))
            X = X[idx]
            print(f"\nDataset: {ds_name} (N→{N_MAX_FULL} capped)")
        else:
            print(f"\nDataset: {ds_name} (N={len(X)})")

        for method_name in E3_METHODS:
            for k in K_VALUES:
                for p in P_VALUES:
                    for rep in range(N_REPS):
                        row = compute_rho_k_condition(X, method_name, k, p, rep)
                        row["dataset"] = ds_name
                        records.append(row)
                        done += 1

                    # Progress
                    subset = [r for r in records
                               if r["dataset"] == ds_name
                               and r["method"] == method_name
                               and r["k"] == k
                               and r["p"] == p]
                    mean_rho = np.nanmean([r["rho"] for r in subset])
                    print(
                        f"  {method_name} k={k:2d} p={p:.2f}: "
                        f"mean_rho={mean_rho:.3f}  ({done}/{total})"
                    )

    return pd.DataFrame(records)


def fit_linear_regression(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (method, dataset, p): regress 1-rho on 1/sqrt(kp).
    Returns DataFrame with R^2, slope, intercept, p_value.
    """
    records = []
    df["one_minus_rho"] = 1.0 - df["rho"]

    for (method, ds, p_val), grp in df.groupby(["method", "dataset", "p"]):
        grp_valid = grp.dropna(subset=["one_minus_rho", "kp_inv_sqrt"])
        if len(grp_valid) < 5:
            continue

        x = grp_valid["kp_inv_sqrt"].values
        y = grp_valid["one_minus_rho"].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r2 = r_value ** 2

        records.append({
            "method": method,
            "dataset": ds,
            "p": p_val,
            "slope": slope,
            "intercept": intercept,
            "R2": r2,
            "p_value": p_value,
            "std_err": std_err,
            "n_obs": len(grp_valid),
        })

    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("E3 — LOF k-Dependence Validation")
    print(f"  Methods: {E3_METHODS}")
    print(f"  k values: {K_VALUES}")
    print(f"  p values: {P_VALUES}")
    print(f"  Datasets: {E3_DATASETS}")
    print(f"  Reps: {N_REPS}")
    print("=" * 60)

    t0 = time.perf_counter()

    print("\nLoading datasets...")
    datasets = load_multiple(E3_DATASETS, skip_on_error=True)
    print(f"Loaded {len(datasets)}/{len(E3_DATASETS)} datasets.")

    df = run_experiment(datasets)

    # Save raw
    raw_path = RESULTS_DIR / "e3_raw_results.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nRaw results saved: {raw_path}")

    # Timing
    timing_path = RESULTS_DIR / "timing_per_condition.csv"
    df_timing = df[["dataset", "method", "p", "rep",
                    "time_fit_s", "time_score_s", "time_total_s"]].copy()
    df_timing["experiment"] = "E3"
    if timing_path.exists():
        df_timing.to_csv(timing_path, mode="a", header=False, index=False)
    else:
        df_timing.to_csv(timing_path, index=False)

    # Regression
    print("\nFitting linear regression: 1-rho ~ 1/sqrt(kp)...")
    reg_df = fit_linear_regression(df)
    reg_path = TABLES_DIR / "e3_regression_results.csv"
    reg_df.to_csv(reg_path, index=False)
    print(f"Regression results saved: {reg_path}")

    # Summary
    print("\nRegression summary (LOF):")
    lof_reg = reg_df[reg_df["method"] == "LOF"]
    if len(lof_reg) > 0:
        print(f"  Mean R^2: {lof_reg['R2'].mean():.3f}")
        print(f"  Mean slope: {lof_reg['slope'].mean():.3f}")
        print(f"  Mean p_value: {lof_reg['p_value'].mean():.3e}")

    t1 = time.perf_counter()
    elapsed = t1 - t0
    print("\n" + "=" * 60)
    print(f"E3 complete. Total wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
