"""
table_E234.py — Generate LaTeX snippets for E2, E3, E4 tables once results are ready.
Run after exp_E2, exp_E3, exp_E4 complete.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = OUTPUT_DIR / "results"
TABLES_DIR = OUTPUT_DIR / "tables"


def table_e2():
    alpha_path = TABLES_DIR / "e2_alpha_estimates.csv"
    if not alpha_path.exists():
        print("E2 alpha estimates not yet available.")
        return
    df = pd.read_csv(alpha_path)
    print("\n=== E2: Alpha estimates ===")
    for method in ["MAD", "ECOD", "LOF"]:
        sub = df[df["method"] == method]
        if len(sub) == 0:
            print(f"  {method}: no data")
            continue
        mean_alpha = sub["alpha_hat"].mean()
        mean_r2 = sub["R2"].mean()
        # 95% CI across datasets (using std)
        std_alpha = sub["alpha_hat"].std()
        ci_lo = mean_alpha - 1.96 * std_alpha / np.sqrt(len(sub))
        ci_hi = mean_alpha + 1.96 * std_alpha / np.sqrt(len(sub))
        print(f"  {method}: alpha={mean_alpha:.3f} [{ci_lo:.3f},{ci_hi:.3f}]  R2={mean_r2:.3f}")
        print(f"    LaTeX: {method} & {mean_alpha:.2f} & [{ci_lo:.2f}, {ci_hi:.2f}] & {mean_r2:.3f} \\\\")


def table_e3():
    reg_path = TABLES_DIR / "e3_regression_results.csv"
    if not reg_path.exists():
        print("E3 regression results not yet available.")
        return
    df = pd.read_csv(reg_path)
    lof_df = df[df["method"] == "LOF"]
    print("\n=== E3: LOF regression by dataset ===")

    # Dataset name mapping (our substitutions → paper names)
    ds_name_map = {
        "annthyroid": "Annthyroid",
        "thyroid":    "Thyroid",
        "letter":     "Letter",
        "optdigits":  "Optdigits",
        "smtp":       "SMTP (KDDCup99$^*$)",
        "shuttle":    "Shuttle",
        "skin":       "Skin (ForestCov$^*$)",
        "glass":      "Glass",
    }
    for _, row in lof_df.groupby(["dataset", "p"]).agg(
        slope=("slope", "mean"), R2=("R2", "mean"), p_value=("p_value", "mean"), n_obs=("n_obs", "mean")
    ).groupby("dataset").agg(
        slope=("slope", "mean"), R2=("R2", "mean"), p_value=("p_value", "mean"), n_obs=("n_obs", "mean")
    ).reset_index().iterrows():
        ds = ds_name_map.get(row["dataset"], row["dataset"])
        p_str = f"${row['p_value']:.2e}$" if row["p_value"] > 0 else "$<$0.001"
        print(f"  {ds} & {row['slope']:.3f} & {row['R2']:.3f} & {p_str} & {int(row['n_obs'])} \\\\")

    mean_r2 = lof_df["R2"].mean()
    mean_slope = lof_df["slope"].mean()
    print(f"\n  Mean R2={mean_r2:.3f}, Mean slope={mean_slope:.3f}")
    print(f"  LaTeX mean row: \\textbf{{Mean}} & {mean_slope:.3f} & {mean_r2:.3f} & — & — \\\\")


def table_e4():
    wilcoxon_path = TABLES_DIR / "e4_rho_vs_rhotilde.csv"
    if not wilcoxon_path.exists():
        print("E4 Wilcoxon results not yet available.")
        return
    df = pd.read_csv(wilcoxon_path)

    METHOD_ORDER = [
        "ThreeSigma", "BoxPlot", "MAD", "ChiSquare",
        "COPOD", "ECOD", "HBOS",
        "IForest",
        "LOF", "KNN", "OCSVM", "Mahalanobis", "KMeans",
    ]
    METHOD_DISPLAY = {
        "ThreeSigma": "3Sigma", "BoxPlot": "BoxPlot", "MAD": "MAD",
        "ChiSquare": "ChiSquare", "COPOD": "COPOD", "ECOD": "ECOD",
        "HBOS": "HBOS", "IForest": "IForest", "LOF": "LOF", "KNN": "KNN",
        "OCSVM": "OCSVM", "Mahalanobis": "Mahalanobis", "KMeans": "KMeans",
    }

    print("\n=== E4: Wilcoxon test (rho_tilde >= rho) ===")
    for method in METHOD_ORDER:
        r = df[df["method"] == method]
        if len(r) == 0:
            p_str = "---"
            print(f"  {METHOD_DISPLAY.get(method, method)} & --- & --- & --- & --- \\\\")
            continue
        row = r.iloc[0]
        p = row["wilcoxon_p"]
        if np.isnan(p):
            p_str = "n/a"
        elif p < 0.001:
            p_str = "$<$0.001"
        else:
            p_str = f"{p:.3f}"
        diff = row["mean_rho_tilde"] - row["mean_rho"]
        print(f"  {METHOD_DISPLAY.get(method, method)} & {row['mean_rho']:.3f} & {row['mean_rho_tilde']:.3f} & {diff:+.3f} & {p_str} \\\\")

    # Threshold sensitivity summary
    thresh_path = TABLES_DIR / "e4_threshold_sensitivity.csv"
    if thresh_path.exists():
        thresh = pd.read_csv(thresh_path)
        # LOF variation across thresholds
        lof_t = thresh[thresh["method"] == "LOF"]
        if len(lof_t) > 0:
            max_var = lof_t.groupby("dataset")["mean_rho"].agg(lambda x: x.max() - x.min()).max()
            print(f"\n  LOF max rho variation across thresholds: {max_var:.2f} pp")
            print(f"  (for E4 interpretation TODO: 'varying by up to {max_var:.0%} percentage points')")


def main():
    table_e2()
    table_e3()
    table_e4()
    print("\nDone.")


if __name__ == "__main__":
    main()
