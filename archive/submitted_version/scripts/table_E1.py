"""
table_E1.py — Generate LaTeX table for E1 results from e1_mean_rho_by_class.csv
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

def fmt(mean, std) -> str:
    """Format mean±std, handling NaN."""
    if np.isnan(mean):
        return "---"
    return f"{mean:.2f}\\!\\!\\!$\\pm$\\!\\!\\!{std:.2f}"

def fmt_bold(mean, std, is_best, is_second) -> str:
    s = f"{mean:.2f}$\\pm${std:.2f}"
    if is_best:
        return f"\\textbf{{{s}}}"
    if is_second:
        return f"\\underline{{{s}}}"
    return s

def main():
    raw_path = RESULTS_DIR / "e1_raw_results.csv"
    df = pd.read_csv(raw_path)

    # Method display names and column order
    METHOD_COLS = [
        ("ThreeSigma", "3Sig"),
        ("BoxPlot",    "Box"),
        ("MAD",        "MAD"),
        ("ChiSquare",  "$\\chi^2$"),
        ("COPOD",      "COPOD"),
        ("ECOD",       "ECOD"),
        ("HBOS",       "HBOS"),
        ("IForest",    "IForest"),
        ("LOF",        "LOF"),
        ("KNN",        "KNN"),
        ("OCSVM",      "OCSVM"),
        ("Mahalanobis","Maha"),
        ("KMeans",     "KMeans"),
        ("AE",         "AE"),
    ]

    methods = [m for m, _ in METHOD_COLS]
    display = [d for _, d in METHOD_COLS]

    tiers = ["T1", "T2", "T3"]
    tier_labels = {"T1": "T1", "T2": "T2", "T3": "T3"}

    # Compute mean±std per (tier, method)
    agg = (
        df.groupby(["tier", "method"])["rho"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    # Also compute "All"
    all_agg = (
        df.groupby("method")["rho"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    all_agg["tier"] = "All"

    agg = pd.concat([agg, all_agg], ignore_index=True)

    # Build table rows
    rows = []
    for tier in ["T1", "T2", "T3", "All"]:
        row_data = []
        sub = agg[agg["tier"] == tier]
        means = {}
        stds = {}
        for m in methods:
            r = sub[sub["method"] == m]
            if len(r) == 0:
                means[m] = np.nan
                stds[m] = np.nan
            else:
                means[m] = r["mean"].iloc[0]
                stds[m] = r["std"].iloc[0]

        # Find best and second-best
        valid_means = {m: v for m, v in means.items() if not np.isnan(v)}
        sorted_means = sorted(valid_means.items(), key=lambda x: x[1], reverse=True)
        best = sorted_means[0][0] if len(sorted_means) > 0 else None
        second = sorted_means[1][0] if len(sorted_means) > 1 else None

        row_str = f"    {tier if tier != 'All' else '\\textbf{All}'}"
        for m in methods:
            mean_v = means[m]
            std_v = stds[m]
            if np.isnan(mean_v):
                cell = "---"
            else:
                s = f"{mean_v:.2f}$\\pm${std_v:.2f}"
                if m == best:
                    cell = f"\\textbf{{{s}}}"
                elif m == second:
                    cell = f"\\underline{{{s}}}"
                else:
                    cell = s
            row_str += f" & {cell}"
        row_str += " \\\\"
        rows.append(row_str)

    # Print Kruskal-Wallis stats
    kw_path = TABLES_DIR / "e1_kruskal_wallis.csv"
    kw = pd.read_csv(kw_path)
    H = kw["H"].iloc[0]
    p = kw["p"].iloc[0]
    print(f"% Kruskal-Wallis H={H:.1f}, p={p:.2e}")
    print()

    # Print table
    print("% --- E1 Main Table (paste into experiments_revised.tex) ---")
    print("% Replace [TODO: replace with Python results] in caption with:")
    print(f"% Kruskal-Wallis $H={H:.1f}$, $p<0.001$.")
    print()
    print("% Row values (T1, T2, T3, All):")
    for row in rows:
        print(row)

    # Save
    out_path = TABLES_DIR / "e1_latex_table.txt"
    with open(out_path, "w") as f:
        f.write(f"% Kruskal-Wallis H={H:.1f}, p={p:.2e}\n\n")
        for row in rows:
            f.write(row + "\n")
    print(f"\nSaved: {out_path}")

    # Also print class-level summary
    print("\n% Class-level mean rho (All tiers, All datasets):")
    class_agg = df.groupby("method_class")["rho"].agg(mean="mean", std="std")
    for cls in ["I", "II", "III", "IV"]:
        if cls in class_agg.index:
            row = class_agg.loc[cls]
            print(f"%   Class {cls}: {row['mean']:.3f} ± {row['std']:.3f}")


if __name__ == "__main__":
    main()
