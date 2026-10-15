"""
plot_E4_threshold.py
--------------------
E4 Visualizations:
  1. Threshold sensitivity: rho vs threshold k% with rho_tilde as horizontal reference
  2. Summary bar chart: mean rho and rho_tilde per method

Reads: results/e4_raw_results.csv
       tables/e4_rho_vs_rhotilde.csv
       tables/e4_threshold_sensitivity.csv
Writes: figures/e4_threshold_sensitivity.pdf
        figures/e4_rho_comparison_bar.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = OUTPUT_DIR / "results"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "figure.dpi": 150,
})

METHOD_COLORS = {
    "IForest": "#e08214",
    "LOF":     "#d73027",
    "COPOD":   "#4dac26",
    "ECOD":    "#1b7837",
    "KNN":     "#c51b7d",
    "AutoEncoder": "#7b3294",
}

CLASS_COLORS = {
    "I":  "#2166ac",
    "II": "#4dac26",
    "III": "#e08214",
    "IV": "#d73027",
}


def load_data() -> tuple:
    raw_path = RESULTS_DIR / "e4_raw_results.csv"
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found. Run exp_E4_rankbased.py first.")
        sys.exit(1)
    df = pd.read_csv(raw_path)

    wilcoxon_path = TABLES_DIR / "e4_rho_vs_rhotilde.csv"
    wilcoxon_df = pd.read_csv(wilcoxon_path) if wilcoxon_path.exists() else pd.DataFrame()

    thresh_path = TABLES_DIR / "e4_threshold_sensitivity.csv"
    thresh_df = pd.read_csv(thresh_path) if thresh_path.exists() else pd.DataFrame()

    return df, wilcoxon_df, thresh_df


def plot_threshold_sensitivity(thresh_df: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    """
    For each dataset in Part B: rho vs threshold_pct, with rho_tilde as horizontal line.
    """
    if len(thresh_df) == 0:
        # Fall back to raw data
        if "part" in df_raw.columns:
            part_b = df_raw[df_raw["part"] == "B"]
        else:
            part_b = df_raw
        thresh_df = (
            part_b
            .groupby(["method", "dataset", "threshold_pct"])
            .agg(mean_rho=("rho", "mean"), mean_rho_tilde=("rho_tilde", "mean"))
            .reset_index()
        )

    if len(thresh_df) == 0:
        print("No Part B data for threshold sensitivity plot.")
        return

    methods = sorted(thresh_df["method"].unique())
    datasets = sorted(thresh_df["dataset"].unique())
    n_ds = len(datasets)
    n_methods = len(methods)

    fig, axes = plt.subplots(
        n_methods, n_ds, figsize=(3.5 * n_ds, 3.5 * n_methods), squeeze=False
    )

    for m_idx, method in enumerate(methods):
        for ds_idx, ds in enumerate(datasets):
            ax = axes[m_idx][ds_idx]
            sub = thresh_df[(thresh_df["method"] == method) & (thresh_df["dataset"] == ds)]

            if len(sub) == 0:
                ax.set_visible(False)
                continue

            sub = sub.sort_values("threshold_pct")
            threshold_pct = sub["threshold_pct"].values * 100  # convert to %

            # rho line (varies with threshold)
            ax.plot(
                threshold_pct, sub["mean_rho"].values,
                color=METHOD_COLORS.get(method, "steelblue"),
                linewidth=2, marker="o", markersize=6,
                label="ρ (Dice)"
            )

            # rho_tilde: constant or near-constant (horizontal reference)
            rho_tilde_mean = sub["mean_rho_tilde"].mean()
            ax.axhline(
                y=rho_tilde_mean,
                color=METHOD_COLORS.get(method, "steelblue"),
                linestyle="--", linewidth=2, alpha=0.8,
                label=f"ρ̃ = {rho_tilde_mean:.2f}"
            )

            # Shade the gap
            y_rho = sub["mean_rho"].values
            x = threshold_pct
            ax.fill_between(
                x, y_rho, rho_tilde_mean,
                where=(y_rho <= rho_tilde_mean),
                alpha=0.15, color=METHOD_COLORS.get(method, "steelblue"),
                label="gap ρ̃−ρ"
            )

            ax.set_ylim(0, 1.08)
            ax.set_xlabel("Threshold k (%)", fontsize=24)
            ax.set_ylabel("Resilience", fontsize=24)
            ax.set_title(f"{method} / {ds}", fontsize=18)
            ax.tick_params(labelsize=22)
            ax.grid(alpha=0.3)

    fig.suptitle(
        "E4 — Threshold Sensitivity: ρ vs k%\n"
        "Dashed line = ρ̃ (threshold-free rank-based resilience)",
        fontsize=24
    )

    legend_elements = [
        Line2D([0], [0], color="gray", linewidth=2, marker="o", markersize=6,
               label="ρ (Dice)"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=2, alpha=0.8,
               label="ρ̃ (rank-based, threshold-free)"),
        mpatches.Patch(alpha=0.3, color="gray", label="gap ρ̃−ρ"),
    ]
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.legend(handles=legend_elements, fontsize=16,
               bbox_to_anchor=(0.5, 0.01), loc="lower center",
               ncol=len(legend_elements), borderaxespad=0, frameon=True)

    out_pdf = FIGURES_DIR / "e4_threshold_sensitivity.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    plt.close(fig)


def plot_rho_comparison_bar(wilcoxon_df: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    """Bar chart: mean rho and rho_tilde per method, with error bars."""
    if len(wilcoxon_df) == 0:
        # Compute from raw
        part_a = df_raw[df_raw.get("part", "A") == "A"] if "part" in df_raw.columns else df_raw
        agg = part_a.groupby("method").agg(
            mean_rho=("rho", "mean"),
            std_rho=("rho", "std"),
            mean_rho_tilde=("rho_tilde", "mean"),
            std_rho_tilde=("rho_tilde", "std"),
        ).reset_index()
        wilcoxon_df = agg

    if len(wilcoxon_df) == 0:
        print("No comparison data for bar chart.")
        return

    # Load method class info
    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        from method_wrappers import METHOD_CLASS_MAP
    except Exception:
        METHOD_CLASS_MAP = {}

    methods = wilcoxon_df["method"].tolist()
    n = len(methods)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, n * 0.7), 5))

    rho_vals = wilcoxon_df.get("mean_rho", wilcoxon_df.get("mean_rho")).values
    rho_tilde_vals = wilcoxon_df.get("mean_rho_tilde", wilcoxon_df.get("mean_rho_tilde")).values

    colors = [CLASS_COLORS.get(METHOD_CLASS_MAP.get(m, "I"), "steelblue") for m in methods]

    bars1 = ax.bar(x - width / 2, rho_vals, width, label="ρ (Dice)", color=colors, alpha=0.7, edgecolor="black")
    bars2 = ax.bar(x + width / 2, rho_tilde_vals, width, label="ρ̃ (rank)", color=colors, alpha=0.45,
                   edgecolor="black", hatch="//")

    # P-value annotations
    if "wilcoxon_p" in wilcoxon_df.columns:
        for i, (m, p_val) in enumerate(zip(methods, wilcoxon_df["wilcoxon_p"].values)):
            if not np.isnan(p_val):
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                ax.text(
                    i, max(rho_vals[i], rho_tilde_vals[i]) + 0.02,
                    sig, ha="center", va="bottom", fontsize=8
                )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Resilience")
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "E4 — Mean ρ and ρ̃ per Method\n"
        "* p<0.05, ** p<0.01, *** p<0.001 (Wilcoxon signed-rank ρ̃ ≥ ρ)"
    )
    ax.legend(fontsize=16)
    ax.grid(axis="y", alpha=0.3)

    # Class dividers
    prev_class = None
    for i, m in enumerate(methods):
        cls = METHOD_CLASS_MAP.get(m, "?")
        if cls != prev_class and prev_class is not None:
            ax.axvline(x=i - 0.5, color="gray", linestyle=":", alpha=0.5)
            ax.text(i - 0.5, 1.12, f"Class {cls}", ha="center", fontsize=8, color="gray")
        prev_class = cls

    plt.tight_layout()
    out_pdf = FIGURES_DIR / "e4_rho_comparison_bar.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    plt.close(fig)


def main():
    print("Plotting E4 threshold sensitivity and rho comparison...")
    df, wilcoxon_df, thresh_df = load_data()
    plot_threshold_sensitivity(thresh_df, df)
    plot_rho_comparison_bar(wilcoxon_df, df)
    print("Done.")


if __name__ == "__main__":
    main()
