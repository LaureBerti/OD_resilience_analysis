"""
plot_E3_kdep.py
---------------
E3 Visualization: 1-rho vs 1/sqrt(kp) scatter with fitted regression lines.

Reads: results/e3_raw_results.csv
       tables/e3_regression_results.csv
Writes: figures/e3_lof_kdep_scatter.pdf
        figures/e3_knn_kdep_scatter.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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

P_MARKERS = {0.05: "o", 0.10: "s", 0.20: "^"}
P_COLORS = {0.05: "#1b7837", 0.10: "#762a83", 0.20: "#e08214"}


def load_data() -> tuple:
    raw_path = RESULTS_DIR / "e3_raw_results.csv"
    reg_path = TABLES_DIR / "e3_regression_results.csv"
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found. Run exp_E3_lof_kdep.py first.")
        sys.exit(1)
    df = pd.read_csv(raw_path)
    df["one_minus_rho"] = 1.0 - df["rho"]
    reg_df = pd.read_csv(reg_path) if reg_path.exists() else pd.DataFrame()
    return df, reg_df


def plot_kdep_scatter(df: pd.DataFrame, reg_df: pd.DataFrame, method: str) -> None:
    df_m = df[df["method"] == method].dropna(subset=["one_minus_rho", "kp_inv_sqrt"])
    if len(df_m) == 0:
        print(f"No data for method={method}")
        return

    datasets = sorted(df_m["dataset"].unique())
    n_ds = len(datasets)
    ncols = min(4, n_ds)
    nrows = int(np.ceil(n_ds / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for ds_idx, ds in enumerate(datasets):
        row_idx = ds_idx // ncols
        col_idx = ds_idx % ncols
        ax = axes[row_idx][col_idx]

        ds_df = df_m[df_m["dataset"] == ds]

        for p_val, marker in P_MARKERS.items():
            sub = ds_df[ds_df["p"] == p_val]
            if len(sub) == 0:
                continue
            # Mean over reps per k
            mean_by_k = sub.groupby("k").agg(
                mean_1mr=("one_minus_rho", "mean"),
                kp_inv_sqrt=("kp_inv_sqrt", "first"),
            ).reset_index()

            ax.scatter(
                mean_by_k["kp_inv_sqrt"], mean_by_k["mean_1mr"],
                marker=marker, color=P_COLORS.get(p_val, "gray"),
                s=50, zorder=5, label=f"p={p_val:.0%}"
            )

        # Overall regression line
        if len(reg_df) > 0:
            reg_row = reg_df[(reg_df["method"] == method) & (reg_df["dataset"] == ds)]
            if len(reg_row) > 0:
                # Use mean slope/intercept across p values
                slope = reg_row["slope"].mean()
                intercept = reg_row["intercept"].mean()
                r2_mean = reg_row["R2"].mean()
                x_range = np.linspace(ds_df["kp_inv_sqrt"].min(), ds_df["kp_inv_sqrt"].max(), 50)
                y_fit = slope * x_range + intercept
                ax.plot(
                    x_range, y_fit, "k--", linewidth=1.5,
                    label=f"fit (R²={r2_mean:.2f})"
                )
        else:
            # Fit directly
            x = ds_df["kp_inv_sqrt"].values
            y = ds_df["one_minus_rho"].values
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() >= 3:
                slope, intercept, r_value, p_value, _ = stats.linregress(x[valid], y[valid])
                r2 = r_value ** 2
                x_range = np.linspace(x[valid].min(), x[valid].max(), 50)
                y_fit = slope * x_range + intercept
                ax.plot(x_range, y_fit, "k--", linewidth=1.5, label=f"fit (R²={r2:.2f})")

        ax.set_title(ds, fontsize=18)
        ax.set_xlabel("1/√(kp)", fontsize=24)
        ax.set_ylabel("1 − ρ", fontsize=24)
        ax.tick_params(labelsize=22)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    # Hide unused axes
    for ds_idx in range(n_ds, nrows * ncols):
        axes[ds_idx // ncols][ds_idx % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()

    fig.suptitle(
        f"E3 — {method}: 1−ρ vs 1/√(kp)\n"
        f"(k∈{{3,5,10,15,20}}, p∈{{5%,10%,20%}}, 30 reps)",
        fontsize=24
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    fig.legend(handles, labels, fontsize=16,
               bbox_to_anchor=(0.5, 0.01), loc="lower center",
               ncol=len(handles), borderaxespad=0, frameon=True)

    out_pdf = FIGURES_DIR / f"e3_{method.lower()}_kdep_scatter.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    plt.close(fig)


def plot_comparison_lof_knn(df: pd.DataFrame) -> None:
    """Single figure comparing LOF and KNN regression slopes."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, method in zip(axes, ["LOF", "KNN"]):
        sub = df[df["method"] == method].dropna(subset=["one_minus_rho", "kp_inv_sqrt"])
        if len(sub) == 0:
            ax.set_title(f"{method}: no data")
            continue

        # Scatter: pool all datasets, colour by dataset
        datasets = sorted(sub["dataset"].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

        for ds, color in zip(datasets, colors):
            ds_sub = sub[sub["dataset"] == ds]
            mean_by_kp = ds_sub.groupby("kp_inv_sqrt")["one_minus_rho"].mean().reset_index()
            ax.scatter(
                mean_by_kp["kp_inv_sqrt"], mean_by_kp["one_minus_rho"],
                color=color, s=20, alpha=0.7, label=ds
            )

        # Pool regression line
        x = sub["kp_inv_sqrt"].values
        y = sub["one_minus_rho"].values
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() >= 3:
            slope, intercept, r_value, p_value, _ = stats.linregress(x[valid], y[valid])
            r2 = r_value ** 2
            x_range = np.linspace(x[valid].min(), x[valid].max(), 50)
            y_fit = slope * x_range + intercept
            ax.plot(
                x_range, y_fit, "k-", linewidth=2,
                label=f"pool fit: slope={slope:.3f}, R²={r2:.2f}"
            )

        ax.set_xlabel("1/√(kp)")
        ax.set_ylabel("1 − ρ")
        ax.set_title(f"{method}: 1−ρ vs 1/√(kp)")
        ax.legend(fontsize=16, ncol=2)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle("E3 — LOF vs KNN k-Dependence (Theorem 1 validation)", fontsize=12)
    plt.tight_layout()

    out_pdf = FIGURES_DIR / "e3_lof_knn_comparison.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    plt.close(fig)


def main():
    print("Plotting E3 k-dependence figures...")
    df, reg_df = load_data()
    plot_kdep_scatter(df, reg_df, "LOF")
    plot_kdep_scatter(df, reg_df, "KNN")
    plot_comparison_lof_knn(df)
    print("Done.")


if __name__ == "__main__":
    main()
