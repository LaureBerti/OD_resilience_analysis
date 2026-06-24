"""
plot_E2_convergence.py
----------------------
E2 Visualizations:
  1. Log-log plot of 1-rho vs n=pN with fitted power-law lines (slope=alpha_hat)
  2. IForest phase-transition plot: rho vs n with vertical line at n=psi=256

Reads: results/e2_raw_results.csv
       tables/e2_alpha_estimates.csv
Writes: figures/e2_loglog_convergence.pdf
        figures/e2_iforest_phase_transition.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

PSI = 256

METHOD_COLORS = {
    "MAD":     "#2166ac",
    "ECOD":    "#4dac26",
    "LOF":     "#d73027",
    "IForest": "#e08214",
}
METHOD_LABELS = {
    "MAD":     "MAD (Class I, expected α≈0.5)",
    "ECOD":    "ECOD (Class II, expected α>0.5)",
    "LOF":     "LOF (Class IV, expected α<0.3)",
    "IForest": "IForest (Class III)",
}


def load_data() -> tuple:
    raw_path = RESULTS_DIR / "e2_raw_results.csv"
    alpha_path = TABLES_DIR / "e2_alpha_estimates.csv"
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found. Run exp_E2_convergence.py first.")
        sys.exit(1)
    df = pd.read_csv(raw_path)
    alpha_df = pd.read_csv(alpha_path) if alpha_path.exists() else pd.DataFrame()
    return df, alpha_df


def plot_loglog_convergence(df: pd.DataFrame, alpha_df: pd.DataFrame) -> None:
    """Log-log plot of mean(1-rho) vs n for Part A methods."""
    df_a = df[df.get("part", df["method"].isin(["MAD", "ECOD", "LOF"])) == "A"].copy()
    if len(df_a) == 0:
        # Fall back: use Part A methods regardless of 'part' column
        df_a = df[df["method"].isin(["MAD", "ECOD", "LOF"])].copy()

    methods = ["MAD", "ECOD", "LOF"]
    fig, ax = plt.subplots(figsize=(5, 3.5))

    for method in methods:
        sub = df_a[df_a["method"] == method].dropna(subset=["rho", "n_sample"])
        if len(sub) == 0:
            continue

        # Mean (1-rho) per n
        mean_by_n = sub.groupby("n_sample")["rho"].mean().reset_index()
        mean_by_n["one_minus_rho"] = 1.0 - mean_by_n["rho"]
        mean_by_n = mean_by_n[mean_by_n["one_minus_rho"] > 0]
        mean_by_n = mean_by_n.sort_values("n_sample")

        ax.scatter(
            mean_by_n["n_sample"], mean_by_n["one_minus_rho"],
            color=METHOD_COLORS.get(method, "gray"),
            s=40, zorder=5, label="_nolegend_"
        )
        ax.plot(
            mean_by_n["n_sample"], mean_by_n["one_minus_rho"],
            color=METHOD_COLORS.get(method, "gray"),
            linewidth=1.5, alpha=0.5
        )

        # Fitted line from alpha_df
        if len(alpha_df) > 0:
            method_alphas = alpha_df[alpha_df["method"] == method]
            if len(method_alphas) > 0:
                alpha_mean = method_alphas["alpha_hat"].mean()
                # Fit intercept from data
                log_n = np.log(mean_by_n["n_sample"].values)
                log_res = np.log(mean_by_n["one_minus_rho"].values)
                valid = np.isfinite(log_n) & np.isfinite(log_res)
                if valid.sum() >= 2:
                    log_A = np.mean(log_res[valid] - alpha_mean * log_n[valid])
                    n_fit = np.logspace(
                        np.log10(mean_by_n["n_sample"].min()),
                        np.log10(mean_by_n["n_sample"].max()),
                        50
                    )
                    y_fit = np.exp(log_A) * n_fit ** alpha_mean
                    ax.plot(
                        n_fit, y_fit,
                        color=METHOD_COLORS.get(method, "gray"),
                        linestyle="--", linewidth=2,
                        label=f"{METHOD_LABELS[method]}\nα̂={alpha_mean:.2f}"
                    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sample size n = pN")
    ax.set_ylabel("Mean (1 − ρ)")
    ax.set_title("E2 — Convergence Rate: 1−ρ vs n (log-log)")
    ax.legend(fontsize=16, bbox_to_anchor=(1.01, 1.0), loc="upper left",
              borderaxespad=0, frameon=True)
    ax.grid(True, which="both", alpha=0.3)

    # Reference slope annotations
    n_ref = np.array([100, 10000])
    ax.plot(n_ref, 2 * n_ref ** (-0.5), "k:", alpha=0.5, linewidth=1)
    ax.text(10000, 2 * 10000 ** (-0.5) * 1.5, "slope −0.5", fontsize=8, color="gray")

    out_pdf = FIGURES_DIR / "e2_loglog_convergence.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    plt.close(fig)


def plot_iforest_phase_transition(df: pd.DataFrame) -> None:
    """IForest rho vs n with vertical line at n=psi."""
    df_b = df[df["method"] == "IForest"].copy()
    if len(df_b) == 0:
        print("No IForest data for phase-transition plot.")
        return

    datasets = df_b["dataset"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

    fig, ax = plt.subplots(figsize=(5, 3.5))

    for ds, color in zip(datasets, colors):
        sub = df_b[df_b["dataset"] == ds].dropna(subset=["rho", "n_sample"])
        if len(sub) == 0:
            continue
        mean_by_n = sub.groupby("n_sample")["rho"].agg(["mean", "std"]).reset_index()
        mean_by_n = mean_by_n.sort_values("n_sample")

        ax.plot(
            mean_by_n["n_sample"], mean_by_n["mean"],
            color=color, linewidth=2, marker="o", markersize=5,
            label=ds
        )
        ax.fill_between(
            mean_by_n["n_sample"],
            mean_by_n["mean"] - mean_by_n["std"],
            mean_by_n["mean"] + mean_by_n["std"],
            color=color, alpha=0.15
        )

    # Phase-transition vertical line
    ax.axvline(x=PSI, color="black", linestyle="--", linewidth=2, alpha=0.8)
    ax.text(
        PSI * 1.05, 0.15,
        f"ψ = {PSI}\n(phase transition)",
        fontsize=9, color="black", va="bottom"
    )

    ax.set_xscale("log")
    ax.set_xlabel("Sample size n = pN")
    ax.set_ylabel("Mean Dice Resilience ρ")
    ax.set_ylim(-0.05, 1.08)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_title(
        f"E2 — IForest Phase Transition at n = ψ = {PSI}\n"
        "(max_samples=256, n_estimators=100)"
    )
    ax.legend(fontsize=16, bbox_to_anchor=(1.01, 1.0), loc="upper left",
              borderaxespad=0, frameon=True, ncol=1)
    ax.grid(True, alpha=0.3)

    out_pdf = FIGURES_DIR / "e2_iforest_phase_transition.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    plt.close(fig)


def main():
    print("Plotting E2 convergence figures...")
    df, alpha_df = load_data()
    plot_loglog_convergence(df, alpha_df)
    plot_iforest_phase_transition(df)
    print("Done.")


if __name__ == "__main__":
    main()
