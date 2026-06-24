"""
plot_E1_violin.py
-----------------
E1 Visualization: Violin plot of rho by method class.

Reads: results/e1_raw_results.csv
Writes: figures/e1_violin_by_class.pdf
        figures/e1_violin_by_class.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 16,
    "figure.dpi": 150,
})

CLASS_COLORS = {
    "I":   "#2166ac",   # blue
    "II":  "#4dac26",   # green
    "III": "#e08214",   # orange
    "IV":  "#d73027",   # red
}

CLASS_LABELS = {
    "I":   "Class I\n(3Sigma, BoxPlot,\nMAD, ChiSq)",
    "II":  "Class II\n(COPOD, ECOD,\nHBOS)",
    "III": "Class III\n(IForest)",
    "IV":  "Class IV\n(LOF, KNN,\nOCSVM, ...)",
}


def load_data() -> pd.DataFrame:
    path = RESULTS_DIR / "e1_raw_results.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run exp_E1_class_partition.py first.")
        sys.exit(1)
    df = pd.read_csv(path)
    return df


def plot_violin(df: pd.DataFrame) -> None:
    classes = ["I", "II", "III", "IV"]
    data_by_class = {c: df[df["method_class"] == c]["rho"].dropna().values for c in classes}

    fig, ax = plt.subplots(figsize=(8, 5))

    positions = [1, 2, 3, 4]
    vp = ax.violinplot(
        [data_by_class[c] for c in classes],
        positions=positions,
        showmedians=True,
        showextrema=True,
        widths=0.7,
    )

    # Color each violin
    for i, (body, cls) in enumerate(zip(vp["bodies"], classes)):
        body.set_facecolor(CLASS_COLORS[cls])
        body.set_edgecolor("black")
        body.set_alpha(0.7)

    for part in ("cbars", "cmins", "cmaxes", "cmedians"):
        if part in vp:
            vp[part].set_color("black")
            vp[part].set_linewidth(1.5)

    # Add individual jitter points
    rng = np.random.default_rng(0)
    for i, (pos, cls) in enumerate(zip(positions, classes)):
        vals = data_by_class[cls]
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(
            pos + jitter, vals,
            color=CLASS_COLORS[cls], alpha=0.2, s=3, zorder=2
        )
        # Add mean marker
        ax.scatter(pos, np.nanmean(vals), marker="D", color="white",
                   edgecolor="black", s=50, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels([CLASS_LABELS[c] for c in classes], fontsize=9,
                       linespacing=1.3)
    ax.set_ylabel("Dice Resilience ρ", fontsize=12)
    ax.set_ylim(-0.02, 1.10)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title(
        "E1 — Dice Resilience by Method Class\n"
        "(p=10%, 50 reps, Tier 1–3 datasets, N=16)",
        fontsize=12
    )
    ax.grid(axis="y", alpha=0.3)

    # Kruskal-Wallis annotation — top-left to avoid overlap with violins
    from scipy import stats
    groups = [data_by_class[c] for c in classes]
    h_stat, p_val = stats.kruskal(*groups)
    ax.text(
        0.02, 0.98,
        f"Kruskal-Wallis H={h_stat:.1f}, p={p_val:.2e}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9, color="gray"
    )

    plt.tight_layout()

    out_pdf = FIGURES_DIR / "e1_violin_by_class.pdf"
    out_png = FIGURES_DIR / "e1_violin_by_class.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    plt.close(fig)


def plot_violin_by_tier(df: pd.DataFrame) -> None:
    """Additional figure: violin split by tier."""
    tiers = sorted(df["tier"].unique())
    classes = ["I", "II", "III", "IV"]
    n_tiers = len(tiers)

    fig, axes = plt.subplots(1, n_tiers, figsize=(4 * n_tiers, 5), sharey=True)
    if n_tiers == 1:
        axes = [axes]

    for ax, tier in zip(axes, tiers):
        tier_df = df[df["tier"] == tier]
        data_by_class = {c: tier_df[tier_df["method_class"] == c]["rho"].dropna().values for c in classes}

        positions = [1, 2, 3, 4]
        for pos, cls in zip(positions, classes):
            if len(data_by_class[cls]) == 0:
                continue
            vp = ax.violinplot(
                [data_by_class[cls]], positions=[pos],
                showmedians=True, widths=0.6,
            )
            for body in vp["bodies"]:
                body.set_facecolor(CLASS_COLORS[cls])
                body.set_alpha(0.7)
            for part in ("cbars", "cmins", "cmaxes", "cmedians"):
                if part in vp:
                    vp[part].set_color("black")

        ax.set_xticks(positions)
        ax.set_xticklabels(["I", "II", "III", "IV"])
        ax.set_title(f"Tier {tier[-1]}", fontsize=11)
        ax.set_xlabel("Method Class")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Dice Resilience ρ")
    fig.suptitle("E1 — Resilience by Class and Dataset Tier", fontsize=12)
    plt.tight_layout()

    out_pdf = FIGURES_DIR / "e1_violin_by_tier.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    plt.close(fig)


def main():
    print("Plotting E1 violin plots...")
    df = load_data()
    plot_violin(df)
    if "tier" in df.columns:
        plot_violin_by_tier(df)
    print("Done.")


if __name__ == "__main__":
    main()
