"""
exp_E6_stratified.py
--------------------
E6 — Stratified vs SRSWOR Sampling: Monotone Improvement for Classes I and II

Hypothesis H6:
  Under proportionate stratified SRSWOR (stratified by outlier label), the
  mean Dice resilience rho_stratified >= rho_SRSWOR for Class I and Class II
  methods; no monotone improvement is expected for LOF (Class IV negative control).
  Formally: Delta_rho = rho_stratified - rho_SRSWOR > 0 for Classes I/II
  (Wilcoxon signed-rank p < 0.05).

Setup:
  - Methods: Class I (ThreeSigma, BoxPlot, MAD, ChiSquare),
             Class II (COPOD, ECOD, HBOS),
             LOF (Class IV negative control)
  - Datasets: Tier 2 + Tier 3 (N >= 1000); 12 datasets
  - Sampling fractions: p in {5%, 10%, 20%}
  - Replications: 30 (Tier 2), 10 (Tier 3)
  - Two sampling arms per (dataset, method, p, rep):
      SRSWOR: np.random.choice(N, n, replace=False)
      Stratified: proportionate SRSWOR within each stratum {outlier, inlier}
  - Stratification variable: ground-truth outlier label y (two strata: y=0, y=1)
    DISCLOSURE: using ground-truth labels for stratification is an optimistic
    advantage not available in practice; see proxy alternative using k-means
    strata (K=5) documented in _stratify_by_kmeans() below.

Output files:
  results/e6_raw_results.csv
  tables/e6_delta_rho.csv
  tables/e6_wilcoxon.csv
  figures/e6_delta_rho_boxplot.pdf
  timing_per_condition.csv (appended)
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation")

# Preflight: fail fast if pyod is missing so the error is obvious rather than
# appearing silently per-method mid-run.
try:
    import pyod  # noqa: F401
except ModuleNotFoundError:
    sys.exit(
        "ERROR: 'pyod' is not installed in the current Python environment.\n"
        "Activate the project venv first:\n"
        "  source /Users/laureberti/Projects/Paper_Improve/Outlier_resilience/.venv/bin/activate\n"
        "Then re-run:  python exp_E6_stratified.py"
    )

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = OUTPUT_DIR / "results"
TABLES_DIR  = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))

from data_loader import (
    TIER2_DATASETS, TIER3_DATASETS, load_multiple, DATASET_TIERS
)
from method_wrappers import make_detector, METHOD_CLASS_MAP
from resilience_metrics import compute_resilience_pair

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
P_VALUES    = [0.05, 0.10, 0.20]
N_REPS_T2   = 30   # Tier 2 datasets
N_REPS_T3   = 10   # Tier 3 datasets (runtime limit)
CONTAMINATION = 0.10
BASE_SEED   = 42

# Focus methods: Classes I/II for monotonicity claim; LOF as negative control.
E6_METHODS = [
    "ThreeSigma", "BoxPlot", "MAD", "ChiSquare",   # Class I
    "COPOD", "ECOD", "HBOS",                        # Class II
    "LOF",                                           # Class IV negative control
]

E6_DATASETS = TIER2_DATASETS + TIER3_DATASETS  # N >= 1000

# Proxy strata K for k-means alternative (used alongside label-based strata)
KMEANS_STRATA_K = 5


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _stratify_by_label(
    N: int,
    y: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Proportionate stratified SRSWOR using ground-truth outlier label.

    Strata: y=0 (inlier), y=1 (outlier).
    Allocation: n_h = round(n * N_h / N) with total = n (Neyman-adjusted).

    DISCLOSURE: uses ground-truth labels — optimistic advantage not available
    in practice without label knowledge. See _stratify_by_kmeans() for the
    label-free proxy.

    Returns sorted index array of length n.
    """
    idx_in  = np.where(y == 0)[0]
    idx_out = np.where(y == 1)[0]

    N_in  = len(idx_in)
    N_out = len(idx_out)

    if N_out == 0 or N_in == 0:
        # Degenerate: fall back to SRSWOR
        return np.sort(rng.choice(N, size=n, replace=False))

    # Proportionate allocation
    n_out = max(1, round(n * N_out / N))
    n_in  = n - n_out

    # Guard against requesting more than stratum size
    n_out = min(n_out, N_out)
    n_in  = min(n_in,  N_in)
    # Re-adjust total if clamped
    if n_in + n_out < n:
        remainder = n - n_in - n_out
        if N_in - n_in >= remainder:
            n_in += remainder
        else:
            n_out += remainder

    sampled_in  = rng.choice(idx_in,  size=n_in,  replace=False)
    sampled_out = rng.choice(idx_out, size=n_out, replace=False)

    return np.sort(np.concatenate([sampled_in, sampled_out]))


def _stratify_by_kmeans(
    X: np.ndarray,
    n: int,
    rng: np.random.Generator,
    K: int = KMEANS_STRATA_K,
    seed: int = BASE_SEED,
) -> np.ndarray:
    """
    Proportionate stratified SRSWOR using k-means cluster assignment as proxy strata.

    This is the label-free alternative: it does not use ground-truth y.
    Strata = k-means cluster labels on standardized X.
    Allocation: proportionate to cluster size.

    Returns sorted index array of length n.
    """
    N = len(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=min(K, N // 2), random_state=seed, n_init=5)
    cluster_labels = km.fit_predict(X_scaled)

    indices_collected = []
    for h in range(km.n_clusters):
        idx_h = np.where(cluster_labels == h)[0]
        N_h   = len(idx_h)
        n_h   = max(1, round(n * N_h / N))
        n_h   = min(n_h, N_h)
        sampled = rng.choice(idx_h, size=n_h, replace=False)
        indices_collected.append(sampled)

    all_idx = np.concatenate(indices_collected)
    # Adjust to exactly n if rounding caused mismatch
    if len(all_idx) > n:
        all_idx = rng.choice(all_idx, size=n, replace=False)
    elif len(all_idx) < n:
        remaining = np.setdiff1d(np.arange(N), all_idx)
        extra = rng.choice(remaining, size=n - len(all_idx), replace=False)
        all_idx = np.concatenate([all_idx, extra])

    return np.sort(all_idx)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _run_arm(
    X: np.ndarray,
    det_full_scores: np.ndarray,
    idx_sample: np.ndarray,
    method_name: str,
    contamination: float,
    seed: int,
) -> dict:
    """
    Fit method on subsample, return rho and rho_tilde vs. full-data restriction.
    """
    X_sample = X[idx_sample]
    scores_full_restricted = det_full_scores[idx_sample]

    det_s = make_detector(method_name, contamination=contamination, random_state=seed)
    det_s.fit(X_sample)
    scores_sample = det_s.decision_function(X_sample)

    return compute_resilience_pair(scores_sample, scores_full_restricted, contamination)


def run_single_condition(
    X: np.ndarray,
    y: np.ndarray,
    method_name: str,
    p: float,
    rep: int,
    contamination: float = CONTAMINATION,
) -> dict:
    """
    Run one (method, p, rep) condition, computing both SRSWOR and stratified arms.
    The full-dataset fit is shared between both arms to save compute.

    Returns a dict with fields for both arms.
    """
    rng  = np.random.default_rng(BASE_SEED + rep)
    seed = BASE_SEED + rep
    N    = len(X)
    n    = max(4, int(np.floor(p * N)))

    row = {
        "method": method_name,
        "p": p,
        "rep": rep,
        "n_full": N,
        "n_sample": n,
        # SRSWOR arm
        "rho_srs":        np.nan,
        "rho_tilde_srs":  np.nan,
        # Stratified (label-based) arm
        "rho_str":        np.nan,
        "rho_tilde_str":  np.nan,
        # Stratified (k-means proxy) arm
        "rho_str_km":     np.nan,
        "rho_tilde_str_km": np.nan,
        # Diagnostics
        "time_total_s":   np.nan,
        "error":          None,
    }

    t0 = time.perf_counter()
    try:
        # Fit on full dataset once; reuse scores for both arms
        det_full = make_detector(method_name, contamination=contamination, random_state=seed)
        det_full.fit(X)
        scores_full = det_full.decision_function(X)

        # --- SRSWOR arm ---
        idx_srs = np.sort(rng.choice(N, size=n, replace=False))
        m_srs   = _run_arm(X, scores_full, idx_srs, method_name, contamination, seed)
        row["rho_srs"]       = m_srs["rho"]
        row["rho_tilde_srs"] = m_srs["rho_tilde"]

        # --- Stratified (label-based) arm ---
        idx_str = _stratify_by_label(N, y, n, rng)
        m_str   = _run_arm(X, scores_full, idx_str, method_name, contamination, seed)
        row["rho_str"]       = m_str["rho"]
        row["rho_tilde_str"] = m_str["rho_tilde"]

        # --- Stratified (k-means proxy) arm ---
        idx_km = _stratify_by_kmeans(X, n, rng, K=KMEANS_STRATA_K, seed=seed)
        m_km   = _run_arm(X, scores_full, idx_km, method_name, contamination, seed)
        row["rho_str_km"]       = m_km["rho"]
        row["rho_tilde_str_km"] = m_km["rho_tilde"]

    except Exception as e:
        warnings.warn(f"Method {method_name} rep {rep} p={p} failed: {e}")
        row["error"] = str(e)

    row["time_total_s"] = time.perf_counter() - t0
    return row


def run_experiment(datasets: dict) -> pd.DataFrame:
    """Run E6 for all (method, dataset, p, rep) combinations."""
    all_records = []
    t_start = time.perf_counter()
    total_done = 0

    for ds_name, (X, y) in datasets.items():
        tier = DATASET_TIERS.get(ds_name, "?")
        n_reps = N_REPS_T3 if tier == "T3" else N_REPS_T2
        print(f"\nDataset: {ds_name} (N={len(X)}, d={X.shape[1]}, tier={tier}, reps={n_reps})")

        for method_name in E6_METHODS:
            cls = METHOD_CLASS_MAP.get(method_name, "?")
            t_method = time.perf_counter()

            for p in P_VALUES:
                for rep in range(n_reps):
                    row = run_single_condition(X, y, method_name, p, rep)
                    row["dataset"]      = ds_name
                    row["tier"]         = tier
                    row["method_class"] = cls
                    all_records.append(row)
                    total_done += 1

            # Per-method summary
            subset = [r for r in all_records
                      if r["dataset"] == ds_name and r["method"] == method_name]
            mean_delta = np.nanmean([
                r["rho_str"] - r["rho_srs"] for r in subset
                if not np.isnan(r["rho_str"]) and not np.isnan(r["rho_srs"])
            ])
            print(
                f"  [{cls}] {method_name}: mean Δρ(str-SRS)={mean_delta:+.4f} "
                f"({time.perf_counter() - t_method:.1f}s)"
            )

    elapsed = time.perf_counter() - t_start
    print(f"\nE6 total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return pd.DataFrame(all_records)


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def compute_delta_rho_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Delta_rho = rho_str - rho_srs per (method, dataset, p).
    Aggregate mean ± std Delta_rho; run Wilcoxon signed-rank per method.
    """
    df = df.copy()
    df["delta_rho_label"] = df["rho_str"] - df["rho_srs"]
    df["delta_rho_km"]    = df["rho_str_km"] - df["rho_srs"]

    records = []
    for (method, p, cls), grp in df.groupby(["method", "p", "method_class"]):
        # Per-dataset mean delta (one observation per dataset for Wilcoxon)
        per_ds = grp.groupby("dataset")["delta_rho_label"].mean().dropna()
        per_ds_km = grp.groupby("dataset")["delta_rho_km"].mean().dropna()

        n_ds = len(per_ds)
        mean_delta = float(per_ds.mean()) if n_ds > 0 else np.nan
        std_delta  = float(per_ds.std())  if n_ds > 0 else np.nan

        # Wilcoxon signed-rank H0: Delta_rho <= 0, H1: Delta_rho > 0
        wx_stat, wx_p = np.nan, np.nan
        if n_ds >= 4:
            try:
                res = stats.wilcoxon(per_ds.values, alternative="greater")
                wx_stat, wx_p = float(res.statistic), float(res.pvalue)
            except Exception:
                pass

        records.append({
            "method":        method,
            "method_class":  cls,
            "p":             p,
            "n_datasets":    n_ds,
            "mean_delta_label": mean_delta,
            "std_delta_label":  std_delta,
            "mean_delta_km": float(per_ds_km.mean()) if len(per_ds_km) > 0 else np.nan,
            "wilcoxon_stat": wx_stat,
            "wilcoxon_p":    wx_p,
        })

    return pd.DataFrame(records)


def run_statistical_tests(df: pd.DataFrame) -> None:
    """Print and save Delta_rho summary and Wilcoxon tests."""
    delta_df = compute_delta_rho_table(df)
    delta_path = TABLES_DIR / "e6_delta_rho.csv"
    delta_df.to_csv(delta_path, index=False)
    print(f"\nDelta-rho table saved: {delta_path}")
    print(delta_df[["method", "method_class", "p", "mean_delta_label",
                     "std_delta_label", "wilcoxon_p"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(df: pd.DataFrame) -> None:
    """
    Figure e6_delta_rho_boxplot.pdf:
      Faceted boxplot (one column per p value) showing distribution of
      Delta_rho = rho_stratified - rho_SRSWOR across datasets × reps,
      grouped by method, colored by class (I=blue, II=orange, IV=red).
      Reference line at Delta_rho = 0 (no improvement).

    Figure e6_delta_rho_km_boxplot.pdf:
      Same plot for k-means proxy strata.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        warnings.warn("matplotlib not available — skipping E6 plots.")
        return

    df = df.copy()
    df["delta_label"] = df["rho_str"]    - df["rho_srs"]
    df["delta_km"]    = df["rho_str_km"] - df["rho_srs"]

    CLASS_COLORS = {"I": "#4878CF", "II": "#D65F5F", "III": "#6ACC65", "IV": "#B47CC7"}
    p_values = sorted(df["p"].unique())
    methods  = E6_METHODS

    for suffix, col in [("label", "delta_label"), ("km", "delta_km")]:
        fig, axes = plt.subplots(1, len(p_values), figsize=(4 * len(p_values), 5),
                                  sharey=True)
        if len(p_values) == 1:
            axes = [axes]

        for ax, p in zip(axes, p_values):
            sub = df[df["p"] == p]
            data_by_method = []
            colors         = []
            labels         = []
            for m in methods:
                grp   = sub[sub["method"] == m][col].dropna().values
                cls   = METHOD_CLASS_MAP.get(m, "?")
                data_by_method.append(grp)
                colors.append(CLASS_COLORS.get(cls, "grey"))
                labels.append(m)

            bp = ax.boxplot(data_by_method, patch_artist=True, notch=False,
                            medianprops=dict(color="black", linewidth=1.5),
                            flierprops=dict(marker=".", markersize=3, alpha=0.4))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.axhline(0, linestyle="--", linewidth=1.0, color="black", alpha=0.7,
                       label="No improvement")
            ax.set_xticks(range(1, len(methods) + 1))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_title(f"p = {p:.0%}", fontsize=10)
            ax.set_xlabel("Method")
            if ax is axes[0]:
                strat_label = ("GT label" if suffix == "label" else "k-means proxy")
                ax.set_ylabel(rf"$\Delta\rho$ (stratified$_{{{strat_label}}}$ − SRSWOR)")

        legend_patches = [
            mpatches.Patch(color=CLASS_COLORS[c], alpha=0.7, label=f"Class {c}")
            for c in ["I", "II", "IV"]
        ]
        fig.legend(handles=legend_patches, loc="upper right", fontsize=8)
        fig.suptitle(
            r"E6 — $\Delta\rho$ by method and sampling fraction" + f"\n(strata: {suffix})",
            fontsize=11
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fpath = FIGURES_DIR / f"e6_delta_rho_{suffix}_boxplot.pdf"
        fig.savefig(fpath, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved: {fpath}")


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_results(df: pd.DataFrame) -> None:
    raw_path = RESULTS_DIR / "e6_raw_results.csv"
    df.to_csv(raw_path, index=False)
    print(f"Raw results saved: {raw_path}")

    timing_path = RESULTS_DIR / "timing_per_condition.csv"
    df_timing = df[["dataset", "method", "p", "rep", "time_total_s"]].copy()
    df_timing["experiment"] = "E6"
    if timing_path.exists():
        df_timing.to_csv(timing_path, mode="a", header=False, index=False)
    else:
        df_timing.to_csv(timing_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("E6 — Stratified vs. SRSWOR Sampling")
    print(f"  Methods:  {E6_METHODS}")
    print(f"  Datasets: {E6_DATASETS}")
    print(f"  p values: {P_VALUES}")
    print(f"  Reps: T2={N_REPS_T2}, T3={N_REPS_T3}")
    print("=" * 65)

    t0 = time.perf_counter()

    print("\nLoading datasets...")
    datasets = load_multiple(E6_DATASETS, skip_on_error=True)
    if not datasets:
        print("ERROR: No datasets loaded.")
        sys.exit(1)
    print(f"Loaded {len(datasets)} / {len(E6_DATASETS)} datasets.")

    df = run_experiment(datasets)
    save_results(df)
    run_statistical_tests(df)
    make_plots(df)

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 65)
    print(f"E6 complete. Total wall time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Results:  {RESULTS_DIR}")
    print(f"Tables:   {TABLES_DIR}")
    print(f"Figures:  {FIGURES_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
