"""
exp_E7_cluster.py
-----------------
E7 — Cluster Sampling: Resilience Degradation via Inflated Design Effect

Hypothesis H7:
  Under cluster sampling (k-means based, equal-probability within-cluster SRSWOR),
  the mean Dice resilience rho_cluster < rho_SRSWOR for Class I and Class II
  methods when K is small (high ICC). The resilience loss correlates positively
  with the design effect DEFF = 1 + (m - 1) * rho_hat_ICC.

  Formally: Delta_rho = rho_cluster - rho_SRSWOR < 0 for Classes I/II at K in {5,10}
  (Wilcoxon signed-rank H0: Delta_rho >= 0, one-sided p < 0.05).

Cluster sampling protocol:
  - Cluster definition: k-means with K clusters on standardized X (n_init=10).
  - Sampling: draw m = round(n / K) records from each cluster by SRSWOR.
    Total n_actual = sum of actual per-cluster draws (may differ slightly from
    target n due to integer rounding and small clusters).
  - K in {5, 10, 20}; p = n/N in {5%, 10%, 20%} => m = round(p*N/K).

ICC estimation:
  For each feature j, within each cluster h, compute within-cluster variance
  s_{h,j}^2 and between-cluster variance using the one-way ANOVA decomposition:
    MSB_j = (1 / (K-1)) * sum_h n_h (x_bar_{h,j} - x_bar_j)^2
    MSW_j = (1 / (N - K)) * sum_h sum_{i in h} (x_{i,j} - x_bar_{h,j})^2
    rho_ICC_j = (MSB_j - MSW_j) / (MSB_j + (m - 1) * MSW_j)

  The aggregate ICC is the mean over features: rho_hat_ICC = mean_j(rho_ICC_j).
  This gives DEFF = 1 + (m - 1) * rho_hat_ICC.

  The theoretical prediction is that higher DEFF => more negative Delta_rho
  (Figure e7_delta_rho_vs_deff_scatter.pdf validates this quantitative relationship).

Setup:
  - Methods: Class I (ThreeSigma, BoxPlot, MAD, ChiSquare),
             Class II (COPOD, ECOD, HBOS),
             Class III (IForest),
             LOF (Class IV) included for Tier 2 only (N-dependent O(N^2) cost)
  - Datasets: Tier 2 + Tier 3 (N >= 1000)
  - K in {5, 10, 20}; p in {5%, 10%, 20%}
  - Reps: 30 (Tier 2), 10 (Tier 3)

Output files:
  results/e7_raw_results.csv
  tables/e7_delta_rho.csv
  tables/e7_deff_summary.csv
  tables/e7_wilcoxon.csv
  figures/e7_delta_rho_vs_deff_scatter.pdf
  figures/e7_delta_rho_by_K_boxplot.pdf
  timing_per_condition.csv (appended)
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR  = Path(__file__).resolve().parent
OUTPUT_DIR  = SCRIPT_DIR.parent
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
P_VALUES     = [0.05, 0.10, 0.20]
K_VALUES     = [5, 10, 20]   # number of clusters
N_REPS_T2    = 30
N_REPS_T3    = 10
CONTAMINATION = 0.10
BASE_SEED    = 42

# Class I + II (theorem applies) + III (check) + LOF T2 only (negative control)
E7_METHODS_ALL  = ["ThreeSigma", "BoxPlot", "MAD", "ChiSquare",
                   "COPOD", "ECOD", "HBOS",
                   "IForest"]
E7_METHODS_T2_EXTRA = ["LOF"]  # LOF only for Tier 2 (N <= 10K)

E7_DATASETS = TIER2_DATASETS + TIER3_DATASETS


# ---------------------------------------------------------------------------
# ICC and DEFF estimation
# ---------------------------------------------------------------------------

def estimate_icc_deff(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    K: int,
    m: float,
) -> tuple[float, float]:
    """
    Estimate the mean intraclass correlation coefficient (ICC) and design effect
    (DEFF) from a cluster partition of X.

    Uses the one-way ANOVA estimator of ICC for each feature j, then averages
    across features (after clamping ICC to [-1, 1] for numerical stability).

    Parameters
    ----------
    X : (N, d) feature matrix
    cluster_labels : (N,) integer cluster assignment from k-means
    K : number of clusters used
    m : average cluster sample size (= n / K)

    Returns
    -------
    (rho_icc, deff) : (float, float)
        Mean ICC across features; DEFF = 1 + (m-1)*rho_icc.
    """
    N, d = X.shape
    icc_per_feature = []

    for j in range(d):
        xj = X[:, j]
        grand_mean = np.mean(xj)
        # Between-cluster sum of squares
        msb = 0.0
        msw = 0.0
        total_within = 0
        cluster_sizes = []
        for h in range(K):
            idx_h = np.where(cluster_labels == h)[0]
            n_h = len(idx_h)
            if n_h == 0:
                continue
            cluster_sizes.append(n_h)
            cluster_mean = np.mean(xj[idx_h])
            msb += n_h * (cluster_mean - grand_mean) ** 2
            msw += np.sum((xj[idx_h] - cluster_mean) ** 2)
            total_within += n_h

        n_clusters_nonempty = len(cluster_sizes)
        if n_clusters_nonempty < 2 or total_within <= n_clusters_nonempty:
            continue  # degenerate

        msb /= (n_clusters_nonempty - 1)
        msw /= (total_within - n_clusters_nonempty)

        if msw < 1e-12:
            # Constant within clusters => ICC = 1
            icc_j = 1.0
        else:
            # ANOVA-based ICC estimator
            m0 = m  # use target m (average sample size)
            icc_j = (msb - msw) / (msb + (m0 - 1) * msw)

        icc_j = float(np.clip(icc_j, -1.0, 1.0))
        icc_per_feature.append(icc_j)

    if not icc_per_feature:
        return 0.0, 1.0

    rho_icc = float(np.mean(icc_per_feature))
    deff    = 1.0 + (m - 1) * rho_icc
    deff    = float(np.clip(deff, 0.01, None))  # DEFF >= 0
    return rho_icc, deff


# ---------------------------------------------------------------------------
# Cluster sampling
# ---------------------------------------------------------------------------

def cluster_sample(
    X: np.ndarray,
    n: int,
    K: int,
    rng: np.random.Generator,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Cluster sampling: k-means partition of X into K clusters, then proportionate
    SRSWOR within each cluster.

    Parameters
    ----------
    X : (N, d)
    n : target total sample size
    K : number of clusters
    rng : numpy Generator
    seed : int for KMeans reproducibility

    Returns
    -------
    (idx_sample, cluster_labels, rho_icc, deff)
        idx_sample : sorted integer indices of sampled records
        cluster_labels : (N,) k-means cluster assignment for all records
        rho_icc : mean ICC estimate
        deff : design effect estimate
    """
    N = len(X)
    K_eff = min(K, N // 2)
    m_target = n / K_eff  # target records per cluster (float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=K_eff, random_state=seed, n_init=10)
    cluster_labels = km.fit_predict(X_scaled)

    # Proportionate within-cluster SRSWOR
    sampled_indices = []
    for h in range(K_eff):
        idx_h = np.where(cluster_labels == h)[0]
        N_h   = len(idx_h)
        # Proportionate allocation: m_h = round(n * N_h / N)
        m_h   = max(1, round(n * N_h / N))
        m_h   = min(m_h, N_h)
        sampled_h = rng.choice(idx_h, size=m_h, replace=False)
        sampled_indices.append(sampled_h)

    all_idx = np.concatenate(sampled_indices)

    # Adjust to exactly n via random trimming / padding
    if len(all_idx) > n:
        all_idx = rng.choice(all_idx, size=n, replace=False)
    elif len(all_idx) < n:
        remaining = np.setdiff1d(np.arange(N), all_idx)
        if len(remaining) > 0:
            extra = rng.choice(
                remaining, size=min(n - len(all_idx), len(remaining)), replace=False
            )
            all_idx = np.concatenate([all_idx, extra])

    all_idx = np.sort(all_idx)

    # Estimate ICC and DEFF on full population clusters
    rho_icc, deff = estimate_icc_deff(X, cluster_labels, K_eff, m_target)

    return all_idx, cluster_labels, rho_icc, deff


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def run_single_condition(
    X: np.ndarray,
    method_name: str,
    p: float,
    K: int,
    rep: int,
    contamination: float = CONTAMINATION,
) -> dict:
    """
    Run one (method, p, K, rep) condition.
    Returns Dice resilience for both SRSWOR and cluster arms.
    Full-data fit is shared between both arms.
    """
    rng  = np.random.default_rng(BASE_SEED + rep)
    seed = BASE_SEED + rep
    N    = len(X)
    n    = max(K + 1, int(np.floor(p * N)))   # ensure n >= K

    row = {
        "method":          method_name,
        "p":               p,
        "K":               K,
        "rep":             rep,
        "n_full":          N,
        "n_sample_target": n,
        "n_sample_actual": np.nan,
        # SRSWOR arm
        "rho_srs":         np.nan,
        "rho_tilde_srs":   np.nan,
        # Cluster arm
        "rho_cluster":     np.nan,
        "rho_tilde_cluster": np.nan,
        # ICC / DEFF diagnostics
        "rho_icc":         np.nan,
        "deff":            np.nan,
        "time_total_s":    np.nan,
        "error":           None,
    }

    t0 = time.perf_counter()
    try:
        # Fit on full dataset once; reuse scores for both arms
        det_full = make_detector(method_name, contamination=contamination, random_state=seed)
        det_full.fit(X)
        scores_full = det_full.decision_function(X)

        # --- SRSWOR arm ---
        idx_srs = np.sort(rng.choice(N, size=n, replace=False))
        det_srs = make_detector(method_name, contamination=contamination, random_state=seed)
        det_srs.fit(X[idx_srs])
        s_srs = det_srs.decision_function(X[idx_srs])
        m_srs = compute_resilience_pair(s_srs, scores_full[idx_srs], contamination)
        row["rho_srs"]       = m_srs["rho"]
        row["rho_tilde_srs"] = m_srs["rho_tilde"]

        # --- Cluster arm ---
        idx_cl, cl_labels, rho_icc, deff = cluster_sample(X, n, K, rng, seed)
        row["n_sample_actual"] = len(idx_cl)
        row["rho_icc"]         = rho_icc
        row["deff"]            = deff

        det_cl = make_detector(method_name, contamination=contamination, random_state=seed)
        det_cl.fit(X[idx_cl])
        s_cl = det_cl.decision_function(X[idx_cl])
        m_cl = compute_resilience_pair(s_cl, scores_full[idx_cl], contamination)
        row["rho_cluster"]       = m_cl["rho"]
        row["rho_tilde_cluster"] = m_cl["rho_tilde"]

    except Exception as e:
        warnings.warn(f"Method {method_name} K={K} p={p} rep={rep} failed: {e}")
        row["error"] = str(e)

    row["time_total_s"] = time.perf_counter() - t0
    return row


def run_experiment(datasets: dict) -> pd.DataFrame:
    """Run E7 for all (method, dataset, p, K, rep) combinations."""
    all_records = []
    t_start = time.perf_counter()

    for ds_name, (X, _y) in datasets.items():
        tier = DATASET_TIERS.get(ds_name, "?")
        n_reps = N_REPS_T3 if tier == "T3" else N_REPS_T2
        methods = E7_METHODS_ALL + (E7_METHODS_T2_EXTRA if tier == "T2" else [])

        print(f"\nDataset: {ds_name} (N={len(X)}, d={X.shape[1]}, tier={tier}, "
              f"reps={n_reps}, methods={len(methods)})")

        for method_name in methods:
            cls = METHOD_CLASS_MAP.get(method_name, "?")
            t_method = time.perf_counter()

            for K in K_VALUES:
                for p in P_VALUES:
                    for rep in range(n_reps):
                        row = run_single_condition(X, method_name, p, K, rep)
                        row["dataset"]      = ds_name
                        row["tier"]         = tier
                        row["method_class"] = cls
                        all_records.append(row)

            # Summary
            subset = [r for r in all_records
                      if r["dataset"] == ds_name and r["method"] == method_name]
            mean_delta = np.nanmean([
                r["rho_cluster"] - r["rho_srs"]
                for r in subset
                if not np.isnan(r.get("rho_cluster", np.nan))
                and not np.isnan(r.get("rho_srs", np.nan))
            ])
            mean_deff = np.nanmean([r["deff"] for r in subset
                                    if not np.isnan(r.get("deff", np.nan))])
            print(
                f"  [{cls}] {method_name}: mean Δρ(cl-SRS)={mean_delta:+.4f}, "
                f"mean DEFF={mean_deff:.3f}  ({time.perf_counter()-t_method:.1f}s)"
            )

    elapsed = time.perf_counter() - t_start
    print(f"\nE7 total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return pd.DataFrame(all_records)


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def compute_delta_rho_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (method, K, p): Delta_rho = rho_cluster - rho_srs.
    Wilcoxon signed-rank H0: Delta_rho >= 0 (cluster no worse).
    """
    df = df.copy()
    df["delta_rho"] = df["rho_cluster"] - df["rho_srs"]

    records = []
    for (method, K, p, cls), grp in df.groupby(["method", "K", "p", "method_class"]):
        per_ds = grp.groupby("dataset")["delta_rho"].mean().dropna()
        n_ds   = len(per_ds)
        mean_d = float(per_ds.mean()) if n_ds > 0 else np.nan
        std_d  = float(per_ds.std())  if n_ds > 0 else np.nan
        mean_deff = float(grp["deff"].mean())

        # H0: Delta_rho >= 0  (one-sided, direction = "less" means cluster is worse)
        wx_stat, wx_p = np.nan, np.nan
        if n_ds >= 4:
            try:
                res = stats.wilcoxon(per_ds.values, alternative="less")
                wx_stat, wx_p = float(res.statistic), float(res.pvalue)
            except Exception:
                pass

        records.append({
            "method":       method,
            "method_class": cls,
            "K":            K,
            "p":            p,
            "n_datasets":   n_ds,
            "mean_delta":   mean_d,
            "std_delta":    std_d,
            "mean_deff":    mean_deff,
            "wilcoxon_stat": wx_stat,
            "wilcoxon_p":    wx_p,
        })

    return pd.DataFrame(records)


def compute_deff_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    DEFF and ICC summary by (dataset, K, p).
    """
    return (
        df.groupby(["dataset", "K", "p"])
        .agg(
            mean_deff=("deff", "mean"),
            std_deff=("deff", "std"),
            mean_icc=("rho_icc", "mean"),
            std_icc=("rho_icc", "std"),
        )
        .reset_index()
    )


def run_statistical_tests(df: pd.DataFrame) -> None:
    delta_df = compute_delta_rho_table(df)
    delta_path = TABLES_DIR / "e7_delta_rho.csv"
    delta_df.to_csv(delta_path, index=False)
    print(f"\nDelta-rho table saved: {delta_path}")
    print(delta_df[["method", "method_class", "K", "p",
                    "mean_delta", "std_delta", "mean_deff", "wilcoxon_p"]].to_string(index=False))

    deff_df = compute_deff_summary(df)
    deff_path = TABLES_DIR / "e7_deff_summary.csv"
    deff_df.to_csv(deff_path, index=False)
    print(f"\nDEFF summary saved: {deff_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(df: pd.DataFrame) -> None:
    """
    Figure e7_delta_rho_vs_deff_scatter.pdf:
      Scatter plot of Delta_rho vs. DEFF (one point per (dataset, method, K, p, rep)).
      Color by class; regression line per class.
      Expected: negative slope for Classes I/II.

    Figure e7_delta_rho_by_K_boxplot.pdf:
      Boxplot of Delta_rho by K (columns) and method (x-axis), colored by class.
      Reference line at 0.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from scipy.stats import linregress
    except ImportError:
        warnings.warn("matplotlib not available — skipping E7 plots.")
        return

    df = df.copy()
    df["delta_rho"] = df["rho_cluster"] - df["rho_srs"]
    df = df.dropna(subset=["delta_rho", "deff"])

    CLASS_COLORS = {"I": "#4878CF", "II": "#D65F5F", "III": "#6ACC65", "IV": "#B47CC7"}

    # --- Scatter plot: Delta_rho vs. DEFF ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for cls, grp in df.groupby("method_class"):
        color = CLASS_COLORS.get(cls, "grey")
        ax.scatter(grp["deff"], grp["delta_rho"],
                   alpha=0.3, s=12, color=color, label=f"Class {cls}")
        # Regression line
        if len(grp) >= 5:
            try:
                slope, intercept, r, p_val, _ = linregress(grp["deff"], grp["delta_rho"])
                x_range = np.linspace(grp["deff"].min(), grp["deff"].max(), 50)
                ax.plot(x_range, slope * x_range + intercept,
                        color=color, linewidth=1.5, linestyle="--",
                        label=f"Class {cls} fit (slope={slope:.3f}, R²={r**2:.2f})")
            except Exception:
                pass

    ax.axhline(0, linestyle=":", color="black", linewidth=1.0)
    ax.set_xlabel(r"Design Effect $\widehat{\mathrm{DEFF}} = 1 + (m-1)\hat\rho_{\mathrm{ICC}}$",
                  fontsize=10)
    ax.set_ylabel(r"$\Delta\rho$ (cluster − SRSWOR)", fontsize=10)
    ax.set_title("E7 — Resilience loss vs. design effect\n"
                 r"($K \in \{5,10,20\}$, $p \in \{5,10,20\}\%$, Tier 2+3 datasets)",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    plt.tight_layout()
    fpath = FIGURES_DIR / "e7_delta_rho_vs_deff_scatter.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {fpath}")

    # --- Boxplot: Delta_rho by K ---
    K_vals   = sorted(df["K"].unique())
    methods  = (E7_METHODS_ALL + E7_METHODS_T2_EXTRA)
    # Filter to methods present in data
    methods = [m for m in methods if m in df["method"].unique()]

    fig, axes = plt.subplots(1, len(K_vals), figsize=(4 * len(K_vals), 5), sharey=True)
    if len(K_vals) == 1:
        axes = [axes]

    for ax, K in zip(axes, K_vals):
        sub = df[df["K"] == K]
        data_by_method = []
        colors         = []
        for m in methods:
            grp = sub[sub["method"] == m]["delta_rho"].dropna().values
            cls = METHOD_CLASS_MAP.get(m, "?")
            data_by_method.append(grp)
            colors.append(CLASS_COLORS.get(cls, "grey"))

        bp = ax.boxplot(data_by_method, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker=".", markersize=3, alpha=0.4))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.axhline(0, linestyle="--", linewidth=1.0, color="black", alpha=0.7)
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"K = {K}", fontsize=10)
        ax.set_xlabel("Method")
        if ax is axes[0]:
            ax.set_ylabel(r"$\Delta\rho$ (cluster − SRSWOR)")

    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS[c], alpha=0.7, label=f"Class {c}")
        for c in ["I", "II", "III", "IV"]
    ]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=8)
    fig.suptitle(r"E7 — $\Delta\rho$ by method and number of clusters $K$", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fpath = FIGURES_DIR / "e7_delta_rho_by_K_boxplot.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {fpath}")


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_results(df: pd.DataFrame) -> None:
    raw_path = RESULTS_DIR / "e7_raw_results.csv"
    df.to_csv(raw_path, index=False)
    print(f"Raw results saved: {raw_path}")

    timing_path = RESULTS_DIR / "timing_per_condition.csv"
    df_timing = df[["dataset", "method", "p", "rep", "time_total_s"]].copy()
    df_timing["experiment"] = "E7"
    if timing_path.exists():
        df_timing.to_csv(timing_path, mode="a", header=False, index=False)
    else:
        df_timing.to_csv(timing_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("E7 — Cluster Sampling: Resilience Degradation")
    print(f"  Methods (all tiers): {E7_METHODS_ALL}")
    print(f"  Extra T2 methods:    {E7_METHODS_T2_EXTRA}")
    print(f"  Datasets: {E7_DATASETS}")
    print(f"  K values: {K_VALUES}")
    print(f"  p values: {P_VALUES}")
    print(f"  Reps: T2={N_REPS_T2}, T3={N_REPS_T3}")
    print("=" * 65)

    t0 = time.perf_counter()

    print("\nLoading datasets...")
    datasets = load_multiple(E7_DATASETS, skip_on_error=True)
    if not datasets:
        print("ERROR: No datasets loaded.")
        sys.exit(1)
    print(f"Loaded {len(datasets)} / {len(E7_DATASETS)} datasets.")

    df = run_experiment(datasets)
    save_results(df)
    run_statistical_tests(df)
    make_plots(df)

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 65)
    print(f"E7 complete. Total wall time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Results:  {RESULTS_DIR}")
    print(f"Tables:   {TABLES_DIR}")
    print(f"Figures:  {FIGURES_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
