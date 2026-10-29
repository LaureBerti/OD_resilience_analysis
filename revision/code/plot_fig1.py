"""plot_fig1.py — Fig 1 (E1): resilience by class, from revision/experiments/E1/raw/*.csv only.

(A) per-dataset class means (box) with one grey line per dataset joining its class means
    (the paired structure used by the Friedman / Wilcoxon tests);
(B) per-dataset mean of every method (one point per method x dataset), grouped by class.
Output: revision/revised/figures/Fig1.tif (PLOS: RGB, LZW, 600 dpi, 7.5 in wide) + Fig1_preview.png
Run: .venv/bin/python revision/code/plot_fig1.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore

PROJECT = Path(__file__).resolve().parents[2]
ORDER = ["I", "II", "III", "IV", "G"]
LABEL = {"I": "I", "II": "II", "III": "III", "IV": "IV", "G": "G"}
METHODS = {"I": ["ThreeSigma", "BoxPlot", "MAD", "ChiSquare", "Mahalanobis", "ROD"],
           "II": ["COPOD", "ECOD", "HBOS"], "III": ["IForest", "EIF"],
           "IV": ["LOF", "KNN", "ABOD", "FeatureBagging", "ADOD"], "G": ["KMeans", "OCSVM"]}
SHORT = {"ThreeSigma": "3σ", "BoxPlot": "Box", "MAD": "MAD", "ChiSquare": "χ²", "Mahalanobis": "Maha",
         "ROD": "ROD", "COPOD": "COPOD", "ECOD": "ECOD", "HBOS": "HBOS", "IForest": "IF", "EIF": "EIF",
         "LOF": "LOF", "KNN": "KNN", "ABOD": "ABOD", "FeatureBagging": "FB", "ADOD": "ADOD",
         "KMeans": "k-m", "OCSVM": "OCSVM"}
COLOR = {"I": "#0072B2", "II": "#009E73", "III": "#E69F00", "IV": "#D55E00", "G": "#7F7F7F"}


@dataclass
class Cfg:
    exp_dir: str = "revision/experiments/E1"
    out_dir: str = "revision/revised/figures"
    dpi: int = 600
    width_in: float = 7.5
    height_in: float = 3.0


ConfigStore.instance().store(name="f1", node=Cfg)


@hydra.main(version_base=None, config_name="f1")
def main(c: Cfg) -> None:
    plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    df = pd.concat([pd.read_csv(f) for f in sorted((PROJECT / c.exp_dir / "raw").glob("*.csv"))])
    dm = df.groupby(["dataset", "method", "method_class"], as_index=False).rho.mean()
    cm = dm.groupby(["dataset", "method_class"]).rho.mean().unstack()[ORDER]
    n_ds, n_rep = cm.shape[0], int(df.groupby(["dataset", "method"]).rep.nunique().max())

    fig, (a, b) = plt.subplots(1, 2, figsize=(c.width_in, c.height_in),
                               gridspec_kw={"width_ratios": [1, 2.2], "wspace": 0.18})
    x = np.arange(len(ORDER))
    for _, row in cm.iterrows():
        a.plot(x[:4], row[ORDER[:4]].values, color="0.75", lw=0.5, zorder=1)
    bp = a.boxplot([cm[k].dropna() for k in ORDER], positions=x, widths=0.5, patch_artist=True,
                   whis=(0, 100), showfliers=False, medianprops=dict(color="black", lw=0.9), zorder=2)
    for patch, k in zip(bp["boxes"], ORDER):
        patch.set(facecolor=COLOR[k], alpha=0.35, edgecolor=COLOR[k], lw=0.8)
    rng = np.random.default_rng(0)
    for i, k in enumerate(ORDER):
        a.scatter(i + rng.uniform(-0.12, 0.12, len(cm)), cm[k], s=7, color=COLOR[k], zorder=3, lw=0)
    a.set_xticks(x, [LABEL[k] for k in ORDER]); a.set_xlabel("Class")
    a.set_ylabel(r"Resilience $\rho$ (dataset mean)"); a.set_ylim(0, 1.02)
    a.text(-0.02, 1.04, "A", transform=a.transAxes, fontweight="bold", fontsize=10, ha="right")

    pos, ticks, labels = 0.0, [], []
    for k in ORDER:
        for m in METHODS[k]:
            v = dm[dm.method == m].rho.values
            b.scatter(pos + rng.uniform(-0.18, 0.18, len(v)), v, s=6, color=COLOR[k], alpha=0.8, lw=0)
            b.hlines(np.nanmean(v), pos - 0.3, pos + 0.3, color="black", lw=0.9)
            ticks.append(pos); labels.append(SHORT[m]); pos += 1
        pos += 0.6
    b.set_xticks(ticks, labels, rotation=90)
    b.set_ylim(0, 1.02); b.set_yticklabels([]); b.set_xlim(-0.8, pos - 0.8)
    b.text(-0.01, 1.04, "B", transform=b.transAxes, fontweight="bold", fontsize=10, ha="right")
    start = 0.0
    for k in ORDER:
        n = len(METHODS[k]); mid = start + (n - 1) / 2
        b.text(mid, 1.045, k, color=COLOR[k], ha="center", va="bottom",
               transform=b.get_xaxis_transform(), fontsize=8)
        start += n + 0.6
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", lw=0.3, color="0.88")
        ax.set_axisbelow(True)
    out = PROJECT / c.out_dir; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "Fig1.tif", dpi=c.dpi, bbox_inches="tight", pad_inches=0.05,
                pil_kwargs={"compression": "tiff_lzw"})
    from PIL import Image
    Image.open(out / "Fig1.tif").convert("RGB").save(out / "Fig1.tif", compression="tiff_lzw", dpi=(c.dpi, c.dpi))
    fig.savefig(out / "Fig1_preview.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
    print(f"Fig1: {n_ds} datasets, {n_rep} replicates per (method, dataset)")


if __name__ == "__main__":
    sys.argv += ["hydra.run.dir=.", "hydra.output_subdir=null"]
    main()
