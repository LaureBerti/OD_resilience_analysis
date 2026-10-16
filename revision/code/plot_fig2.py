"""plot_fig2.py — Fig 2 (E2): convergence with the sample size n, from revision/experiments/E2/raw only.

(A) 1 - mean rho against n (log-log) per method: median over datasets of the per-dataset mean
    (30 replicates), band = interquartile range over datasets; dotted reference slope -1/2.
(B) rho against n for IForest and EIF: median and interquartile range over datasets; dashed line n = psi = 256.
n is binned on a log grid so that datasets of different N can be pooled; bins holding fewer than min_datasets datasets are omitted.
Output: revision/revised/figures/Fig2.tif (RGB, LZW, 600 dpi) + Fig2_preview.png
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
STYLE = {"ThreeSigma": ("3σ", "#0072B2", "-", "o"), "MAD": ("MAD", "#56B4E9", "--", "s"),
         "COPOD": ("COPOD", "#009E73", "-", "^"), "ECOD": ("ECOD", "#2CA02C", "--", "v"),
         "IForest": ("IForest", "#E69F00", "-", "D"), "EIF": ("EIF", "#B8860B", "--", "d"),
         "LOF": ("LOF", "#D55E00", "-", "x"), "KNN": ("KNN", "#CC79A7", "--", "+")}


@dataclass
class Cfg:
    exp_dir: str = "revision/experiments/E2"
    out_dir: str = "revision/revised/figures"
    min_n: int = 10
    min_datasets: int = 12
    dpi: int = 600


ConfigStore.instance().store(name="f2", node=Cfg)


def summarise(g, col, bins, min_datasets):
    g = g.assign(nb=pd.cut(g.n, bins, labels=np.sqrt(bins[:-1] * bins[1:])))
    g = g.groupby(["method", "nb", "dataset"], observed=True)[col].mean().reset_index()
    s = g.groupby(["method", "nb"], observed=True)[col].agg(
        med="median", lo=lambda x: x.quantile(0.25), hi=lambda x: x.quantile(0.75), k="count").reset_index()
    s["nb"] = s.nb.astype(float)
    return s[s.k >= min_datasets]


@hydra.main(version_base=None, config_name="f2")
def main(c: Cfg) -> None:
    plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
                         "axes.linewidth": 0.6})
    df = pd.concat([pd.read_csv(f) for f in sorted((PROJECT / c.exp_dir / "raw").glob("*.csv"))])
    g = df.groupby(["method", "dataset", "p"], as_index=False).agg(rho=("rho", "mean"), n=("n", "first"))
    g = g[g.n >= c.min_n]
    g["div"] = (1 - g.rho).clip(lower=1e-3)
    bins = np.geomspace(c.min_n, 5000, 12)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.5, 3.5), gridspec_kw={"wspace": 0.28})
    s = summarise(g, "div", bins, c.min_datasets)
    for m, (lab, col, ls, mk) in STYLE.items():
        t = s[s.method == m]
        if t.empty: continue
        a.plot(t.nb, t.med, ls=ls, color=col, marker=mk, ms=3, lw=1, label=lab)
        a.fill_between(t.nb, t.lo, t.hi, color=col, alpha=0.10, lw=0)
    x0, x1 = 30, 3000
    a.plot([x0, x1], [0.5 * (x0 / x0) ** -0.5, 0.5 * (x1 / x0) ** -0.5], ":", color="black", lw=0.9,
           label="slope $-1/2$")
    a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(1e-2, 1.5)
    a.set_xlabel("Sample size $n$"); a.set_ylabel(r"$1-\bar\rho$ (median over datasets)")
    a.legend(ncol=5, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), columnspacing=1.0, handlelength=2.2)
    a.text(-0.02, 1.04, "A", transform=a.transAxes, fontweight="bold", fontsize=10, ha="right")
    s2 = summarise(g[g.method.isin(["IForest", "EIF"])], "rho", bins, c.min_datasets)
    for m in ["IForest", "EIF"]:
        lab, col, ls, mk = STYLE[m]; t = s2[s2.method == m]
        b.plot(t.nb, t.med, ls=ls, color=col, marker=mk, ms=3, lw=1, label=lab)
        b.fill_between(t.nb, t.lo, t.hi, color=col, alpha=0.15, lw=0)
    b.axvline(256, ls="--", color="0.4", lw=0.8)
    b.text(256 / 1.08, 0.05, r"$n=\psi=256$", color="0.3", fontsize=7, ha="right", va="bottom",
           bbox=dict(facecolor="white", edgecolor="none", pad=1))
    b.set_xscale("log"); b.set_ylim(0, 1.02); b.set_xlabel("Sample size $n$")
    b.set_ylabel(r"$\bar\rho$ (median over datasets)")
    b.legend(fontsize=7, frameon=False, loc="upper left")
    b.text(-0.02, 1.04, "B", transform=b.transAxes, fontweight="bold", fontsize=10, ha="right")
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False); ax.grid(lw=0.3, color="0.9"); ax.set_axisbelow(True)
    out = PROJECT / c.out_dir
    fig.savefig(out / "Fig2.tif", dpi=c.dpi, bbox_inches="tight", pad_inches=0.05,
                pil_kwargs={"compression": "tiff_lzw"})
    from PIL import Image
    Image.open(out / "Fig2.tif").convert("RGB").save(out / "Fig2.tif", compression="tiff_lzw", dpi=(c.dpi, c.dpi))
    fig.savefig(out / "Fig2_preview.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
    print("Fig2 datasets per method:", g.groupby("method").dataset.nunique().to_dict(),
          "reps:", int(df.groupby(["dataset", "method", "p"]).rep.nunique().max()))


if __name__ == "__main__":
    sys.argv += ["hydra.run.dir=.", "hydra.output_subdir=null"]
    main()
