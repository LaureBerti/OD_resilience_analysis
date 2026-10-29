"""paper_numbers.py — numbers quoted in the text of the paper that no table prints.

Writes analysis/experiments/paper_numbers.csv (item, key, value) and
analysis/experiments/dataset_characteristics.csv (Table 4, S3 Table). Needs the datasets in data/.
  E1 zero spread  fits (full fit; samples of p = 10 %, seeds 42+r) with a non-constant feature of zero MAD / IQR
  E1 ties         MAD and BoxPlot replicates with a tie at the K-th score; median of the larger tie count
  E1 ROD          mean rho of ROD with and without the projection (d > 5)
  E3 capped       LOF and KNN exponents with all cells and without the cells where k > n - 1
  E5 forest terms psi^2/(N - psi^2) and psi/(n - psi) of Proposition 3 per dataset and p
  E6 trees        datasets with rho(T=1000) > rho(T=50), non-increasing steps, Wilcoxon and Holm p
  E8 types        anomaly type with the lowest class mean, per class and dataset
Run: .venv/bin/python analysis/code/paper_numbers.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECT = HERE.parents[1]
EXP = ROOT / "experiments"
sys.path.insert(0, str(HERE))
import datasets_v2 as DS
from analyze_rest import load, two_stage


@dataclass
class Cfg:
    n_cap: int = 5000
    seed: int = 42
    p: float = 0.10
    reps: int = 50
    psi: int = 256
    out_dir: str = "experiments"


ConfigStore.instance().store(name="pn", node=Cfg)


def characteristics(c):
    rows = []
    for name in DS.CORE_DATASETS + list(DS.EXTRA):
        X, y = DS._raw(name)
        N = len(X)
        rows.append(dict(dataset=name, N=N, d=X.shape[1], outlier_pct=round(100 * float(np.mean(y)), 2),
                         tier=DS.tier_of(N), N_ref=min(N, c.n_cap), const_feats=int((X.std(0) == 0).sum()),
                         dup_rows=int(N - len(np.unique(X, axis=0))),
                         set="core" if name in DS.CORE_DATASETS else "E2-extra"))
    return pd.DataFrame(rows)


def zero_spread(A):
    nc = A.std(0) > 0
    med = np.median(A, 0)
    q1, q3 = np.percentile(A, [25, 75], axis=0)
    return int(((np.median(np.abs(A - med), 0) == 0) & nc).sum()), int(((q3 - q1 == 0) & nc).sum())


def e1_zero_spread(c, add):
    for name in DS.CORE_DATASETS:
        X, _, _ = DS.load(name, c.n_cap, c.seed)
        N = len(X); n = max(2, int(np.floor(c.p * N)))
        mad_full, iqr_full = zero_spread(X)
        mad_s = iqr_s = 0
        for r in range(c.reps):
            idx = np.sort(np.random.default_rng(c.seed + r).choice(N, size=n, replace=False))
            a, b = zero_spread(X[idx]); mad_s += a > 0; iqr_s += b > 0
        add("E1 zero spread", f"{name} MAD: features in full fit / samples of {c.reps}", f"{mad_full} / {mad_s}")
        add("E1 zero spread", f"{name} IQR: features in full fit / samples of {c.reps}", f"{iqr_full} / {iqr_s}")


def e1_raw(add):
    df = load("E1")
    m = df[df.method.isin(["MAD", "BoxPlot"]) & (df.degenerate != True)]
    tied = (m.ties_S > 0) | (m.ties_D > 0)
    add("E1 ties", "share of MAD and BoxPlot replicates tied at the K-th score", round(float(tied.mean()), 3))
    add("E1 ties", "median of max(ties_S, ties_D) among tied replicates",
        float(m[["ties_S", "ties_D"]].max(axis=1)[tied].median()))
    rod = df[df.method == "ROD"].groupby(["dataset", "d"], as_index=False).rho.mean()
    add("E1 ROD", "mean rho, d <= 5 (no projection)", round(float(rod[rod.d <= 5].rho.mean()), 3))
    add("E1 ROD", "mean rho, d > 5", round(float(rod[rod.d > 5].rho.mean()), 3))


def e3_capped(add):
    df = load("E3")
    df["k"] = df.variant.str.extract(r"n_neighbors=(\d+)").astype(int)
    g = df.groupby(["method", "dataset", "k", "p"], as_index=False).agg(rho=("rho", "mean"), n=("n", "first"))
    g["y"] = np.log(1 - g.rho.clip(upper=1 - 1e-9)); g["logk"] = np.log(g.k); g["logp"] = np.log(g.p)
    g["capped"] = g.k > g.n - 1
    for m, t in g.groupby("method"):
        t = t[t.rho < 1]
        add("E3 capped", f"{m} capped cells", int(t.capped.sum()))
        for lab, u in (("all cells", t), ("capped cells left out", t[~t.capped])):
            r = two_stage(u, ("logk", "logp"))
            for x, nm in (("logk", "a"), ("logp", "b")):
                add("E3 capped", f"{m} {nm}, {lab}", f"{r[x]['coef']:.4f} [{r[x]['lo']:.4f}, {r[x]['hi']:.4f}]")


def e5_terms(c, add):
    df = load("E5")
    for (name, p), u in df.groupby(["dataset", "p"]):
        N, n = int(u.N.iloc[0]), int(u.n.iloc[0])
        add("E5 forest terms", f"{name} p={p:g}: psi^2/(N-psi^2), psi/(n-psi)",
            f"{c.psi**2 / (N - c.psi**2):.4f}, {c.psi / (n - c.psi):.4f}")


def e6_trees(add):
    df = load("X5")
    df["T"] = df.variant.str.extract(r"n_estimators=(\d+)").astype(int)
    g = df.groupby(["method", "dataset", "T"], as_index=False).rho.mean()
    ps = {}
    steps, worst, n_steps = 0, 0.0, 0
    for m, t in g.groupby("method"):
        w = t.pivot(index="dataset", columns="T", values="rho")
        Ts = sorted(w.columns)
        add("E6 trees", f"{m}: datasets with rho(T={Ts[-1]}) > rho(T={Ts[0]})", f"{int((w[Ts[-1]] > w[Ts[0]]).sum())} of {len(w)}")
        d = w[Ts].diff(axis=1).iloc[:, 1:]
        steps += int((d <= 0).sum().sum()); n_steps += d.size
        worst = max(worst, float(d[d <= 0].abs().max().max()) if (d <= 0).any().any() else 0.0)
        ps[m] = stats.wilcoxon(w[Ts[-1]], w[Ts[0]]).pvalue
    add("E6 trees", "non-increasing steps / all steps", f"{steps} / {n_steps}")
    add("E6 trees", "largest change at a non-increasing step", round(worst, 4))
    order = sorted(ps, key=ps.get)
    holm = {m: min(1.0, max((len(ps) - i) * ps[k] for i, k in enumerate(order[:order.index(m) + 1])))
            for m in order}
    for m in order:
        add("E6 trees", f"{m}: Wilcoxon p, Holm-adjusted p", f"{ps[m]:.4f}, {holm[m]:.4f}")


def e8_types(add):
    df = load("X7")
    df["type"] = df.dataset.str.split("_").str[0]; df["base"] = df.dataset.str.split("_", n=1).str[1]
    g = df.groupby(["method", "method_class", "type", "base"], as_index=False).rho.mean()
    cl = g.groupby(["method_class", "base", "type"]).rho.mean().unstack()
    for (k, base), row in cl.iterrows():
        if k in ("I", "II"):
            add("E8 types", f"Class {k} on {base}: " + ", ".join(f"{t} {row[t]:.4f}" for t in row.index),
                f"lowest: {row.idxmin()}")


@hydra.main(version_base=None, config_name="pn")
def main(c: Cfg) -> None:
    out = ROOT / c.out_dir; out.mkdir(parents=True, exist_ok=True)
    rows = []
    add = lambda item, key, value: rows.append(dict(item=item, key=key, value=value))
    characteristics(c).to_csv(out / "dataset_characteristics.csv", index=False)
    e1_zero_spread(c, add)
    e1_raw(add)
    e3_capped(add)
    e5_terms(c, add)
    e6_trees(add)
    e8_types(add)
    t = pd.DataFrame(rows)
    t.to_csv(out / "paper_numbers.csv", index=False)
    print(t.to_string())


if __name__ == "__main__":
    sys.argv += ["hydra.run.dir=.", "hydra.output_subdir=null"]
    main()
