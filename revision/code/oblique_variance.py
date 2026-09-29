"""oblique_variance.py — per-tree depth variance of axis-parallel vs oblique (fully extended) isolation trees.

Tests the stylized derivation (S1 Appendix, Lemma S3): under independent per-split separation with
equal feature ranges, Var_oblique / Var_axis ~ (pi/2) s^2 and E-depth ratio ~ sqrt(pi/2) s, where
s = ||delta||_1 / (sqrt(d) ||delta||_2) and delta = x - (coordinate-wise median), features min-max scaled.
Both forests use the SAME subsamples (psi = 256) for every tree, so only the split rule differs.
Records analysed: those near the flagging threshold (mean-depth rank between 8% and 12% under the
axis-parallel forest, i.e. around the top-10% cut). Output: revision/experiments/E6b/*.csv
Run: .venv/bin/python revision/code/oblique_variance.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))


@dataclass
class Cfg:
    datasets: list = field(default_factory=lambda: ["letter", "thyroid", "optdigits", "satellite",
                                                    "pendigits", "annthyroid", "shuttle", "smtp"])
    n_trees: int = 300
    psi: int = 256
    n_cap: int = 5000
    seed: int = 42
    band: list = field(default_factory=lambda: [0.08, 0.12])


ConfigStore.instance().store(name="ov", node=Cfg)


def build(X, depth, limit, oblique, rng):
    n, d = X.shape
    if depth >= limit or n <= 1:
        return ("leaf", n)
    if oblique:
        w = rng.standard_normal(d)
    else:
        w = np.zeros(d); w[rng.integers(d)] = 1.0
    proj = X @ w
    lo, hi = proj.min(), proj.max()
    if lo >= hi:
        return ("leaf", n)
    c = rng.uniform(lo, hi)
    m = proj <= c
    return ("split", w, c, build(X[m], depth + 1, limit, oblique, rng), build(X[~m], depth + 1, limit, oblique, rng))


def cfac(n):
    return 0.0 if n <= 1 else 1.0 if n == 2 else 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


def depths(X, tree):
    out = np.empty(len(X))
    def go(idx, t, dep):
        if len(idx) == 0:
            return
        if t[0] == "leaf":
            out[idx] = dep + cfac(t[1]); return
        _, w, c, l, r = t
        m = X[idx] @ w <= c
        go(idx[m], l, dep + 1); go(idx[~m], r, dep + 1)
    go(np.arange(len(X)), tree, 0)
    return out


@hydra.main(version_base=None, config_name="ov")
def main(c: Cfg) -> None:
    import datasets_v2 as DS
    out = PROJECT / "revision" / "experiments" / "E6b"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in c.datasets:
        X, _, _ = DS.load(name, c.n_cap, c.seed)
        X = X[:, X.std(0) > 0]
        X = (X - X.min(0)) / (X.max(0) - X.min(0))                  # equal feature ranges
        n, d = X.shape
        rng = np.random.default_rng(c.seed)
        limit = int(np.ceil(np.log2(c.psi)))
        Dax = np.empty((c.n_trees, n)); Dob = np.empty((c.n_trees, n))
        for t in range(c.n_trees):
            idx = rng.choice(n, size=min(c.psi, n), replace=False)   # same subsample for both rules
            Dax[t] = depths(X, build(X[idx], 0, limit, False, np.random.default_rng(c.seed + 10 * t)))
            Dob[t] = depths(X, build(X[idx], 0, limit, True, np.random.default_rng(c.seed + 10 * t + 1)))
        mean_ax = Dax.mean(0)
        order = np.argsort(mean_ax)                                  # smallest depth = most anomalous
        lo_i, hi_i = int(c.band[0] * n), int(c.band[1] * n)
        band = order[lo_i:hi_i]
        delta = X - np.median(X, 0)
        s = np.abs(delta).sum(1) / (np.sqrt(d) * np.linalg.norm(delta, axis=1) + 1e-12)
        for i in band:
            rows.append(dict(dataset=name, d=d, record=int(i), s=s[i], var_ax=Dax[:, i].var(ddof=1),
                             var_ob=Dob[:, i].var(ddof=1), mean_ax=Dax[:, i].mean(), mean_ob=Dob[:, i].mean()))
        print(name, d, len(band), flush=True)
    df = pd.DataFrame(rows)
    df["ratio_var"] = df.var_ob / df.var_ax
    df["pred_var"] = (np.pi / 2) * df.s ** 2
    df.to_csv(out / "per_record.csv", index=False)
    summ = df.groupby(["dataset", "d"]).agg(records=("record", "size"), s_median=("s", "median"),
                                            var_ratio_median=("ratio_var", "median"),
                                            pred_median=("pred_var", "median")).reset_index()
    summ.to_csv(out / "summary.csv", index=False)
    print(summ.round(3).to_string())


if __name__ == "__main__":
    sys.argv += ["hydra.run.dir=.", "hydra.output_subdir=null"]
    main()
