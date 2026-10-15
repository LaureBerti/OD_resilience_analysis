"""validate_reimplementations.py — R4: test every reimplemented detector against a reference.

(a) ADOD: our vectorised port (detectors_v2.adod) vs the authors' original functions from
    github.com/Qian-Lily/ADOD/adod.py executed as plain Python (numba no-op; GPU FAISS flat L2
    index replaced by the equivalent exact CPU k-NN search).
(b) Earlier wrappers (scripts/method_wrappers.py) vs the reference
    implementations now used (detectors_v2), on the in-sample scores of each dataset.

Output: revision/experiments/validation/reimpl_validation.csv (+ .md). CPU-only.
Run: .venv/bin/python revision/code/validate_reimplementations.py [val.datasets='[glass,vertebral]'] [val.adod_only=true]
"""
from __future__ import annotations

import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE))
from datasets_v2 import _legacy_scripts
sys.path.insert(0, str(_legacy_scripts(PROJECT)))


@dataclass
class ValConfig:
    adod_repo: str = "external/ADOD"
    adod_datasets: list = field(default_factory=lambda: ["lympho", "glass", "vertebral", "ionosphere", "breastw"])
    adod_extra_subsample: int = 1000
    adod_large: list = field(default_factory=lambda: ["annthyroid", "pendigits", "satellite"])
    adod_only: bool = False
    datasets: list = field(default_factory=lambda: ["lympho", "glass", "vertebral", "ionosphere", "breastw",
                                                    "letter", "thyroid", "annthyroid", "pendigits", "satellite",
                                                    "shuttle", "smtp", "skin", "covertype", "musk", "optdigits"])
    n_cap: int = 5000
    seed: int = 42
    out_dir: str = "revision/experiments/validation"


@dataclass
class Root:
    val: ValConfig = field(default_factory=ValConfig)


ConfigStore.instance().store(name="val", node=Root)


def load(name, n_cap, seed):
    from data_loader import load_dataset
    X, y = load_dataset(name)
    if len(X) > n_cap:
        idx = np.sort(np.random.default_rng(seed).choice(len(X), n_cap, replace=False))
        X, y = X[idx], y[idx]
    return np.asarray(X, float), y


def original_adod_scores(X, repo):
    """Run the authors' ADOD.fit pipeline (adod.py) with exact CPU k-NN."""
    import numba
    numba.prange = range
    sys.modules.setdefault("faiss", types.ModuleType("faiss"))
    sys.path.insert(0, repo)
    import importlib
    A = importlib.import_module("adod")
    from sklearn.neighbors import NearestNeighbors
    import scipy.stats as st
    n = len(X)
    perp = int(np.sqrt(n)) * 2
    k = min(n, 3 * perp, 2048)
    d, idx = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
    sq = (d ** 2).astype(np.float32)
    sigma = A.binary_search_perplexity_knn(sq, perp)
    kd = np.sqrt(sq)
    b = st.norm.ppf(0.999, loc=0, scale=sigma)
    mut = A.compute_mutual_neighbor_knn(kd, idx, b)
    dens = A.compute_local_density(mut, b)
    return np.asarray(A.compute_outlier_scores_knn(dens, mut, kd, idx), float)


def cmp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    r = spearmanr(a[ok], b[ok]).statistic if ok.sum() > 2 and a[ok].std() > 0 and b[ok].std() > 0 else np.nan
    k = max(1, int(0.1 * len(a)))
    ta, tb = set(np.argsort(-a, kind="stable")[:k]), set(np.argsort(-b, kind="stable")[:k])
    return r, len(ta & tb) / k, float(np.std(a)), float(np.std(b))


@hydra.main(version_base=None, config_name="val")
def main(cfg: Root) -> None:
    import detectors_v2 as D
    import method_wrappers as W
    c = cfg.val
    out = PROJECT / c.out_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    todo = ([(n, None) for n in c.adod_datasets] + [("annthyroid", c.adod_extra_subsample)]
            + [(n, c.n_cap) for n in c.adod_large])
    for name, sub in todo:
        X, _ = load(name, sub or c.n_cap, c.seed)
        repo = Path(c.adod_repo) if Path(c.adod_repo).is_absolute() else PROJECT / c.adod_repo
        t = time.time(); ref = original_adod_scores(X, str(repo)); t_ref = time.time() - t
        t = time.time(); ours = D.adod(X); t_ours = time.time() - t
        r, top, sa, sb = cmp(ours, ref)
        rows.append(dict(check="ADOD port vs original adod.py", method="ADOD", dataset=name, n=len(X),
                         spearman=r, top10_overlap=top, max_abs_diff=float(np.max(np.abs(ours - ref))),
                         std_new=sa, std_ref=sb, t_ref_s=t_ref, t_new_s=t_ours))
        print(rows[-1], flush=True)
    if c.adod_only:
        prev = pd.read_csv(out / "reimpl_validation.csv")
        df = pd.concat([pd.DataFrame(rows), prev[prev.check != "ADOD port vs original adod.py"]], ignore_index=True)
        df.to_csv(out / "reimpl_validation.csv", index=False)
        print(f"wrote {out/'reimpl_validation.csv'} (ADOD rows replaced)")
        return
    pairs = ["ADOD", "EIF", "ABOD", "FeatureBagging", "HBOS", "Mahalanobis", "LOF", "KNN", "IForest",
             "ROD", "OCSVM", "COPOD", "ECOD"]
    for name in c.datasets:
        X, _ = load(name, c.n_cap, c.seed)
        for m in pairs:
            try:
                old = W.make_detector(m, contamination=0.1, random_state=c.seed).fit(X).decision_function(X)
                new = D.score(m, X, seed=c.seed)
                r, top, sa, sb = cmp(old, new)
                rows.append(dict(check="earlier implementation vs reference", method=m, dataset=name, n=len(X),
                                 spearman=r, top10_overlap=top, std_new=sb, std_ref=sa,
                                 old_constant=bool(sa < 1e-12)))
            except Exception as e:
                rows.append(dict(check="earlier implementation vs reference", method=m, dataset=name, n=len(X),
                                 error=f"{type(e).__name__}: {e}"[:200]))
            print(name, m, {k: rows[-1].get(k) for k in ("spearman", "top10_overlap", "error")}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "reimpl_validation.csv", index=False)
    print(f"wrote {out/'reimpl_validation.csv'}")


if __name__ == "__main__":
    sys.argv += ["hydra.run.dir=.", "hydra.output_subdir=null"]
    main()
