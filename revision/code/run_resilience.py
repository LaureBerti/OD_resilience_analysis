"""run_resilience.py — single runner for every resilience experiment of the PLOS ONE revision.

One unit of work = (experiment, dataset, method[, variant]). For each replicate r (seed 42 + r):
  S = simple random sample without replacement of n = max(2, floor(p N)) records of the reference set D;
  s_D = in-sample scores of the method fitted on D;  s_S = in-sample scores of the method fitted on S.
Recorded per replicate (all label-free):
  rho        Dice between the top-K records of S under s_S and under s_D restricted to S, K = max(1, floor(q n))
             (equal cardinalities; q = 0.10 unless stated). When several records share the K-th score the
             top-K set is not unique; rho is then the EXPECTED Dice when the tied records are chosen uniformly
             at random, independently for the two fits (exact, computed from flagging probabilities);
  rho_k{5,8,10,12,15}  the same at q = 5 … 15 % (threshold sensitivity, E4 Part B);
  rho_adapt  adaptive cut-off: O[S] = (top-floor(q N) of D under s_D) ∩ S, of size k_S, compared with the
             top-k_S records of S under s_S;
  rho_def1   unequal-size reading: the same O[S] (size k_S) vs the top-floor(q n) of S under s_S
             (quantifies the ceiling created by the random size of O[S]);
             for these two variants, with ties, the ratio of expected intersection to expected sizes is used;
  rho_tilde  Spearman correlation of s_S and s_D restricted to S;
  nonfinite_D / nonfinite_S  number of non-finite scores (e.g. ABOD on exact duplicates); they are set to
             the least-outlying finite score of the same vector before any metric is computed;
  degenerate True if either score vector is constant on S (all metrics then NaN, never silently kept);
  ties_S, ties_D  records tied with the k-th score (beyond the k-th itself).
Output: revision/experiments/<exp>/raw/<dataset>__<method>[__<variant>].csv — resumable; a unit is
skipped only if its file holds the expected number of parsed rows (completeness judged from data).

CPU-only, local. Examples:
  .venv/bin/python revision/code/run_resilience.py exp=E1
  .venv/bin/python revision/code/run_resilience.py exp=E1 run.methods='[ADOD]' run.datasets='[glass]' run.n_reps=2
  .venv/bin/python revision/code/run_resilience.py --write-config
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hydra
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import detectors_v2 as D
import datasets_v2 as DS

THRESHOLDS = (0.05, 0.08, 0.10, 0.12, 0.15)


@dataclass
class RunConfig:
    exp: str = "E1"
    methods: list = field(default_factory=list)
    datasets: list = field(default_factory=list)
    ps: list = field(default_factory=list)
    n_reps: int = 0
    n_cap: int = -1
    q: float = 0.10
    n_jobs: int = 12
    base_seed: int = 42
    out_root: str = "revision/experiments"
    variants: list = field(default_factory=list)


@dataclass
class Root:
    exp: str = "E1"
    run: RunConfig = field(default_factory=RunConfig)


ConfigStore.instance().store(name="run", node=Root)

ALL16 = DS.CORE_DATASETS
EXPERIMENTS = {
    "E1": dict(methods=D.ALL_METHODS, datasets=ALL16, ps=[0.10], n_reps=50, n_cap=5000),
    "E1b": dict(methods=D.ALL_METHODS, datasets=ALL16, ps=[0.01, 0.05, 0.10], n_reps=20, n_cap=5000),
    "E2": dict(methods=["ThreeSigma", "MAD", "COPOD", "ECOD", "IForest", "EIF", "LOF", "KNN"],
               datasets=DS.E2_DATASETS, ps=[0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
               n_reps=30, n_cap=5000),
    **{f"E2b{Nr}": dict(methods=["ThreeSigma", "MAD", "COPOD", "ECOD", "IForest", "LOF", "KNN", "ABOD"],
                        datasets=["optdigits", "satellite", "pendigits", "annthyroid", "shuttle", "smtp", "skin",
                                  "covertype", "pageblocks", "satimage2", "mnist", "mammography", "magic_gamma",
                                  "campaign", "celeba", "fraud", "http"],
                        ps=[0.05, 0.10, 0.20], n_reps=30, n_cap=Nr) for Nr in (1000, 2000, 5000)},
    "E3": dict(methods=["LOF", "KNN"], datasets=["annthyroid", "glass", "letter", "lympho", "optdigits",
                                                   "skin", "smtp", "thyroid"],
               ps=[0.05, 0.10, 0.20], n_reps=30, n_cap=5000,
               variants=[f"n_neighbors={k}" for k in (3, 5, 10, 15, 20)]),
    "E5": dict(methods=["ThreeSigma", "BoxPlot", "MAD", "ChiSquare", "Mahalanobis", "COPOD", "ECOD",
                        "HBOS", "IForest", "EIF"],
               datasets=["covertype", "skin", "smtp"], ps=[0.01, 0.05, 0.10], n_reps=10, n_cap=0),
    "X5": dict(methods=["IForest", "EIF"], datasets=["letter", "thyroid", "optdigits", "satellite",
                                                     "pendigits", "annthyroid", "shuttle", "smtp"],
               ps=[0.10], n_reps=20, n_cap=5000,
               variants=[f"n_estimators={t}" for t in (50, 100, 200, 500, 1000)]),
    "X6": dict(methods=D.ALL_METHODS, datasets=[f"gauss_d{d}" for d in (2, 10, 50, 100)],
               ps=[0.10], n_reps=30, n_cap=0),
    "X7": dict(methods=D.ALL_METHODS,
               datasets=[f"{t}_{b}" for t in ("local", "global", "clustered")
                         for b in ("annthyroid", "pendigits", "satellite", "thyroid")],
               ps=[0.10], n_reps=30, n_cap=5000),
}


def topk(s: np.ndarray, k: int) -> np.ndarray:
    """Probability that each record is among the top-k when ties at the k-th value are broken uniformly
    at random: 1 above the k-th value, (k - #above) / #tied at it, 0 below."""
    pr = np.zeros(len(s), float)
    if k <= 0:
        return pr
    k = min(k, len(s))
    v = np.sort(s)[::-1][k - 1]
    above, tied = s > v, s == v
    pr[above] = 1.0
    pr[tied] = (k - above.sum()) / tied.sum()
    return pr


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Expected Dice of two independently tie-broken sets given flagging probabilities a, b
    (exact when both sets have fixed sizes; ratio of expectations otherwise)."""
    den = a.sum() + b.sum()
    return float(2.0 * (a * b).sum() / den) if den > 0 else np.nan


def ties_at(s: np.ndarray, k: int) -> int:
    if k <= 0 or k > len(s):
        return 0
    v = np.sort(s)[::-1][k - 1]
    return int(np.sum(s == v) - 1)


def clean(s: np.ndarray):
    bad = ~np.isfinite(s)
    if bad.any():
        s = s.copy()
        s[bad] = np.min(s[~bad]) if (~bad).any() else 0.0
    return s, int(bad.sum())


def metrics(s_S, s_DS, s_D, idx, q, N):
    n = len(s_S)
    out = {}
    deg = (np.std(s_S) == 0) or (np.std(s_DS) == 0)
    out["degenerate"] = bool(deg)
    k = max(1, int(np.floor(q * n)))
    out["k"] = k
    out["ties_S"], out["ties_D"] = ties_at(s_S, k), ties_at(s_DS, k)
    if deg:
        for c in ["rho", "rho_adapt", "rho_def1", "rho_tilde"] + [f"rho_k{int(t*100)}" for t in THRESHOLDS]:
            out[c] = np.nan
        out["k_S"] = np.nan
        return out
    out["rho"] = dice(topk(s_S, k), topk(s_DS, k))
    for t in THRESHOLDS:
        kt = max(1, int(np.floor(t * n)))
        out[f"rho_k{int(round(t*100))}"] = dice(topk(s_S, kt), topk(s_DS, kt))
    OD = topk(s_D, max(1, int(np.floor(q * N))))
    OS_def = OD[idx]
    kS = int(round(OS_def.sum()))
    out["k_S"] = kS
    out["rho_adapt"] = dice(topk(s_S, kS), OS_def) if kS > 0 else np.nan
    out["rho_def1"] = dice(topk(s_S, k), OS_def)
    out["rho_tilde"] = float(spearmanr(s_S, s_DS).statistic)
    return out


def parse_variant(v: str) -> dict:
    if not v:
        return {}
    kv = {}
    for part in v.split(","):
        a, b = part.split("=")
        kv[a] = int(b) if b.lstrip("-").isdigit() else float(b) if b.replace(".", "", 1).isdigit() else b
    return kv


def unit_path(root: Path, ds: str, m: str, variant: str) -> Path:
    tag = f"__{variant.replace('=', '-').replace(',', '_')}" if variant else ""
    return root / "raw" / f"{ds}__{m}{tag}.csv"


def run_unit(exp, ds, m, variant, ps, n_reps, n_cap, q, base_seed, root):
    path = unit_path(root, ds, m, variant)
    expected = len(ps) * n_reps
    if path.exists():
        try:
            done = pd.read_csv(path)
            if len(done) == expected and done["rep"].notna().all():
                return f"skip {path.name}"
        except Exception:
            pass
    X, y, meta = DS.load(ds, n_cap=n_cap, seed=base_seed)
    N = len(X)
    kw = parse_variant(variant)
    stochastic = m in D.STOCHASTIC
    cache_D = None
    rows = []
    for p in ps:
        n = max(2, int(np.floor(p * N)))
        for r in range(n_reps):
            seed = base_seed + r
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(N, size=n, replace=False))
            row = dict(experiment=exp, dataset=ds, method=m, method_class=D.METHOD_CLASS[m],
                       submitted_class=D.SUBMITTED_CLASS[m], variant=variant, p=p, rep=r, seed=seed,
                       N=N, N_orig=meta["N_orig"], d=X.shape[1], tier=meta["tier"], n=n, error="")
            try:
                t0 = time.perf_counter()
                if stochastic or cache_D is None:
                    s_D = D.score(m, X, seed=seed, **kw)
                    if not stochastic:
                        cache_D = s_D
                else:
                    s_D = cache_D
                t1 = time.perf_counter()
                s_S = D.score(m, X[idx], seed=seed, **kw)
                t2 = time.perf_counter()
                s_D, row["nonfinite_D"] = clean(s_D)
                s_S, row["nonfinite_S"] = clean(s_S)
                row.update(metrics(s_S, s_D[idx], s_D, idx, q, N))
                row.update(t_fit_D_s=t1 - t0 if (stochastic or r == 0) else np.nan, t_fit_S_s=t2 - t1)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"[:300]
                row["degenerate"] = np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    if (df["error"] != "").all():
        path.with_suffix(".FAILED.txt").write_text(df["error"].iloc[0])
        return f"FAILED {path.name}: {df['error'].iloc[0]}"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    nd = int(df["degenerate"].fillna(False).astype(bool).sum())
    ne = int((df["error"] != "").sum())
    return f"done {path.name}: rho={np.nanmean(df.get('rho', np.nan)):.3f} deg={nd} err={ne}"


@hydra.main(version_base=None, config_name="run")
def main(cfg: Root) -> None:
    c = cfg.run
    exp = cfg.exp
    spec = EXPERIMENTS[exp]
    methods = list(c.methods) or spec["methods"]
    datasets = list(c.datasets) or spec["datasets"]
    ps = list(c.ps) or spec["ps"]
    n_reps = c.n_reps or spec["n_reps"]
    n_cap = spec["n_cap"] if c.n_cap < 0 else c.n_cap
    variants = list(c.variants) or spec.get("variants", [""])
    root = PROJECT / c.out_root / exp
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / f"config_{time.strftime('%Y%m%d_%H%M%S')}.yaml").write_text(OmegaConf.to_yaml(cfg))
    units = [(exp, ds, m, v, ps, n_reps, n_cap, c.q, c.base_seed, root)
             for ds in datasets for m in methods for v in variants]
    first = run_unit(*units[0])
    print(first, flush=True)
    if first.startswith("FAILED"):
        raise SystemExit(f"first unit failed — aborting {exp}: {first}")
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=c.n_jobs, verbose=0, batch_size=1)(
        delayed(_safe)(*u) for u in units[1:])
    for r in res:
        print(r, flush=True)
    n_fail = sum(r.startswith("FAILED") for r in res)
    print(f"{exp}: {len(units)} units, {n_fail} failed", flush=True)


def _safe(*a):
    try:
        return run_unit(*a)
    except Exception:
        return "FAILED " + traceback.format_exc(limit=2).replace("\n", " | ")[:400]


if __name__ == "__main__":
    if "--write-config" in sys.argv:
        (PROJECT / "conf").mkdir(exist_ok=True)
        (PROJECT / "conf" / "run_resilience.yaml").write_text(OmegaConf.to_yaml(OmegaConf.structured(Root)))
        print("wrote conf/run_resilience.yaml"); sys.exit(0)
    sys.argv += ["hydra.run.dir=.", "hydra.output_subdir=null"]
    main()
