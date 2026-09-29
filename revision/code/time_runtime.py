"""time_runtime.py — runtime of fitting + scoring on the full dataset vs on a sample (E5 runtime).

Run ONLY on an otherwise idle machine (single process, BLAS threads = 1); the per-replicate timings
recorded during E5 were taken while other experiments shared the CPU and are not used.
For each method and dataset: median over `repeats` of the wall time of score(method, D) and of
score(method, S) for S of size floor(pN), p in ps. Output: revision/experiments/E5/tables/runtime_idle.csv
Run: .venv/bin/python revision/code/time_runtime.py
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hydra
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))


@dataclass
class Cfg:
    datasets: list = field(default_factory=lambda: ["covertype", "skin", "smtp"])
    methods: list = field(default_factory=lambda: ["ThreeSigma", "BoxPlot", "MAD", "ChiSquare", "Mahalanobis",
                                                   "COPOD", "ECOD", "HBOS", "IForest", "EIF"])
    ps: list = field(default_factory=lambda: [0.01, 0.05, 0.10])
    repeats: int = 5
    seed: int = 42


ConfigStore.instance().store(name="tm", node=Cfg)


def wall(f):
    t = time.perf_counter(); f(); return time.perf_counter() - t


@hydra.main(version_base=None, config_name="tm")
def main(c: Cfg) -> None:
    import datasets_v2 as DS
    import detectors_v2 as D
    rows = []
    for ds in c.datasets:
        X, _, _ = DS.load(ds, 0, c.seed)
        N = len(X)
        for m in c.methods:
            tD = np.median([wall(lambda: D.score(m, X, seed=c.seed + r)) for r in range(c.repeats)])
            for p in c.ps:
                idx = np.sort(np.random.default_rng(c.seed).choice(N, int(p * N), replace=False))
                tS = np.median([wall(lambda: D.score(m, X[idx], seed=c.seed + r)) for r in range(c.repeats)])
                rows.append(dict(dataset=ds, N=N, method=m, p=p, n=len(idx), t_full_s=tD, t_sample_s=tS,
                                 speedup=tD / tS))
                print(rows[-1], flush=True)
    out = PROJECT / "revision" / "experiments" / "E5" / "tables" / "runtime_idle.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    sys.argv += ["hydra.run.dir=.", "hydra.output_subdir=null"]
    main()
