"""eif_direction_check.py — ROC-AUC of the submitted EIF wrapper vs the reference EIF (isotree) and IForest.

Labels are used here only to check the direction of the scores (an AUC far below 0.5 means the
scores rank normal records above anomalies). Output: revision/experiments/validation/eif_direction_auc.csv
Run: .venv/bin/python revision/code/eif_direction_check.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE))
from datasets_v2 import _legacy_scripts  # noqa: E402
sys.path.insert(0, str(_legacy_scripts(PROJECT)))


@dataclass
class Cfg:
    datasets: list = field(default_factory=lambda: ["glass", "breastw", "thyroid", "pendigits", "shuttle"])
    n_cap: int = 5000
    seed: int = 42


ConfigStore.instance().store(name="eif", node=Cfg)


@hydra.main(version_base=None, config_name="eif")
def main(c: Cfg) -> None:
    import datasets_v2 as DS
    import detectors_v2 as D
    import method_wrappers as W
    rows = []
    for name in c.datasets:
        X, y, _ = DS.load(name, c.n_cap, c.seed)
        old = W.make_detector("EIF", random_state=c.seed).fit(X).decision_function(X)
        rows.append(dict(dataset=name, auc_submitted_eif=roc_auc_score(y, old),
                         auc_isotree_eif=roc_auc_score(y, D.eif(X, seed=c.seed)),
                         auc_iforest=roc_auc_score(y, D.iforest(X, seed=c.seed))))
        print(rows[-1], flush=True)
    out = PROJECT / "revision" / "experiments" / "validation" / "eif_direction_auc.csv"
    pd.DataFrame(rows).round(3).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    sys.argv += ["hydra.run.dir=.", "hydra.output_subdir=null"]
    main()
