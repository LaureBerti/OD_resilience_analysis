"""make_si_tables.py — Supporting Information tables S1-S3 (CSV) from the result files.

S1_Table.csv  implementation check (validation/reimpl_validation.csv)
S2_Table.csv  per-dataset E1/E4 results (E1/raw): rho, rho_tilde, rho with the two other reference-set
              definitions, rho at q = 5..15 %, number of excluded (degenerate) replicates
S3_Table.csv  characteristics and sources of all 29 datasets (experiments/dataset_characteristics.csv)
Run: .venv/bin/python analysis/code/make_si_tables.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECT = HERE.parents[1]
EXP = ROOT / "experiments"
sys.path.insert(0, str(HERE))
import datasets_v2 as DS


@dataclass
class Cfg:
    out_dir: str = "SI"


ConfigStore.instance().store(name="si", node=Cfg)


@hydra.main(version_base=None, config_name="si")
def main(c: Cfg) -> None:
    out = ROOT / c.out_dir; out.mkdir(parents=True, exist_ok=True)
    v = pd.read_csv(EXP / "validation" / "reimpl_validation.csv")
    v["comparison"] = v["check"].map({"ADOD port vs original adod.py": "ADOD: our implementation vs authors' code",
                                      "earlier implementation vs reference": "earlier implementation vs reference"})
    cols = ["comparison", "method", "dataset", "n", "spearman", "top10_overlap", "std_ref", "std_new"]
    s1 = v[cols].rename(columns={"n": "records", "spearman": "spearman_rank_correlation",
                                 "top10_overlap": "overlap_of_top_10pct", "std_ref": "sd_scores_earlier_or_authors",
                                 "std_new": "sd_scores_reference_or_ours"})
    a = v[v["check"] == "ADOD port vs original adod.py"]
    a = a[["dataset", "n", "spearman", "top10_overlap", "max_abs_diff"]].rename(
        columns={"n": "records", "spearman": "spearman_rank_correlation", "top10_overlap": "overlap_of_top_10pct",
                 "max_abs_diff": "largest_absolute_score_difference"})
    a["spearman_rank_correlation"] = a["spearman_rank_correlation"].round(4)
    a.to_csv(out / "S1_Table.csv", index=False, float_format="%.4g")
    att = EXP / "validation"; att.mkdir(parents=True, exist_ok=True)
    r1 = s1[~s1.comparison.str.startswith("ADOD")]
    r1.assign(**{c: r1[c].round(4) for c in ["spearman_rank_correlation", "overlap_of_top_10pct"]}).to_csv(
        att / "implementations_vs_reference.csv", index=False, float_format="%.4g")
    summ = r1.groupby("method").agg(n_datasets_spearman=("spearman_rank_correlation", "count"),
                                    median_spearman=("spearman_rank_correlation", "median"),
                                    min_spearman=("spearman_rank_correlation", "min"),
                                    n_datasets_top10=("overlap_of_top_10pct", "count"),
                                    median_top10=("overlap_of_top_10pct", "median"),
                                    min_top10=("overlap_of_top_10pct", "min")).sort_values("median_spearman")
    summ.round(4).to_csv(att / "implementations_vs_reference_summary.csv")

    df = pd.concat([pd.read_csv(f) for f in sorted((EXP / "E1" / "raw").glob("*.csv"))], ignore_index=True)
    g = df.groupby(["dataset", "tier", "method", "method_class"]).agg(
        replicates=("rep", "size"),
        excluded_degenerate=("degenerate", lambda s: int(s.fillna(False).astype(bool).sum())),
        rho=("rho", "mean"), rho_sd=("rho", "std"), rho_tilde=("rho_tilde", "mean"),
        rho_adaptive_cutoff=("rho_adapt", "mean"), rho_unequal_size=("rho_def1", "mean"),
        rho_q5=("rho_k5", "mean"), rho_q8=("rho_k8", "mean"), rho_q12=("rho_k12", "mean"),
        rho_q15=("rho_k15", "mean"), nonfinite_scores_full_fit=("nonfinite_D", "max")).reset_index()
    g.round(4).to_csv(out / "S2_Table.csv", index=False)

    t = pd.read_csv(EXP / "dataset_characteristics.csv")
    src = {n: "ADBench classical" for n in DS.CORE_DATASETS}
    src.update({n: ("ODDS via ADOD repository" if f.endswith("_ODDS") else "ADBench via ADOD repository")
                for n, f in DS.EXTRA.items()})
    t["source"] = t.dataset.map(src)
    t.to_csv(out / "S3_Table.csv", index=False)
    print("wrote", sorted(p.name for p in out.glob("S*_Table.csv")))


if __name__ == "__main__":
    sys.argv += ["hydra.run.dir=.", "hydra.output_subdir=null"]
    main()
