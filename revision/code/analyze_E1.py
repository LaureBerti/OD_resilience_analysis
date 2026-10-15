"""analyze_E1.py — every E1 / E4 number of the revised paper, computed from revision/experiments/E1/raw.

Outputs (revision/experiments/E1/tables/):
  completeness.csv          rows per (dataset, method) vs expected, degenerate / error / non-finite counts
  table_e1_tiers.csv/.tex   mean ± sd of per-dataset mean rho by method and tier (tiers of the dataset table) + All
  class_by_dataset.csv      class means per dataset (unit of analysis for the tests)
  tests.csv                 Friedman (datasets as blocks, classes I-IV), Kendall's W, mixed model, 6 pairwise
                            Wilcoxon signed-rank tests over datasets with Holm correction
  e4_rho_vs_rhotilde.csv    per method: mean rho, mean rho_tilde, Wilcoxon (dataset level) rho_tilde > rho
  e4_thresholds.csv         mean rho at q = 5,8,10,12,15 % per method, and its range
  r2_variants.csv           rho vs rho_adapt vs rho_def1 (cardinality of the reference set)
  headline.csv              class means and the (1 - rho_IV)/(1 - rho_I) ratio at p = 10 %
CPU-only, seconds. Run: .venv/bin/python revision/code/analyze_E1.py
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
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import detectors_v2 as D

CLASSES = ["I", "II", "III", "IV"]
ORDER = ["ThreeSigma", "BoxPlot", "MAD", "ChiSquare", "Mahalanobis", "ROD", "COPOD", "ECOD", "HBOS",
         "IForest", "EIF", "LOF", "KNN", "ABOD", "FeatureBagging", "ADOD", "KMeans", "OCSVM"]


@dataclass
class Cfg:
    exp_dir: str = "revision/experiments/E1"
    expected_reps: int = 50


ConfigStore.instance().store(name="a1", node=Cfg)


def holm(pvals):
    p = np.asarray(pvals, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


@hydra.main(version_base=None, config_name="a1")
def main(c: Cfg) -> None:
    root = PROJECT / c.exp_dir
    out = root / "tables"; out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(f) for f in sorted((root / "raw").glob("*.csv"))], ignore_index=True)
    df["error"] = df["error"].fillna("")
    comp = df.groupby(["dataset", "method"]).agg(
        rows=("rep", "size"), degenerate=("degenerate", lambda s: int(s.fillna(False).astype(bool).sum())),
        errors=("error", lambda s: int((s != "").sum())), nonfinite_D=("nonfinite_D", "max"),
        rho_nan=("rho", lambda s: int(s.isna().sum()))).reset_index()
    comp["complete"] = comp["rows"] == c.expected_reps
    comp.to_csv(out / "completeness.csv", index=False)
    print("units:", len(comp), " incomplete:", int((~comp.complete).sum()),
          " with degenerate reps:", int((comp.degenerate > 0).sum()), " with errors:", int((comp.errors > 0).sum()))

    dm = df.groupby(["dataset", "tier", "method", "method_class"], as_index=False)[
        ["rho", "rho_tilde", "rho_adapt", "rho_def1"] + [f"rho_k{t}" for t in (5, 8, 10, 12, 15)]].mean()

    rows = []
    for m in ORDER:
        r = {"class": D.METHOD_CLASS[m], "method": m}
        for t in ["T1", "T2", "T3", "T4", "All"]:
            s = dm[(dm.method == m) & ((dm.tier == t) if t != "All" else True)]["rho"]
            r[f"{t}_mean"], r[f"{t}_sd"], r[f"{t}_n"] = s.mean(), s.std(ddof=1), s.notna().sum()
        rows.append(r)
    tab = pd.DataFrame(rows)
    for cl in CLASSES + ["G"]:
        r = {"class": cl, "method": "Group G mean" if cl == "G" else f"Class {cl} mean"}
        for t in ["T1", "T2", "T3", "T4", "All"]:
            x = dm[(dm.method_class == cl) & ((dm.tier == t) if t != "All" else True)]
            s = x.groupby("dataset")["rho"].mean()
            r[f"{t}_mean"], r[f"{t}_sd"], r[f"{t}_n"] = s.mean(), s.std(ddof=1), s.notna().sum()
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "table_e1_tiers.csv", index=False)
    with open(out / "table_e1_tiers.tex", "w") as f:
        for _, r in tab.iterrows():
            cells = " & ".join("---" if not np.isfinite(r[f"{t}_mean"]) else
                               (f"{r[f'{t}_mean']:.2f}" + (f"$\\pm${r[f'{t}_sd']:.2f}" if np.isfinite(r[f'{t}_sd']) else ""))
                               for t in ["T1", "T2", "T3", "T4", "All"])
            f.write(f"{r['class']} & {r['method']} & {cells} \\\\\n")

    cm = dm[dm.method_class.isin(CLASSES)].groupby(["dataset", "method_class"])["rho"].mean().unstack()
    cm.to_csv(out / "class_by_dataset.csv")
    cm = cm.dropna()
    tests = []
    fr = stats.friedmanchisquare(*[cm[k] for k in CLASSES])
    nb, kk = len(cm), len(CLASSES)
    W = fr.statistic / (nb * (kk - 1))
    tests.append(dict(test="Friedman (classes I-IV, datasets as blocks)", n_blocks=nb, statistic=fr.statistic,
                      df=kk - 1, p=fr.pvalue, effect=f"Kendall W={W:.3f}"))
    pairs = [(a, b) for i, a in enumerate(CLASSES) for b in CLASSES[i + 1:]]
    pw = [stats.wilcoxon(cm[a], cm[b]) for a, b in pairs]
    adj = holm([w.pvalue for w in pw])
    for (a, b), w, pa in zip(pairs, pw, adj):
        tests.append(dict(test=f"Wilcoxon signed-rank {a} vs {b} (datasets paired)", n_blocks=nb,
                          statistic=w.statistic, p=w.pvalue, p_holm=pa,
                          effect=f"median diff={np.median(cm[a]-cm[b]):+.3f}; mean diff={np.mean(cm[a]-cm[b]):+.3f}"))
    try:
        import statsmodels.formula.api as smf
        x = dm[dm.method_class.isin(CLASSES)].dropna(subset=["rho"]).copy()
        x = x.reset_index(drop=True)
        md = smf.mixedlm("rho ~ C(method_class, Treatment('I'))", x, groups=np.ones(len(x)),
                         vc_formula={"dataset": "0 + C(dataset)", "method": "0 + C(method)"}).fit(reml=True)
        for name in md.params.index:
            if "method_class" in name:
                tests.append(dict(test=f"Mixed model: {name} (random: dataset, method)", n_blocks=nb,
                                  statistic=md.params[name], p=md.pvalues[name],
                                  effect=f"95% CI [{md.conf_int().loc[name,0]:.3f}, {md.conf_int().loc[name,1]:.3f}]"))
    except Exception as e:
        tests.append(dict(test=f"Mixed model failed: {e}"))
    pd.DataFrame(tests).to_csv(out / "tests.csv", index=False)

    e4 = []
    for m in ORDER:
        s = dm[dm.method == m].dropna(subset=["rho", "rho_tilde"])
        w = stats.wilcoxon(s.rho_tilde, s.rho) if len(s) > 5 else None
        e4.append(dict(method=m, method_class=D.METHOD_CLASS[m], n_datasets=len(s), rho=s.rho.mean(),
                       rho_tilde=s.rho_tilde.mean(), diff=(s.rho_tilde - s.rho).mean(),
                       wilcoxon_p=w.pvalue if w else np.nan))
    e4 = pd.DataFrame(e4); e4["p_holm"] = holm(e4.wilcoxon_p.fillna(1).values)
    e4.to_csv(out / "e4_rho_vs_rhotilde.csv", index=False)

    th = dm.groupby("method")[[f"rho_k{t}" for t in (5, 8, 10, 12, 15)]].mean()
    th["range"] = th.max(1) - th.min(1)
    th.loc[ORDER].to_csv(out / "e4_thresholds.csv")

    v = dm.groupby("method")[["rho", "rho_adapt", "rho_def1"]].mean().loc[ORDER]
    v["max_abs_diff_rho_adapt"] = dm.assign(d=(dm.rho - dm.rho_adapt).abs()).groupby("method").d.max().loc[ORDER]
    v.to_csv(out / "r2_variants.csv")

    names = {"ThreeSigma": "$3\\sigma$", "ChiSquare": "$\\chi^2$", "FeatureBagging": "Feature Bagging",
             "KMeans": "$k$-means", "Mahalanobis": "Mahalanobis"}
    def cell(m, sd):
        return "---" if not np.isfinite(m) else f"{m:.2f}" + (f"$\\pm${sd:.2f}" if np.isfinite(sd) else "")
    lines = {"methods": [], "classes": []}
    for _, r in tab.iterrows():
        key = "classes" if r["method"].startswith(("Class ", "Group ")) else "methods"
        nm = (("Group G mean" if r["class"] == "G" else f"Class {r['class']} mean") if key == "classes" else names.get(r["method"], r["method"]))
        lines[key].append(f"{r['class']} & {nm} & " + " & ".join(cell(r[f'{t}_mean'], r[f'{t}_sd'])
                                                                 for t in ["T1", "T2", "T3", "T4", "All"]) + " \\\\")
    for key, L in lines.items():
        (out / f"table_e1_tiers_{key}.tex").write_text("\n".join(L) + "\n")
    th = dm.groupby("method")[[f"rho_k{t}" for t in (5, 8, 10, 12, 15)]].mean()
    rng = (th.max(1) - th.min(1))
    e4t = ["\\begin{tabular}{llccccc}", "\\hline",
           "\\textbf{Class} & \\textbf{Method} & $\\bar\\rho$ & $\\bar{\\tilde\\rho}$ & $\\bar{\\tilde\\rho}-\\bar\\rho$ & $p$ (Holm) & $\\Delta_q$ \\\\", "\\hline"]
    for _, r in e4.iterrows():
        pv = "$<0.001$" if r.p_holm < 0.001 else f"{r.p_holm:.3f}"
        e4t.append(f"{r.method_class} & {names.get(r.method, r.method)} & {r.rho:.2f} & {r.rho_tilde:.2f} & "
                   f"{r['diff']:+.2f} & {pv} & {rng[r.method]:.2f} \\\\")
    e4t += ["\\hline", "\\end{tabular}"]
    (out / "table_e4.tex").write_text("\n".join(e4t) + "\n")
    rt = dm[dm.method_class.isin(CLASSES)].groupby(["dataset", "method_class"]).rho_tilde.mean().unstack().mean()
    pd.DataFrame([rt]).to_csv(out / "class_means_rho_tilde.csv", index=False)
    print("rho_tilde class means:", rt.round(3).to_dict())

    ca = dm[dm.method_class.isin(CLASSES)].groupby(["dataset", "method_class"]).rho.mean().unstack().mean()
    head = pd.DataFrame([dict(**{f"class_{k}": ca[k] for k in CLASSES},
                              ratio_IV_over_I=(1 - ca["IV"]) / (1 - ca["I"]))])
    head.to_csv(out / "headline.csv", index=False)
    print(tab.round(2).to_string()); print(pd.DataFrame(tests).to_string()); print(e4.round(3).to_string())
    print(v.round(3).to_string()); print(head.round(3).to_string())


if __name__ == "__main__":
    sys.argv += ["hydra.run.dir=.", "hydra.output_subdir=null"]
    main()
