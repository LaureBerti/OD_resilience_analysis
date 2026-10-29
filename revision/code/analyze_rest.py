"""analyze_rest.py — tables for E1b, E2, E2b, E3, E5, E6 (=X5), E7 (=X6), E8 (=X7).

The numbers of the Results section outside E1/E4 come from the CSVs written here
(revision/experiments/<exp>/tables/), except those listed in paper_numbers.py. Dataset = unit of analysis; replicates are averaged first.
Run: .venv/bin/python revision/code/analyze_rest.py [a.only='[E2,E3]']
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from scipy import stats

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
EXP = PROJECT / "revision" / "experiments"
CLASSES = ["I", "II", "III", "IV"]


@dataclass
class Cfg:
    only: list = field(default_factory=list)
    min_n_rate: int = 50
    min_n_forest: int = 512


ConfigStore.instance().store(name="ar", node=Cfg)


def load(exp: str) -> pd.DataFrame | None:
    files = sorted((EXP / exp / "raw").glob("*.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["error"] = df["error"].fillna("")
    df["variant"] = df["variant"].fillna("")
    return df


def out(exp):
    o = EXP / exp / "tables"; o.mkdir(parents=True, exist_ok=True); return o


def health(df, exp):
    h = df.groupby(["dataset", "method", "variant"]).agg(
        rows=("rep", "size"), degenerate=("degenerate", lambda s: int(s.fillna(False).astype(bool).sum())),
        errors=("error", lambda s: int((s != "").sum()))).reset_index()
    h.to_csv(out(exp) / "completeness.csv", index=False)
    print(f"[{exp}] units={len(h)} rows={len(df)} degenerate_units={(h.degenerate>0).sum()} "
          f"error_units={(h.errors>0).sum()}")


def mixed_slope(d, y, x, group="dataset"):
    """Random intercept and slope per dataset; if that fit fails OR does not converge, fall back to a
    random intercept only (recorded in the returned model label)."""
    import warnings as _w
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    m, label = None, "random intercept + slope"
    try:
        with _w.catch_warnings():
            _w.simplefilter("error", ConvergenceWarning)
            m = smf.mixedlm(f"{y} ~ {x}", d, groups=d[group], re_formula=f"~{x}").fit(reml=True)
        if not m.converged:
            m = None
    except Exception:
        m = None
    if m is None:
        label = "random intercept"
        m = smf.mixedlm(f"{y} ~ {x}", d, groups=d[group]).fit(reml=True)
    ci = m.conf_int().loc[x]
    return m.params[x], ci[0], ci[1], m.pvalues[x], label


def e1b(c):
    df = load("E1b")
    if df is None: return
    health(df, "E1b")
    excl = df[df.n < 10].groupby(["dataset", "p"]).n.first()
    excl.to_csv(out("E1b") / "excluded_small_n.csv")
    df = df[df.n >= 10]
    dm = df.groupby(["p", "dataset", "method", "method_class"], as_index=False).rho.mean()
    cm = dm[dm.method_class.isin(CLASSES + ["G"])].groupby(["p", "dataset", "method_class"]).rho.mean().unstack()
    rows = []
    for p, t in cm.groupby(level=0):
        t = t.droplevel(0)
        tt = t[CLASSES].dropna()
        fr = stats.friedmanchisquare(*[tt[k] for k in CLASSES])
        rows.append(dict(p=p, **{f"class_{k}": t[k].mean() for k in CLASSES + ["G"]}, n_datasets=len(tt),
                         friedman_chi2=fr.statistic, friedman_p=fr.pvalue,
                         IV_lowest_in_n_datasets=int((tt["IV"] < tt[["I", "II", "III"]].min(1)).sum())))
    pd.DataFrame(rows).to_csv(out("E1b") / "class_by_p.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())


def e2(c):
    df = load("E2")
    if df is None: return
    health(df, "E2")
    g = df.groupby(["method", "dataset", "p"], as_index=False).agg(rho=("rho", "mean"), n=("n", "first"),
                                                                     N=("N", "first"))
    g.to_csv(out("E2") / "rho_by_n.csv", index=False)
    g["y"] = np.log(1 - g.rho.clip(upper=1 - 1e-9))
    g["logn"] = np.log(g.n)
    g["logfpc"] = np.log(g.n / (1 - g.p))
    rows = []
    for m, t in g.groupby("method"):
        lim = c.min_n_forest if m in ("IForest", "EIF") else c.min_n_rate
        t = t[(t.n >= lim) & (t.rho < 1)]
        if t.dataset.nunique() < 3: continue
        for x in ("logn", "logfpc"):
            sl = []
            for _, u in t.groupby("dataset"):
                if len(u) >= 3 and u[x].nunique() >= 3:
                    sl.append(-np.polyfit(u[x], u["y"], 1)[0])
            sl = np.array(sl); k = len(sl)
            se = sl.std(ddof=1) / np.sqrt(k); tq = stats.t.ppf(0.975, k - 1)
            rows.append(dict(method=m, regressor=x, kappa=sl.mean(), ci_low=sl.mean() - tq * se,
                             ci_high=sl.mean() + tq * se, median=np.median(sl), n_datasets=k,
                             n_cells=len(t), min_n=lim, model="two-stage: per-dataset OLS slope, t-interval over datasets"))
    r = pd.DataFrame(rows); r.to_csv(out("E2") / "rates.csv", index=False)
    print(r.round(3).to_string())
    f = g[g.method.isin(["IForest", "EIF"])].copy()
    f["n_bin"] = pd.cut(f.n, [0, 32, 128, 256, 512, 1024, 2048, 5000])
    fd = f.groupby(["method", "n_bin", "dataset"], observed=True).rho.mean().reset_index()
    fb = fd.groupby(["method", "n_bin"], observed=True).rho.agg(["mean", "min", "max", "count"]).reset_index()
    fb.to_csv(out("E2") / "forest_rho_by_n.csv", index=False)
    print(fb.round(3).to_string())


def two_stage(t, xs, min_cells=4):
    import statsmodels.api as sm
    coefs = []
    for _, u in t.groupby("dataset"):
        if len(u) < min_cells or any(u[x].nunique() < 2 for x in xs):
            continue
        f = sm.OLS(u["y"], sm.add_constant(u[list(xs)])).fit()
        coefs.append(f.params[list(xs)])
    C = pd.DataFrame(coefs)
    res = {}
    for x in xs:
        v = C[x].values; k = len(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(k)
        q = stats.t.ppf(0.975, k - 1)
        res[x] = dict(coef=m, lo=m - q * se, hi=m + q * se, p=2 * stats.t.sf(abs(m / se), k - 1), k=k)
    return res


def e2b(c):
    parts = []
    for Nr in (1000, 2000, 5000):
        d = load(f"E2b{Nr}")
        if d is not None:
            health(d, f"E2b{Nr}"); d["N_ref"] = Nr; parts.append(d)
    if not parts: return
    df = pd.concat(parts)
    g = df.groupby(["method", "method_class", "dataset", "p", "N_ref"], as_index=False).agg(
        rho=("rho", "mean"), n=("n", "first"))
    g["y"] = np.log(1 - g.rho.clip(upper=1 - 1e-9)); g["logn"] = np.log(g.n); g["logp"] = np.log(g.p)
    import statsmodels.formula.api as smf
    rows = []
    for m, t in g.groupby("method"):
        t = t[t.rho < 1]
        ts = two_stage(t, ("logn", "logp"))
        mm = smf.mixedlm("y ~ logn + logp", t, groups=t["dataset"]).fit(reml=True)
        ci = mm.conf_int()
        a, b = ts["logn"], ts["logp"]
        rows.append(dict(method=m, method_class=t.method_class.iloc[0], model="two-stage: per-dataset OLS, t-interval over datasets",
                         coef_logn=a["coef"], n_lo=a["lo"], n_hi=a["hi"], ci_n=f"[{a['lo']:.3f}, {a['hi']:.3f}]",
                         p_logn=a["p"], coef_logp=b["coef"], p_lo=b["lo"], p_hi=b["hi"],
                         ci_p=f"[{b['lo']:.3f}, {b['hi']:.3f}]", p_logp=b["p"], n_datasets=a["k"],
                         mixed_coef_logn=mm.params["logn"], mixed_ci_n=f"[{ci.loc['logn',0]:.3f}, {ci.loc['logn',1]:.3f}]",
                         mixed_coef_logp=mm.params["logp"], mixed_ci_p=f"[{ci.loc['logp',0]:.3f}, {ci.loc['logp',1]:.3f}]"))
    r = pd.DataFrame(rows); r.to_csv(EXP / "E2b_n_vs_p.csv", index=False)
    print(r.round(3).to_string())
    names = {"ThreeSigma": "$3\\sigma$"}
    order = ["ThreeSigma", "MAD", "COPOD", "ECOD", "IForest", "LOF", "KNN", "ABOD"]
    L = ["\\begin{tabular}{llcc}", "\\hline",
         "\\textbf{Class} & \\textbf{Method} & $a$ (sample size $n$) & $b$ (sampling fraction $p$) \\\\", "\\hline"]
    for m in order:
        x = r[r.method == m].iloc[0]
        L.append(f"{x.method_class} & {names.get(m, m)} & ${x.coef_logn:+.2f}$ [{x.n_lo:+.2f}, {x.n_hi:+.2f}] & "
                 f"${x.coef_logp:+.2f}$ [{x.p_lo:+.2f}, {x.p_hi:+.2f}] \\\\")
    L += ["\\hline", "\\end{tabular}"]
    t = EXP / "E2b5000" / "tables"; t.mkdir(parents=True, exist_ok=True)
    (t / "table_e2b.tex").write_text("\n".join(L) + "\n")


def e3(c):
    df = load("E3")
    if df is None: return
    health(df, "E3")
    df["k"] = df.variant.str.extract(r"n_neighbors=(\d+)").astype(int)
    g = df.groupby(["method", "dataset", "k", "p"], as_index=False).rho.mean()
    g["y"] = np.log(1 - g.rho.clip(upper=1 - 1e-9)); g["logk"] = np.log(g.k); g["logp"] = np.log(g.p)
    g["x_old"] = 1 / np.sqrt(g.k * g.p); g["x_nh"] = np.sqrt((1 - g.p) / g.k); g["one_minus"] = 1 - g.rho
    import statsmodels.formula.api as smf
    rows = []
    for m, t in g.groupby("method"):
        t = t[t.rho < 1]
        ts = two_stage(t, ("logk", "logp")); a, b = ts["logk"], ts["logp"]
        mm = smf.mixedlm("y ~ logk + logp", t, groups=t["dataset"]).fit(reml=True); ci = mm.conf_int()
        r = dict(method=m, model="two-stage: per-dataset OLS, t-interval over datasets", n_datasets=a["k"],
                 a_logk=a["coef"], ci_k=f"[{a['lo']:.3f}, {a['hi']:.3f}]", p_logk=a["p"],
                 b_logp=b["coef"], ci_p=f"[{b['lo']:.3f}, {b['hi']:.3f}]", p_logp=b["p"],
                 mixed_a_logk=mm.params["logk"], mixed_ci_k=f"[{ci.loc['logk',0]:.3f}, {ci.loc['logk',1]:.3f}]",
                 mixed_b_logp=mm.params["logp"], mixed_ci_p=f"[{ci.loc['logp',0]:.3f}, {ci.loc['logp',1]:.3f}]")
        for x in ("x_old", "x_nh"):
            r2 = [smf.ols(f"one_minus ~ {x}", s).fit().rsquared for _, s in t.groupby("dataset")]
            r[f"median_R2_{x}"] = np.median(r2)
        rows.append(r)
    r = pd.DataFrame(rows); r.to_csv(out("E3") / "k_p_fit.csv", index=False)
    g.to_csv(out("E3") / "rho_by_k_p.csv", index=False)
    print(r.round(3).to_string())


def e5(c):
    df = load("E5")
    if df is None: return
    health(df, "E5")
    t = df.groupby(["method", "method_class", "p"]).agg(rho=("rho", "mean"), rho_sd=("rho", "std"),
                                                        t_S=("t_fit_S_s", "mean")).reset_index()
    tD = df.groupby(["method", "dataset"]).t_fit_D_s.mean().groupby("method").mean().rename("t_D")
    t = t.merge(tD, on="method"); t["speedup"] = t.t_D / t.t_S
    per_ds = df.groupby(["method", "dataset", "p"]).rho.mean().unstack("p")
    t.to_csv(out("E5") / "summary.csv", index=False); per_ds.to_csv(out("E5") / "rho_by_dataset.csv")
    print(t.round(3).to_string())
    rt_path = out("E5") / "runtime_idle.csv"
    rho = per_ds.groupby(level=0).mean()
    names = {"ThreeSigma": "$3\\sigma$", "ChiSquare": "$\\chi^2$"}
    order = ["ThreeSigma", "BoxPlot", "MAD", "ChiSquare", "Mahalanobis", "COPOD", "ECOD", "HBOS", "IForest", "EIF"]
    cls = dict(zip(df.method, df.method_class))
    L = ["\\begin{tabular}{llccccc}", "\\hline",
         "\\textbf{Class} & \\textbf{Method} & $\\rho$, $p=1\\%$ & $\\rho$, $p=5\\%$ & $\\rho$, $p=10\\%$ & "
         "$t_{\\mathrm{full}}$ (s) & speed-up at $1\\%$ / $10\\%$ \\\\", "\\hline"]
    rt = pd.read_csv(rt_path) if rt_path.exists() else None
    for m in order:
        r = rho.loc[m]
        if rt is not None:
            x = rt[rt.method == m]
            tf = f"{x.t_full_s.min():.2f}--{x.t_full_s.max():.2f}"
            sp = f"{x[x.p == 0.01].speedup.median():.0f} / {x[x.p == 0.10].speedup.median():.1f}"
        else:
            tf = sp = "---"
        L.append(f"{cls[m]} & {names.get(m, m)} & {r[0.01]:.2f} & {r[0.05]:.2f} & {r[0.1]:.2f} & {tf} & {sp} \\\\")
    L += ["\\hline", "\\end{tabular}"]
    (out("E5") / "table_e5.tex").write_text("\n".join(L) + "\n")


def e6(c):
    df = load("X5")
    if df is None: return
    health(df, "X5")
    df["T"] = df.variant.str.extract(r"n_estimators=(\d+)").astype(int)
    g = df.groupby(["method", "dataset", "T"], as_index=False).rho.mean()
    s = g.groupby(["method", "T"]).rho.agg(["mean", "std", "count"]).reset_index()
    rows = []
    for m, t in g.groupby("method"):
        w = t.pivot(index="dataset", columns="T", values="rho").dropna()
        lo, hi = w.columns.min(), w.columns.max()
        wt = stats.wilcoxon(w[hi], w[lo])
        rows.append(dict(method=m, T_low=lo, T_high=hi, gain=float((w[hi] - w[lo]).mean()), p=wt.pvalue,
                         n_datasets=len(w)))
    s.to_csv(out("X5") / "rho_by_T.csv", index=False); pd.DataFrame(rows).to_csv(out("X5") / "T_test.csv", index=False)
    print(s.round(3).to_string()); print(pd.DataFrame(rows).round(4).to_string())


def e7(c):
    df = load("X6")
    if df is None: return
    health(df, "X6")
    g = df.groupby(["method", "method_class", "dataset", "d"], as_index=False).rho.mean()
    t = g.pivot_table(index=["method_class", "method"], columns="d", values="rho")
    cl = g.groupby(["method_class", "d"]).rho.mean().unstack()
    t.to_csv(out("X6") / "rho_by_d.csv"); cl.to_csv(out("X6") / "class_by_d.csv")
    print(t.round(3).to_string()); print(cl.round(3).to_string())


def e8(c):
    df = load("X7")
    if df is None: return
    health(df, "X7")
    df["type"] = df.dataset.str.split("_").str[0]; df["base"] = df.dataset.str.split("_", n=1).str[1]
    g = df.groupby(["method", "method_class", "type", "base"], as_index=False).rho.mean()
    cl = g[g.method_class.isin(CLASSES + ["G"])].groupby(["type", "base", "method_class"]).rho.mean().unstack()
    rows = []
    for ty, t in cl.groupby(level=0):
        rows.append(dict(type=ty, **{f"class_{k}": t[k].mean() for k in CLASSES + ["G"]},
                         IV_lowest=int((t["IV"] < t[["I", "II", "III"]].min(1)).sum()), n_bases=len(t)))
    t = g.pivot_table(index=["method_class", "method"], columns="type", values="rho")
    t.to_csv(out("X7") / "rho_by_type.csv"); pd.DataFrame(rows).to_csv(out("X7") / "class_by_type.csv", index=False)
    print(t.round(3).to_string()); print(pd.DataFrame(rows).round(3).to_string())


@hydra.main(version_base=None, config_name="ar")
def main(c: Cfg) -> None:
    fns = dict(E1b=e1b, E2=e2, E2b=e2b, E3=e3, E5=e5, E6=e6, E7=e7, E8=e8)
    for k, f in fns.items():
        if not c.only or k in c.only:
            f(c)


if __name__ == "__main__":
    sys.argv += ["hydra.run.dir=.", "hydra.output_subdir=null"]
    main()
