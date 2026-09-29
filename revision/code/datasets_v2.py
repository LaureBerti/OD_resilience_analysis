"""datasets_v2.py — dataset access for the revision experiments.

load(name, n_cap, seed) -> (X, y, meta)
  * the 16 core benchmark datasets (data/*.npz, via github_repo/scripts/data_loader.py);
  * 13 extra datasets with N >= 1,000 for E2 (data/extra/*.npz, see data/extra/README.md);
  * gauss_d{d}: synthetic Gaussian data for the dimension study (X6);
  * {local,global,clustered}_{base}: ADBench realistic-synthetic anomalies (X7) generated from a
    core dataset following ADBench's generate_realistic_synthetic (Han et al. 2022): a Gaussian
    mixture (1-9 components, chosen by BIC) is fitted to the z-scored normal records; normals are
    resampled from it; local anomalies use covariances x alpha, clustered anomalies use means x alpha
    (alpha = 5), global anomalies are uniform on [min(1+0.1), max(1+0.1)] per feature; the number
    of anomalies equals the base dataset's.
If N > n_cap (n_cap > 0), one fixed random subset of n_cap records (seed) is the reference set D.
Tiers follow the paper's dataset table, on the ORIGINAL size N_orig:
  T1 N < 1,000;  T2 1,000 <= N < 10,000;  T3 10,000 <= N < 100,000;  T4 N >= 100,000.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]

def _legacy_scripts(project):
    """Folder of the earlier pipeline (data loader, earlier wrappers): github_repo/scripts in the paper
    project, scripts/ in the public repository."""
    for cand in (project / "github_repo" / "scripts", project / "scripts"):
        if (cand / "data_loader.py").exists():
            return cand
    raise FileNotFoundError("data_loader.py not found in github_repo/scripts or scripts/")


sys.path.insert(0, str(_legacy_scripts(PROJECT)))

CORE_DATASETS = ["lympho", "glass", "vertebral", "ionosphere", "breastw",
                 "letter", "musk", "thyroid", "optdigits", "satellite", "pendigits", "annthyroid",
                 "shuttle", "smtp", "skin", "covertype"]
EXTRA = {"cardio": "cardio_ODDS", "vowels": "vowels_ODDS", "waveform": "Waveform_ADBench",
         "speech": "speech_ODDS", "pageblocks": "PageBlocks_ADBench", "satimage2": "satimage-2_ODDS",
         "mnist": "mnist_ODDS", "mammography": "mammography_ODDS", "magic_gamma": "magic_gamma_ADBench",
         "campaign": "campaign_ADBench", "celeba": "celeba_ADBench", "fraud": "fraud_ADBench",
         "http": "http_ODDS"}
CORE_N1000 = ["letter", "musk", "thyroid", "optdigits", "satellite", "pendigits", "annthyroid",
              "shuttle", "smtp", "skin", "covertype"]
E2_DATASETS = CORE_N1000 + list(EXTRA)          # 11 + 13 = 24 datasets, all with N >= 1,000


def tier_of(N: int) -> str:
    return "T1" if N < 1000 else "T2" if N < 10000 else "T3" if N < 100000 else "T4"


@lru_cache(maxsize=64)
def _raw(name: str):
    if name in EXTRA:
        z = np.load(PROJECT / "data" / "extra" / f"{EXTRA[name]}.npz", allow_pickle=False)
        return np.asarray(z["X"], float), np.asarray(z["y"], int).ravel()
    if name.startswith("gauss_d"):
        return _gauss(int(name[len("gauss_d"):]))
    for t in ("local", "global", "clustered"):
        if name.startswith(t + "_"):
            return _adbench_synthetic(t, name[len(t) + 1:])
    from data_loader import load_dataset
    X, y = load_dataset(name)
    return np.asarray(X, float), np.asarray(y, int).ravel()


def _gauss(d: int, N: int = 5000, frac: float = 0.05, shift: float = 5.0, seed: int = 7):
    """Inliers N(0, I_d); outliers N(shift * u, I_d) for one random unit vector u, so the
    inlier-outlier separation does not change with d."""
    rng = np.random.default_rng(seed + d)
    n_out = int(round(frac * N))
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    X = np.vstack([rng.standard_normal((N - n_out, d)), rng.standard_normal((n_out, d)) + shift * u])
    y = np.r_[np.zeros(N - n_out, int), np.ones(n_out, int)]
    p = rng.permutation(N)
    return X[p], y[p]


def _adbench_synthetic(kind: str, base: str, alpha: float = 5.0, percentage: float = 0.1,
                       seed: int = 42, n_cap: int = 5000):
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    X, y = _raw(base)
    if len(X) > n_cap:
        i = np.sort(np.random.default_rng(seed).choice(len(X), n_cap, replace=False))
        X, y = X[i], y[i]
    Xn = StandardScaler().fit_transform(X[y == 0])
    pts_n, pts_a = int((y == 0).sum()), int((y == 1).sum())
    bic = [GaussianMixture(n_components=c, random_state=seed).fit(Xn).bic(Xn) for c in range(1, 10)]
    gm = GaussianMixture(n_components=int(np.argmin(bic)) + 1, random_state=seed).fit(Xn)
    Xs_n = gm.sample(pts_n)[0]
    rng = np.random.default_rng(seed)
    if kind == "local":
        gm.covariances_ = alpha * gm.covariances_
        gm.precisions_cholesky_ = None  # not used by sample()
        Xs_a = gm.sample(pts_a)[0]
    elif kind == "clustered":
        gm.means_ = alpha * gm.means_
        Xs_a = gm.sample(pts_a)[0]
    else:  # global
        lo = Xs_n.min(0) * (1 + percentage); hi = Xs_n.max(0) * (1 + percentage)
        Xs_a = rng.uniform(lo, hi, size=(pts_a, Xs_n.shape[1]))
    Xo = np.vstack([Xs_n, Xs_a]); yo = np.r_[np.zeros(pts_n, int), np.ones(pts_a, int)]
    p = rng.permutation(len(Xo))
    return Xo[p], yo[p]


def load(name: str, n_cap: int = 5000, seed: int = 42):
    X, y = _raw(name)
    N_orig = len(X)
    if n_cap and N_orig > n_cap:
        idx = np.sort(np.random.default_rng(seed).choice(N_orig, n_cap, replace=False))
        X, y = X[idx], y[idx]
    return X, y, dict(N_orig=N_orig, tier=tier_of(N_orig))
