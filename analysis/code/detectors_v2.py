"""
detectors_v2.py — in-sample outlier scores for the 18 benchmark detectors.

Every detector is exposed as a function  score_fn(X, seed) -> np.ndarray of shape (len(X),),
returning the IN-SAMPLE outlier score of each training record (higher = more anomalous),
i.e. the quantity PyOD calls `decision_scores_`. Resilience only ever compares scores of
records that belong to the training set (records of S under the model fitted on D, and
records of S under the model fitted on S), so in-sample scores are the correct object.

Reference implementations are used wherever one exists in this environment:
    COPOD, ECOD, HBOS, ABOD, FeatureBagging, ROD           -> PyOD 0.9.9 (default parameters)
    LOF, OCSVM, IForest                                     -> scikit-learn (PyOD wraps these)
    KNN                                                     -> scikit-learn NearestNeighbors (PyOD definition)
    Mahalanobis                                             -> scikit-learn EmpiricalCovariance
    EIF                                                     -> isotree (ndim = d: extended model of Hariri et al.)
    ADOD                                                    -> CPU port of the authors' code (github.com/Qian-Lily/ADOD,
                                                               adod.py), exact k-NN in place of GPU FAISS; validated in
                                                               validate_reimplementations.py
Textbook rules implemented directly (no library reference exists): ThreeSigma, BoxPlot, MAD,
ChiSquare (diagonal Mahalanobis), KMeans (distance to nearest of 8 centroids, standardised features).

Method classes (see paper Tables 2 and 5):
    I   global location/scale statistics : ThreeSigma, BoxPlot, MAD, ChiSquare, Mahalanobis, ROD
    II  empirical-CDF / histogram        : COPOD, ECOD, HBOS
    III internal-subsampling ensembles   : IForest, EIF
    IV  k-NN-graph methods               : LOF, KNN, ABOD, FeatureBagging, ADOD
    G   global fitted models (reported separately): KMeans, OCSVM
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

METHOD_CLASS = {
    "ThreeSigma": "I", "BoxPlot": "I", "MAD": "I", "ChiSquare": "I", "Mahalanobis": "I", "ROD": "I",
    "COPOD": "II", "ECOD": "II", "HBOS": "II",
    "IForest": "III", "EIF": "III",
    "LOF": "IV", "KNN": "IV", "ABOD": "IV", "FeatureBagging": "IV", "ADOD": "IV",
    "KMeans": "G", "OCSVM": "G",
}
EARLIER_CLASS = {
    "ThreeSigma": "I", "BoxPlot": "I", "MAD": "I", "ChiSquare": "I",
    "COPOD": "II", "ECOD": "II", "HBOS": "II", "IForest": "III", "EIF": "III",
    "LOF": "IV", "KNN": "IV", "OCSVM": "IV", "Mahalanobis": "IV", "KMeans": "IV",
    "ROD": "IV", "ABOD": "IV", "FeatureBagging": "IV", "ADOD": "IV",
}
STOCHASTIC = {"IForest", "EIF", "KMeans", "FeatureBagging"}


def _f(X):
    return np.ascontiguousarray(np.asarray(X, dtype=float))


def three_sigma(X, seed=0, **kw):
    X = _f(X); s = X.std(0); s[s == 0] = 1e-10
    return np.max(np.abs(X - X.mean(0)) / s, axis=1)


def boxplot(X, seed=0, k=1.5, **kw):
    X = _f(X); q1, q3 = np.percentile(X, [25, 75], axis=0); iqr = q3 - q1; iqr[iqr == 0] = 1e-10
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return np.max(np.maximum(0, (lo - X) / iqr) + np.maximum(0, (X - hi) / iqr), axis=1)


def mad(X, seed=0, **kw):
    X = _f(X); med = np.median(X, 0); m = np.median(np.abs(X - med), 0); m[m == 0] = 1e-10
    return np.max(np.abs(X - med) / m, axis=1)


def chisquare(X, seed=0, **kw):
    X = _f(X); v = X.var(0); v[v == 0] = 1e-10
    return np.sum((X - X.mean(0)) ** 2 / v, axis=1)


def mahalanobis(X, seed=0, **kw):
    from sklearn.covariance import EmpiricalCovariance
    X = _f(X)
    return EmpiricalCovariance().fit(X).mahalanobis(X)


def rod(X, seed=0, max_dim=5, **kw):
    """PyOD ROD. ROD enumerates all C(d,3) 3-D subspaces; for d > max_dim the data are first
    projected on the first max_dim principal components (C(5,3) = 10 subspaces)."""
    from pyod.models.rod import ROD
    X = _f(X)
    if X.shape[1] > max_dim:
        from sklearn.decomposition import PCA
        X = PCA(n_components=min(max_dim, X.shape[0] - 1), random_state=0).fit_transform(X)
    return ROD().fit(X).decision_scores_


def _nonconstant(X):
    """COPOD and ECOD weight tails by the sign of each feature's skewness. For a feature that is
    constant in the training set, or nearly so (scipy then returns NaN to avoid precision loss), the
    skewness is undefined and every score becomes NaN. Such features are dropped before fitting."""
    from scipy.stats import skew
    X = _f(X)
    keep = (X.std(0) > 0) & np.isfinite(skew(X, axis=0))
    return X[:, keep] if keep.any() else X


def copod(X, seed=0, **kw):
    from pyod.models.copod import COPOD
    return COPOD().fit(_nonconstant(X)).decision_scores_


def ecod(X, seed=0, **kw):
    from pyod.models.ecod import ECOD
    return ECOD().fit(_nonconstant(X)).decision_scores_


def hbos(X, seed=0, **kw):
    from pyod.models.hbos import HBOS
    return HBOS().fit(_f(X)).decision_scores_


def iforest(X, seed=0, n_estimators=100, max_samples=256, **kw):
    from sklearn.ensemble import IsolationForest
    X = _f(X)
    m = IsolationForest(n_estimators=n_estimators, max_samples=min(max_samples, len(X)),
                        random_state=seed).fit(X)
    return -m.score_samples(X)


def eif(X, seed=0, n_estimators=200, max_samples=256, **kw):
    """Extended Isolation Forest (Hariri et al. 2021) via isotree with ndim = d (full extension
    level), normally distributed split coefficients, no range penalty, random split points."""
    from isotree import IsolationForest as IsoTree
    X = _f(X)
    d = X.shape[1]
    m = IsoTree(ndim=max(d, 1), sample_size=min(max_samples, len(X)), ntrees=n_estimators,
                coefs="normal", penalize_range=False, prob_pick_pooled_gain=0.0,
                prob_pick_avg_gain=0.0, missing_action="fail", random_seed=int(seed), nthreads=1)
    m.fit(X)
    return m.predict(X, output="score")


def lof(X, seed=0, n_neighbors=20, **kw):
    from sklearn.neighbors import LocalOutlierFactor
    X = _f(X)
    m = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(X) - 1)).fit(X)
    return -m.negative_outlier_factor_


def knn(X, seed=0, n_neighbors=5, **kw):
    """PyOD KNN default: distance to the k-th nearest neighbour, self excluded (method='largest')."""
    X = _f(X)
    k = min(n_neighbors, len(X) - 1)
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return d[:, k]


def abod(X, seed=0, n_neighbors=5, **kw):
    from pyod.models.abod import ABOD
    X = _f(X)
    return ABOD(method="fast", n_neighbors=min(n_neighbors, len(X) - 2)).fit(X).decision_scores_


def feature_bagging(X, seed=0, **kw):
    from pyod.models.feature_bagging import FeatureBagging
    X = _f(X)
    base = None
    if len(X) <= 21:
        from pyod.models.lof import LOF
        base = LOF(n_neighbors=max(1, len(X) - 2))
    return FeatureBagging(base_estimator=base, random_state=seed).fit(X).decision_scores_


def _adod_sigma(sq_d, perplexity, iters=50, tol=1e-5):
    """Vectorised copy of binary_search_perplexity_knn (ADOD adod.py); sq_d excludes self."""
    n = len(sq_d)
    beta = np.ones(n); bmin = np.full(n, -np.inf); bmax = np.full(n, np.inf)
    logU = np.log(perplexity)
    done = np.zeros(n, bool)
    with np.errstate(all="ignore"):
        for _ in range(iters):
            P = np.exp(-sq_d * beta[:, None])
            sP = P.sum(1)
            H = np.log(sP) + beta * (sq_d * P).sum(1) / sP
            Hd = H - logU
            done |= np.abs(Hd) <= tol
            act = ~done
            up = act & (Hd > 0)
            dn = act & ~(Hd > 0)
            bmin[up] = beta[up]
            beta[up] = np.where(np.isinf(bmax[up]), beta[up] * 2.0, (beta[up] + bmax[up]) / 2.0)
            bmax[dn] = beta[dn]
            beta[dn] = np.where(np.isinf(bmin[dn]), beta[dn] / 2.0, (beta[dn] + bmin[dn]) / 2.0)
    return np.sqrt(1.0 / beta)


def adod(X, seed=0, probability=0.999, perplexity=None, return_parts=False, **kw):
    """ADOD (Qian et al., ICDM 2024) in-sample scores; faithful port of ADOD.fit in adod.py.
    perplexity = 2*floor(sqrt(n)), k = min(n, 3*perplexity, 2048), mutual-neighbour graph
    within per-point Gaussian boundaries, score = own inverse density + distance-weighted mean
    of (neighbour inverse density - own inverse density)."""
    X = _f(X)
    n = len(X)
    perp = int(np.sqrt(n)) * 2 if perplexity is None else perplexity
    k = min(n, 3 * perp, 2048)
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dist, idx = nn.kneighbors(X)
    sigma = _adod_sigma(dist[:, 1:] ** 2, perp)
    b = norm.ppf(probability, loc=0, scale=sigma)
    within = (dist <= b[:, None]) & (dist <= b[idx])
    rows = np.repeat(np.arange(n), k)[within.ravel()]
    cols = idx.ravel()[within.ravel()]
    M = sparse.coo_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(n, n)).tocsr()
    M = (M + M.T).astype(bool).tolil(); M.setdiag(False); M = M.tocsr()
    dens = (np.asarray(M.sum(1)).ravel() + 1.0) / b
    inv = 1.0 / dens
    r = np.repeat(np.arange(n), k); c = idx.ravel()
    mut = np.asarray(M[r, c]).ravel().reshape(n, k)
    valid = mut & (dist > 0)
    w = np.where(valid, 1.0 / np.where(dist > 0, dist, 1.0), 0.0)
    ws = w.sum(1)
    diff = (w * (inv[idx] - inv[:, None])).sum(1)
    score = inv + np.where(ws > 0, diff / np.where(ws > 0, ws, 1.0), 0.0)
    if return_parts:
        return score, dict(sigma=sigma, boundary=b, density=dens, k=k, perplexity=perp)
    return score


def kmeans(X, seed=0, n_clusters=8, **kw):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(_f(X))
    km = KMeans(n_clusters=min(n_clusters, len(Z)), n_init=10, random_state=seed).fit(Z)
    return np.min(np.linalg.norm(Z[:, None, :] - km.cluster_centers_[None], axis=2), axis=1)


def ocsvm(X, seed=0, **kw):
    from sklearn.svm import OneClassSVM
    X = _f(X)
    return -OneClassSVM(kernel="rbf", nu=0.5, gamma="scale").fit(X).decision_function(X)


SCORERS = {
    "ThreeSigma": three_sigma, "BoxPlot": boxplot, "MAD": mad, "ChiSquare": chisquare,
    "Mahalanobis": mahalanobis, "ROD": rod,
    "COPOD": copod, "ECOD": ecod, "HBOS": hbos,
    "IForest": iforest, "EIF": eif,
    "LOF": lof, "KNN": knn, "ABOD": abod, "FeatureBagging": feature_bagging, "ADOD": adod,
    "KMeans": kmeans, "OCSVM": ocsvm,
}
ALL_METHODS = list(SCORERS)


def score(method: str, X, seed: int = 0, **kw) -> np.ndarray:
    s = np.asarray(SCORERS[method](X, seed=seed, **kw), dtype=float).ravel()
    if len(s) != len(X):
        raise RuntimeError(f"{method}: {len(s)} scores for {len(X)} records")
    return s
