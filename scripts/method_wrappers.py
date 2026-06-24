"""
method_wrappers.py
------------------
Unified wrappers for the 18 benchmark detectors (plus an optional AutoEncoder when torch is available).

Each wrapper exposes:
    fit(X)                    -> self
    decision_function(X)      -> np.ndarray scores (higher = more anomalous)
    predict(X, contamination) -> np.ndarray binary labels {0,1}

PyOD methods: COPOD, ECOD, HBOS, IForest, LOF, KNN, OCSVM, AutoEncoder
Custom methods: ThreeSigma, BoxPlot, MAD, ChiSquare, Mahalanobis, KMeans

# SUOD excluded: requires numba/llvmlite (needs cmake). Use pyod[suod] + cmake if needed.

Method classes for theoretical analysis (18 benchmark detectors):
    Class I  (moment-based): ThreeSigma, BoxPlot, MAD, ChiSquare
    Class II (CDF/histogram): COPOD, ECOD, HBOS
    Class III (subsampling):  IForest, EIF
    Class IV (local density): LOF, KNN, OCSVM, Mahalanobis, KMeans, ROD, ABOD, FeatureBagging, ADOD
    (AutoEncoder, Class IV, is optional and enabled only when torch is available.)
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.ensemble import IsolationForest as SklearnIForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseOutlierDetector(ABC):
    """Common interface for all outlier detectors."""

    def __init__(self, contamination: float = 0.10, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self._fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseOutlierDetector":
        pass

    @abstractmethod
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        pass

    def predict(
        self, X: np.ndarray, contamination: Optional[float] = None
    ) -> np.ndarray:
        """Return binary labels using top-k threshold."""
        if contamination is None:
            contamination = self.contamination
        scores = self.decision_function(X)
        n = len(scores)
        k = max(1, int(np.floor(contamination * n)))
        threshold = np.sort(scores)[-k]
        return (scores >= threshold).astype(int)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted yet.")


# ---------------------------------------------------------------------------
# Class I — Moment-based methods (custom implementations)
# ---------------------------------------------------------------------------

class ThreeSigmaDetector(BaseOutlierDetector):
    """
    3-Sigma rule: outlier if |x_j - mu_j| > 3*sigma_j for any feature j.
    Score = max over features of (|x_j - mu_j| / sigma_j).
    """

    def __init__(self, contamination: float = 0.10, random_state: int = 42):
        super().__init__(contamination, random_state)
        self._means: Optional[np.ndarray] = None
        self._stds: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "ThreeSigmaDetector":
        X = np.asarray(X, dtype=float)
        self._means = np.mean(X, axis=0)
        self._stds = np.std(X, axis=0)
        self._stds = np.where(self._stds == 0, 1e-10, self._stds)
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        z = np.abs(X - self._means) / self._stds
        return np.max(z, axis=1)


class BoxPlotDetector(BaseOutlierDetector):
    """
    BoxPlot (Tukey fence) rule: outlier if x_j < Q1-1.5*IQR or x_j > Q3+1.5*IQR.
    Score = max over features of signed distance to nearest fence (normalised by IQR).
    """

    def __init__(self, contamination: float = 0.10, random_state: int = 42, k: float = 1.5):
        super().__init__(contamination, random_state)
        self.k = k
        self._q1: Optional[np.ndarray] = None
        self._q3: Optional[np.ndarray] = None
        self._iqr: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "BoxPlotDetector":
        X = np.asarray(X, dtype=float)
        self._q1 = np.percentile(X, 25, axis=0)
        self._q3 = np.percentile(X, 75, axis=0)
        self._iqr = self._q3 - self._q1
        self._iqr = np.where(self._iqr == 0, 1e-10, self._iqr)
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        lower = self._q1 - self.k * self._iqr
        upper = self._q3 + self.k * self._iqr
        below = np.maximum(0, (lower - X) / self._iqr)
        above = np.maximum(0, (X - upper) / self._iqr)
        return np.max(below + above, axis=1)


class MADDetector(BaseOutlierDetector):
    """
    MAD (Median Absolute Deviation): outlier if |x_j - median_j| > 3.5*MAD_j.
    Score = max over features of (|x_j - median_j| / MAD_j).
    """

    def __init__(self, contamination: float = 0.10, random_state: int = 42):
        super().__init__(contamination, random_state)
        self._medians: Optional[np.ndarray] = None
        self._mads: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "MADDetector":
        X = np.asarray(X, dtype=float)
        self._medians = np.median(X, axis=0)
        self._mads = np.median(np.abs(X - self._medians), axis=0)
        self._mads = np.where(self._mads == 0, 1e-10, self._mads)
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        z = np.abs(X - self._medians) / self._mads
        return np.max(z, axis=1)


class ChiSquareDetector(BaseOutlierDetector):
    """
    Chi-Square distance from feature-wise mean (assumes approximate normality).
    Score = sum over features of ((x_j - mu_j)^2 / sigma_j^2).
    This is the Mahalanobis distance with diagonal covariance.
    """

    def __init__(self, contamination: float = 0.10, random_state: int = 42):
        super().__init__(contamination, random_state)
        self._means: Optional[np.ndarray] = None
        self._vars: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "ChiSquareDetector":
        X = np.asarray(X, dtype=float)
        self._means = np.mean(X, axis=0)
        self._vars = np.var(X, axis=0)
        self._vars = np.where(self._vars == 0, 1e-10, self._vars)
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        chi2 = np.sum((X - self._means) ** 2 / self._vars, axis=1)
        return chi2


# ---------------------------------------------------------------------------
# Class II — CDF/histogram-based (PyOD wrappers)
# ---------------------------------------------------------------------------

class COPODDetector(BaseOutlierDetector):
    """Wrapper for PyOD COPOD."""

    def __init__(self, contamination: float = 0.10, random_state: int = 42):
        super().__init__(contamination, random_state)
        self._model = None

    def fit(self, X: np.ndarray) -> "COPODDetector":
        from pyod.models.copod import COPOD  # type: ignore
        self._model = COPOD(contamination=self.contamination)
        self._model.fit(np.asarray(X, dtype=float))
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._model.decision_function(np.asarray(X, dtype=float))


class ECODDetector(BaseOutlierDetector):
    """Wrapper for PyOD ECOD."""

    def __init__(self, contamination: float = 0.10, random_state: int = 42):
        super().__init__(contamination, random_state)
        self._model = None

    def fit(self, X: np.ndarray) -> "ECODDetector":
        from pyod.models.ecod import ECOD  # type: ignore
        self._model = ECOD(contamination=self.contamination)
        self._model.fit(np.asarray(X, dtype=float))
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._model.decision_function(np.asarray(X, dtype=float))


class HBOSDetector(BaseOutlierDetector):
    """
    HBOS (Histogram-Based Outlier Score) — pure numpy, independent of pyod.
    Score = sum of -log(density_j) across features j (log-scale combination).
    Matches the original Goldstein & Dengel 2012 formulation.
    """

    def __init__(
        self,
        contamination: float = 0.10,
        random_state: int = 42,
        n_bins: int = 10,
    ):
        super().__init__(contamination, random_state)
        self.n_bins = n_bins
        self._bin_edges: list = []
        self._densities: list = []

    def fit(self, X: np.ndarray) -> "HBOSDetector":
        X = np.asarray(X, dtype=float)
        self._bin_edges = []
        self._densities = []
        for j in range(X.shape[1]):
            col = X[:, j]
            counts, edges = np.histogram(col, bins=self.n_bins)
            # Normalise to density (area = 1); add small epsilon to avoid log(0)
            widths = np.diff(edges)
            widths = np.where(widths == 0, 1e-10, widths)
            density = counts / (counts.sum() * widths + 1e-10)
            self._bin_edges.append(edges)
            self._densities.append(density)
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        scores = np.zeros(len(X))
        for j, (edges, density) in enumerate(zip(self._bin_edges, self._densities)):
            col = X[:, j]
            # Assign each point to the nearest bin; clip boundary values
            idx = np.searchsorted(edges[1:-1], col)
            d = np.maximum(density[idx], 1e-10)
            scores += -np.log(d)
        return scores


# ---------------------------------------------------------------------------
# Class III — Subsampling-based (PyOD IForest)
# ---------------------------------------------------------------------------

class IForestDetector(BaseOutlierDetector):
    """
    Isolation Forest (sklearn) with fixed psi=256 (max_samples=256).
    Uses sklearn directly to avoid pyod 0.9.9 / sklearn 1.x API incompatibility.
    """

    def __init__(
        self,
        contamination: float = 0.10,
        random_state: int = 42,
        n_estimators: int = 100,
        max_samples: int = 256,
    ):
        super().__init__(contamination, random_state)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self._model = None

    def fit(self, X: np.ndarray) -> "IForestDetector":
        n = len(X)
        effective_max = min(self.max_samples, n)
        self._model = SklearnIForest(
            n_estimators=self.n_estimators,
            max_samples=effective_max,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self._model.fit(np.asarray(X, dtype=float))
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        # sklearn IForest returns negative scores (more negative = more anomalous).
        # Negate so higher score = more anomalous, consistent with other detectors.
        return -self._model.score_samples(np.asarray(X, dtype=float))


# ---------------------------------------------------------------------------
# Class IV — Local density / distance-based
# ---------------------------------------------------------------------------

class LOFDetector(BaseOutlierDetector):
    """LOF (sklearn) — uses sklearn directly to avoid pyod/sklearn 1.x incompatibility."""

    def __init__(
        self,
        contamination: float = 0.10,
        random_state: int = 42,
        n_neighbors: int = 20,
    ):
        super().__init__(contamination, random_state)
        self.n_neighbors = n_neighbors
        self._model = None
        self._X_train = None

    def fit(self, X: np.ndarray) -> "LOFDetector":
        X = np.asarray(X, dtype=float)
        n = len(X)
        effective_k = min(self.n_neighbors, n - 1)
        self._model = LocalOutlierFactor(
            n_neighbors=effective_k,
            contamination=self.contamination,
            novelty=True,
        )
        self._model.fit(X)
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        # sklearn LOF novelty=True: negative_outlier_factor_ → negate for consistency.
        return -self._model.score_samples(np.asarray(X, dtype=float))


class KNNDetector(BaseOutlierDetector):
    """KNN outlier detector (sklearn NearestNeighbors) — score = distance to k-th neighbor."""

    def __init__(
        self,
        contamination: float = 0.10,
        random_state: int = 42,
        n_neighbors: int = 5,
    ):
        super().__init__(contamination, random_state)
        self.n_neighbors = n_neighbors
        self._nn = None

    def fit(self, X: np.ndarray) -> "KNNDetector":
        X = np.asarray(X, dtype=float)
        n = len(X)
        effective_k = min(self.n_neighbors, n - 1)
        self._nn = NearestNeighbors(n_neighbors=effective_k)
        self._nn.fit(X)
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        dists, _ = self._nn.kneighbors(np.asarray(X, dtype=float))
        return dists[:, -1]  # distance to k-th nearest neighbor


class OCSVMDetector(BaseOutlierDetector):
    """One-Class SVM (sklearn) — uses sklearn directly to avoid pyod/sklearn 1.x incompatibility."""

    def __init__(
        self,
        contamination: float = 0.10,
        random_state: int = 42,
        kernel: str = "rbf",
        nu: float = 0.5,
    ):
        super().__init__(contamination, random_state)
        self.kernel = kernel
        self.nu = nu
        self._model = None

    def fit(self, X: np.ndarray) -> "OCSVMDetector":
        self._model = OneClassSVM(kernel=self.kernel, nu=self.nu)
        self._model.fit(np.asarray(X, dtype=float))
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        # sklearn OCSVM: positive = inlier, negative = outlier. Negate for consistency.
        return -self._model.decision_function(np.asarray(X, dtype=float))


class MahalanobisDetector(BaseOutlierDetector):
    """
    Mahalanobis distance-based outlier detector (custom, scipy).
    Robust to singular covariance via pseudo-inverse.
    """

    def __init__(self, contamination: float = 0.10, random_state: int = 42):
        super().__init__(contamination, random_state)
        self._mean: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "MahalanobisDetector":
        X = np.asarray(X, dtype=float)
        self._mean = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        try:
            self._cov_inv = np.linalg.pinv(cov)
        except Exception:
            self._cov_inv = np.eye(X.shape[1])
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        diff = X - self._mean
        scores = np.array(
            [float(d @ self._cov_inv @ d) for d in diff]
        )
        return scores


class KMeansDetector(BaseOutlierDetector):
    """
    KMeans-based outlier detector (custom, sklearn).
    Score = distance to nearest cluster centroid.
    """

    def __init__(
        self,
        contamination: float = 0.10,
        random_state: int = 42,
        n_clusters: int = 8,
    ):
        super().__init__(contamination, random_state)
        self.n_clusters = n_clusters
        self._kmeans = None
        self._scaler = None

    def fit(self, X: np.ndarray) -> "KMeansDetector":
        X = np.asarray(X, dtype=float)
        n = len(X)
        effective_k = min(self.n_clusters, n)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._kmeans = SklearnKMeans(
            n_clusters=effective_k,
            random_state=self.random_state,
            n_init=10,
        )
        self._kmeans.fit(X_scaled)
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        X_scaled = self._scaler.transform(X)
        centers = self._kmeans.cluster_centers_
        dists = np.min(
            np.linalg.norm(X_scaled[:, None, :] - centers[None, :, :], axis=2),
            axis=1,
        )
        return dists


class AutoEncoderDetector(BaseOutlierDetector):
    """Wrapper for PyOD AutoEncoder."""

    def __init__(
        self,
        contamination: float = 0.10,
        random_state: int = 42,
        hidden_neurons: Optional[list] = None,
        epochs: int = 100,
        batch_size: int = 32,
    ):
        super().__init__(contamination, random_state)
        self.hidden_neurons = hidden_neurons or [64, 32, 32, 64]
        self.epochs = epochs
        self.batch_size = batch_size
        self._model = None

    def fit(self, X: np.ndarray) -> "AutoEncoderDetector":
        from pyod.models.auto_encoder import AutoEncoder  # type: ignore
        X = np.asarray(X, dtype=float)
        self._model = AutoEncoder(
            hidden_neurons=self.hidden_neurons,
            epochs=self.epochs,
            batch_size=self.batch_size,
            contamination=self.contamination,
            random_state=self.random_state,
            verbose=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X)
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._model.decision_function(np.asarray(X, dtype=float))


# ---------------------------------------------------------------------------
# Class III — Extended Isolation Forest  (Hariri et al., IEEE TKDE 2021)
# ---------------------------------------------------------------------------

def _c_factor(n: int) -> float:
    """Expected path length in an unsuccessful BST search (IForest normalisation)."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


def _eif_score_batch(X: np.ndarray, tree, depths: np.ndarray,
                     idx: np.ndarray, depth: int) -> None:
    """Vectorised batch descent through one EIF tree."""
    if len(idx) == 0:
        return
    if tree[0] == 'leaf':
        depths[idx] = depth + _c_factor(tree[1])
        return
    _, n_vec, intercept, left, right = tree
    proj = X[idx] @ n_vec
    go_left = proj <= intercept
    _eif_score_batch(X, left,  depths, idx[go_left],  depth + 1)
    _eif_score_batch(X, right, depths, idx[~go_left], depth + 1)


def _eif_build_tree(X: np.ndarray, depth: int, limit: int,
                    ext_level: int, rng: np.random.Generator):
    n, d = X.shape
    if depth >= limit or n <= 1:
        return ('leaf', n)
    n_vec = np.zeros(d)
    dims = rng.choice(d, size=min(ext_level + 1, d), replace=False)
    n_vec[dims] = rng.standard_normal(size=len(dims))
    proj = X @ n_vec
    p_min, p_max = proj.min(), proj.max()
    if p_min >= p_max:
        return ('leaf', n)
    intercept = rng.uniform(p_min, p_max)
    mask = proj <= intercept
    return (
        'split', n_vec, intercept,
        _eif_build_tree(X[mask], depth + 1, limit, ext_level, rng),
        _eif_build_tree(X[~mask], depth + 1, limit, ext_level, rng),
    )


class EIFDetector(BaseOutlierDetector):
    """
    Extended Isolation Forest (Hariri, Kind, Brunner — IEEE TKDE 2021).
    Random hyperplane splits; otherwise same scoring as IForest.
    Class III: retains IForest's internal-subsampling structure.
    doi:10.1109/TKDE.2019.2947676 | GitHub: sahandha/eif
    """

    def __init__(self, contamination: float = 0.10, random_state: int = 42,
                 n_estimators: int = 200, max_samples: int = 256,
                 extension_level: Optional[int] = None):
        super().__init__(contamination, random_state)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.extension_level = extension_level
        self._trees: list = []
        self._c: float = 0.0

    def fit(self, X: np.ndarray) -> "EIFDetector":
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        sample_size = min(self.max_samples, n)
        ext_level = (d - 1) if self.extension_level is None else min(self.extension_level, d - 1)
        limit = int(np.ceil(np.log2(sample_size))) if sample_size > 1 else 1
        self._c = _c_factor(sample_size)
        rng = np.random.default_rng(self.random_state)
        self._trees = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n, size=sample_size, replace=False)
            self._trees.append(_eif_build_tree(X[idx], 0, limit, ext_level, rng))
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        n = len(X)
        c = self._c if self._c > 0 else 1.0
        all_idx = np.arange(n)
        total = np.zeros(n)
        for tree in self._trees:
            depths = np.empty(n)
            _eif_score_batch(X, tree, depths, all_idx, 0)
            total += depths
        return -(2.0 ** (-total / len(self._trees) / c))


# ---------------------------------------------------------------------------
# Class IV — ADOD  (Qian et al., IEEE ICDM 2024)
# ---------------------------------------------------------------------------

def _adod_sigmas_batch(dists_sq: np.ndarray, target_perp: float,
                       tol: float = 1e-3, max_iter: int = 40) -> np.ndarray:
    n = dists_sq.shape[0]
    lo = np.full(n, 1e-6)
    hi = np.full(n, 1e6)
    sigma = np.ones(n)
    active = np.ones(n, dtype=bool)
    for _ in range(max_iter):
        if not active.any():
            break
        s = active.copy()
        sigma[s] = (lo[s] + hi[s]) * 0.5
        sig2 = 2.0 * sigma[s] ** 2
        neg = -dists_sq[s] / sig2[:, None]
        neg -= neg.max(axis=1, keepdims=True)
        p = np.exp(neg)
        p /= p.sum(axis=1, keepdims=True) + 1e-15
        H = -np.sum(p * np.log(p + 1e-15), axis=1)
        perp = np.exp(H)
        converged = np.abs(perp - target_perp) < tol
        widen  = ~converged & (perp <  target_perp)
        narrow = ~converged & (perp >= target_perp)
        lo[s]  = np.where(widen,  sigma[s], lo[s])
        hi[s]  = np.where(narrow, sigma[s], hi[s])
        active[s] = ~converged
    return sigma


class ADODDetector(BaseOutlierDetector):
    """
    Adaptive Density Outlier Detection (Qian et al., IEEE ICDM 2024).
    CPU reimplementation using sklearn NearestNeighbors (original uses GPU FAISS).
    Class IV: neighbourhood-graph sufficient statistic → sensitive to subsampling.
    https://ieeexplore.ieee.org/document/10884235 | GitHub: Qian-Lily/ADOD
    """

    def __init__(self, contamination: float = 0.10, random_state: int = 42,
                 perplexity: Optional[float] = None, probability: float = 0.999,
                 n_neighbors: Optional[int] = None):
        super().__init__(contamination, random_state)
        self.perplexity = perplexity
        self.probability = probability
        self.n_neighbors = n_neighbors
        self._X_train: Optional[np.ndarray] = None
        self._densities: Optional[np.ndarray] = None
        self._boundaries: Optional[np.ndarray] = None
        self._nn: Optional[NearestNeighbors] = None

    @staticmethod
    def _resolve(n: int, perplexity, n_neighbors):
        perp = perplexity if perplexity is not None else max(2.0, float(np.sqrt(n)))
        k    = n_neighbors if n_neighbors is not None else min(int(perp) + 1, 50, n - 1)
        return perp, max(2, k)

    def fit(self, X: np.ndarray) -> "ADODDetector":
        from scipy.stats import norm as _norm
        X = np.asarray(X, dtype=float)
        n = len(X)
        perp, k = self._resolve(n, self.perplexity, self.n_neighbors)
        if n <= k + 1:
            k = max(1, n - 2)
        self._nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
        self._nn.fit(X)
        dists, _ = self._nn.kneighbors(X)
        dists = dists[:, 1:]
        sigmas    = _adod_sigmas_batch(dists ** 2, perp)
        boundaries = np.maximum(sigmas * _norm.ppf(self.probability), 1e-10)
        in_bnd     = dists <= boundaries[:, None]
        counts     = in_bnd.sum(axis=1).astype(float)
        self._X_train    = X
        self._boundaries = boundaries
        self._densities  = (counts + 1.0) / boundaries
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        from scipy.stats import norm as _norm
        X = np.asarray(X, dtype=float)
        n_train = len(self._X_train)
        perp, _ = self._resolve(n_train, self.perplexity, self.n_neighbors)
        dists, idx = self._nn.kneighbors(X)
        dists = dists[:, 1:]
        idx   = idx[:, 1:]
        sigmas     = _adod_sigmas_batch(dists ** 2, perp)
        boundaries = np.maximum(sigmas * _norm.ppf(self.probability), 1e-10)
        in_bnd     = dists <= boundaries[:, None]
        counts     = in_bnd.sum(axis=1).astype(float)
        densities  = (counts + 1.0) / boundaries
        scores = np.empty(len(X))
        for i in range(len(X)):
            nbr_mask = in_bnd[i]
            nd = self._densities[idx[i][nbr_mask]] if nbr_mask.any() else self._densities[idx[i]]
            scores[i] = nd.mean() / (densities[i] + 1e-12)
        return scores


# ---------------------------------------------------------------------------
# Class IV — ROD  (Almardeny et al., IEEE TKDE 2020)
# ---------------------------------------------------------------------------

class RODDetector(BaseOutlierDetector):
    """
    Rotation-based Outlier Detection (Almardeny et al., IEEE TKDE 2020).
    doi:10.1109/TKDE.2020.3036524
    ROD enumerates C(d,3) 3-D subspaces; infeasible for large d.
    When d > _MAX_DIM we project to _MAX_DIM PCA components first.
    """

    _MAX_DIM = 5   # C(5,3)=10 subspaces — ~6 s/fit at n=5000

    def __init__(self, contamination: float = 0.10, random_state: int = 42):
        super().__init__(contamination, random_state)
        self._model = None
        self._pca   = None

    def fit(self, X: np.ndarray) -> "RODDetector":
        from pyod.models.rod import ROD  # type: ignore
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        if d > self._MAX_DIM:
            from sklearn.decomposition import PCA
            n_comp = min(self._MAX_DIM, n - 1, d)
            if n_comp < 3:
                self._model = None; self._pca = None
                self._fitted = True
                return self
            self._pca = PCA(n_components=n_comp, random_state=self.random_state)
            X = self._pca.fit_transform(X)
        self._model = ROD(contamination=self.contamination)
        self._model.fit(X)
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self._model is None:
            return np.zeros(len(X))
        X = np.asarray(X, dtype=float)
        if self._pca is not None:
            X = self._pca.transform(X)
        return self._model.decision_function(X)


# ---------------------------------------------------------------------------
# Class IV — ABOD  (Kriegel et al., KDD 2008) — pure numpy FastABOD
# ---------------------------------------------------------------------------

class ABODDetector(BaseOutlierDetector):
    """
    Angle-Based Outlier Detection (Kriegel, Schubert, Zimek — KDD 2008).
    FastABOD: k-NN approximation. Score = VAR of angle-weights; outliers
    have LOW variance → negate so higher output = more anomalous.
    doi:10.1145/1401890.1401946
    """

    def __init__(self, contamination: float = 0.10, random_state: int = 42,
                 n_neighbors: int = 10):
        super().__init__(contamination, random_state)
        self.n_neighbors = n_neighbors
        self._X_train: Optional[np.ndarray] = None
        self._nn: Optional[NearestNeighbors] = None

    def fit(self, X: np.ndarray) -> "ABODDetector":
        X = np.asarray(X, dtype=float)
        k = min(self.n_neighbors, len(X) - 1)
        self._nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
        self._nn.fit(X)
        self._X_train = X
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        _, indices = self._nn.kneighbors(X)
        indices = indices[:, 1:]
        scores = np.empty(len(X))
        for i in range(len(X)):
            diff = self._X_train[indices[i]] - X[i]
            dist_sq = (diff * diff).sum(axis=1)
            nz = dist_sq > 0
            d = diff[nz]; ds = dist_sq[nz]
            if len(d) < 2:
                scores[i] = 0.0
                continue
            dots  = d @ d.T
            denom = np.outer(ds, ds)
            aw = dots / (denom + 1e-15)
            upper = aw[np.triu_indices(len(d), k=1)]
            scores[i] = np.var(upper) if len(upper) else 0.0
        return -scores


# ---------------------------------------------------------------------------
# Class IV — Feature Bagging  (Lazarevic & Kumar, KDD 2005) — LOF ensemble
# ---------------------------------------------------------------------------

class FeatureBaggingDetector(BaseOutlierDetector):
    """
    Feature Bagging for Outlier Detection (Lazarevic & Kumar — KDD 2005).
    Ensemble of LOF detectors on random feature subsets; scores averaged.
    doi:10.1145/1081870.1081891
    """

    def __init__(self, contamination: float = 0.10, random_state: int = 42,
                 n_estimators: int = 10, max_features: float = 0.5,
                 n_neighbors: int = 20):
        super().__init__(contamination, random_state)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.n_neighbors  = n_neighbors
        self._feature_sets: list = []
        self._lofs: list = []

    def fit(self, X: np.ndarray) -> "FeatureBaggingDetector":
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        n_feat = max(1, int(np.ceil(self.max_features * d)))
        k = min(self.n_neighbors, n - 1)
        rng = np.random.default_rng(self.random_state)
        self._feature_sets = []
        self._lofs = []
        for _ in range(self.n_estimators):
            feats = np.sort(rng.choice(d, size=n_feat, replace=False))
            lof = LocalOutlierFactor(n_neighbors=k,
                                     contamination=self.contamination,
                                     novelty=True)
            lof.fit(X[:, feats])
            self._feature_sets.append(feats)
            self._lofs.append(lof)
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        scores = np.zeros(len(X))
        for feats, lof in zip(self._feature_sets, self._lofs):
            scores += -lof.score_samples(X[:, feats])
        return scores / len(self._lofs)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

def _autoencoder_available() -> bool:
    """Return True if AutoEncoder's torch backend is importable."""
    try:
        import torch  # noqa: F401
        from pyod.models.auto_encoder import AutoEncoder  # noqa: F401
        return True
    except ImportError:
        return False


_AE_AVAILABLE: bool = _autoencoder_available()

if not _AE_AVAILABLE:
    import warnings as _warnings
    _warnings.warn(
        "AutoEncoder disabled: torch not installed (no cmake / numba-free install). "
        "Run 'brew install cmake && pip install numba torch' to enable it.",
        ImportWarning,
        stacklevel=1,
    )

METHOD_CLASS_MAP: dict[str, str] = {
    "ThreeSigma": "I",
    "BoxPlot": "I",
    "MAD": "I",
    "ChiSquare": "I",
    "COPOD": "II",
    "ECOD": "II",
    "HBOS": "II",
    "IForest": "III",
    "LOF": "IV",
    "KNN": "IV",
    "OCSVM": "IV",
    "Mahalanobis": "IV",
    "KMeans": "IV",
    "EIF": "III",
    "ROD": "IV",
    "ABOD": "IV",
    "FeatureBagging": "IV",
    "ADOD": "IV",
    **({} if not _AE_AVAILABLE else {"AutoEncoder": "IV"}),
}

ALL_METHODS = list(METHOD_CLASS_MAP.keys())

CLASS_I_METHODS = [m for m, c in METHOD_CLASS_MAP.items() if c == "I"]
CLASS_II_METHODS = [m for m, c in METHOD_CLASS_MAP.items() if c == "II"]
CLASS_III_METHODS = [m for m, c in METHOD_CLASS_MAP.items() if c == "III"]
CLASS_IV_METHODS = [m for m, c in METHOD_CLASS_MAP.items() if c == "IV"]


def make_detector(
    method_name: str,
    contamination: float = 0.10,
    random_state: int = 42,
    **kwargs,
) -> BaseOutlierDetector:
    """
    Factory function: instantiate a detector by name.

    Parameters
    ----------
    method_name : str
        One of the keys in METHOD_CLASS_MAP.
    contamination : float
    random_state : int
    **kwargs : passed to the constructor

    Returns
    -------
    BaseOutlierDetector instance (not yet fitted)
    """
    constructors: dict[str, type] = {
        "ThreeSigma": ThreeSigmaDetector,
        "BoxPlot": BoxPlotDetector,
        "MAD": MADDetector,
        "ChiSquare": ChiSquareDetector,
        "COPOD": COPODDetector,
        "ECOD": ECODDetector,
        "HBOS": HBOSDetector,
        "IForest": IForestDetector,
        "LOF": LOFDetector,
        "KNN": KNNDetector,
        "OCSVM": OCSVMDetector,
        "Mahalanobis": MahalanobisDetector,
        "KMeans": KMeansDetector,
        "EIF": EIFDetector,
        "ROD": RODDetector,
        "ABOD": ABODDetector,
        "FeatureBagging": FeatureBaggingDetector,
        "ADOD": ADODDetector,
        "AutoEncoder": AutoEncoderDetector,
    }
    if method_name not in constructors:
        raise ValueError(
            f"Unknown method '{method_name}'. "
            f"Choose from: {sorted(constructors.keys())}"
        )
    cls = constructors[method_name]
    return cls(contamination=contamination, random_state=random_state, **kwargs)
