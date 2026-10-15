"""
resilience_metrics.py
---------------------
Dice-based and rank-based resilience metrics.

Dice resilience (rho):
    rho = 2 * |O_S ∩ O_full[S]| / (|O_S| + |O_full[S]|)
    where O_S = top-k outliers on the sample, O_full[S] = restriction of
    full-dataset top-k outliers to records in S.

Rank-based resilience (rho_tilde):
    rho_tilde = Spearman(scores_sample, scores_full_restricted)
    Threshold-free: compares score rankings over the common record set S.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def dice_resilience(
    labels_sample: np.ndarray,
    labels_full_restricted: np.ndarray,
) -> float:
    """
    Compute Dice-based resilience rho.

    Parameters
    ----------
    labels_sample : np.ndarray of shape (n,), dtype int {0,1}
        Binary outlier labels assigned by the method trained on S alone.
        1 = outlier, 0 = inlier.
    labels_full_restricted : np.ndarray of shape (n,), dtype int {0,1}
        Binary outlier labels from the method trained on full D, restricted
        to records in S.

    Returns
    -------
    float
        Dice coefficient in [0, 1].  Returns np.nan if both sets are empty.
    """
    labels_sample = np.asarray(labels_sample, dtype=int)
    labels_full_restricted = np.asarray(labels_full_restricted, dtype=int)

    if labels_sample.shape != labels_full_restricted.shape:
        raise ValueError(
            f"Shape mismatch: labels_sample {labels_sample.shape} vs "
            f"labels_full_restricted {labels_full_restricted.shape}"
        )

    a = np.sum(labels_sample)
    b = np.sum(labels_full_restricted)
    intersection = np.sum((labels_sample == 1) & (labels_full_restricted == 1))

    denom = a + b
    if denom == 0:
        return np.nan  # both sets empty: undefined

    return float(2.0 * intersection / denom)


def rank_resilience(
    scores_sample: np.ndarray,
    scores_full_restricted: np.ndarray,
) -> float:
    """
    Compute rank-based resilience rho_tilde (Spearman correlation).

    Parameters
    ----------
    scores_sample : np.ndarray of shape (n,)
        Outlier scores from the method trained on S alone, indexed over S.
    scores_full_restricted : np.ndarray of shape (n,)
        Outlier scores from the method trained on full D, restricted to S.

    Returns
    -------
    float
        Spearman correlation coefficient in [-1, 1].  Returns np.nan if
        fewer than 3 observations or constant input.
    """
    scores_sample = np.asarray(scores_sample, dtype=float)
    scores_full_restricted = np.asarray(scores_full_restricted, dtype=float)

    if scores_sample.shape != scores_full_restricted.shape:
        raise ValueError(
            f"Shape mismatch: scores_sample {scores_sample.shape} vs "
            f"scores_full_restricted {scores_full_restricted.shape}"
        )

    n = len(scores_sample)
    if n < 3:
        return np.nan

    if np.std(scores_sample) == 0 or np.std(scores_full_restricted) == 0:
        # Constant scores: Spearman undefined; treat as nan
        return np.nan

    result = stats.spearmanr(scores_sample, scores_full_restricted)
    corr = result.statistic if hasattr(result, "statistic") else result.correlation
    return float(corr)


def scores_to_labels(scores: np.ndarray, contamination: float = 0.10) -> np.ndarray:
    """
    Convert continuous scores to binary labels using top-k threshold.

    Parameters
    ----------
    scores : np.ndarray of shape (n,)
        Outlier scores (higher = more anomalous).
    contamination : float
        Fraction of records to label as outliers.

    Returns
    -------
    np.ndarray of shape (n,), dtype int {0,1}
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    k = max(1, int(np.floor(contamination * n)))
    threshold = np.sort(scores)[-k]
    labels = (scores >= threshold).astype(int)
    return labels


def compute_resilience_pair(
    scores_sample: np.ndarray,
    scores_full_restricted: np.ndarray,
    contamination: float = 0.10,
) -> dict:
    """
    Compute both rho and rho_tilde from raw scores.

    Parameters
    ----------
    scores_sample : np.ndarray
        Outlier scores from method fitted on S.
    scores_full_restricted : np.ndarray
        Outlier scores from method fitted on D, restricted to S.
    contamination : float
        Used to threshold scores into binary labels for Dice resilience.

    Returns
    -------
    dict with keys: rho (float), rho_tilde (float)
    """
    labels_s = scores_to_labels(scores_sample, contamination)
    labels_f = scores_to_labels(scores_full_restricted, contamination)

    rho = dice_resilience(labels_s, labels_f)
    rho_tilde = rank_resilience(scores_sample, scores_full_restricted)

    return {"rho": rho, "rho_tilde": rho_tilde}
