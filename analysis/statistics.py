"""
statistics.py

Statistical analysis module for the population analysis framework.
Provides Friedman test, Wilcoxon signed-rank test,
Holm post-hoc correction, and summary statistics.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


def compute_summary(results: List[float]) -> Dict[str, float]:
    """
    Compute summary statistics for a list of results.

    Parameters
    ----------
    results : list of float
        Best fitness values from multiple runs.

    Returns
    -------
    dict
        mean, median, std, best, worst.
    """

    arr = np.array(results)

    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0),
        "best": float(np.min(arr)),
        "worst": float(np.max(arr)),
        "count": len(arr),
    }


def friedman_test(data: np.ndarray) -> Tuple[float, float]:
    """
    Perform the Friedman test.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_problems, n_algorithms).
        Each row is a problem, each column is an algorithm.
        Values are typically mean fitness or ranks.

    Returns
    -------
    (statistic, p_value)
    """

    stat, p_value = stats.friedmanchisquare(
        *[data[:, i] for i in range(data.shape[1])]
    )

    return float(stat), float(p_value)


def wilcoxon_test(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    Perform the Wilcoxon signed-rank test between two algorithms.

    Parameters
    ----------
    x : np.ndarray
        Results of algorithm A (one per problem/run).
    y : np.ndarray
        Results of algorithm B.
    alternative : str
        "two-sided", "less", "greater".

    Returns
    -------
    (statistic, p_value)
    """

    # Remove ties (identical values)
    diff = x - y
    mask = diff != 0

    if mask.sum() < 1:
        return 0.0, 1.0

    stat, p_value = stats.wilcoxon(
        x[mask],
        y[mask],
        alternative=alternative,
    )

    return float(stat), float(p_value)


def pairwise_wilcoxon(
    data: np.ndarray,
    algorithm_names: List[str],
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Perform pairwise Wilcoxon tests between all algorithm pairs.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_problems, n_algorithms).
    algorithm_names : list of str

    Returns
    -------
    dict of dict
        result[alg_a][alg_b] = (statistic, p_value)
    """

    n_algs = data.shape[1]
    results = {}

    for i in range(n_algs):
        results[algorithm_names[i]] = {}

        for j in range(n_algs):

            if i == j:
                results[algorithm_names[i]][algorithm_names[j]] = (
                    0.0, 1.0
                )
                continue

            stat, p_val = wilcoxon_test(data[:, i], data[:, j])
            results[algorithm_names[i]][algorithm_names[j]] = (
                stat, p_val
            )

    return results


def holm_posthoc(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Tuple[int, float, float, bool]]:
    """
    Apply the Holm step-down post-hoc correction.

    Parameters
    ----------
    p_values : list of float
        Raw p-values from pairwise comparisons.
    alpha : float
        Significance level.

    Returns
    -------
    list of (original_index, p_value, adjusted_alpha, is_significant)
    """

    n = len(p_values)

    # Sort by p-value
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    results = []

    for rank, (idx, p_val) in enumerate(indexed):

        adjusted_alpha = alpha / (n - rank)

        is_significant = p_val < adjusted_alpha

        results.append((idx, p_val, adjusted_alpha, is_significant))

    return results


def average_rank(data: np.ndarray) -> np.ndarray:
    """
    Compute average rank of each algorithm across problems.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_problems, n_algorithms).
        Lower values = better.

    Returns
    -------
    np.ndarray
        Average ranks of shape (n_algorithms,).
    """

    n_problems, n_algs = data.shape

    ranks = np.zeros_like(data, dtype=float)

    for i in range(n_problems):
        ranks[i] = stats.rankdata(data[i])

    return ranks.mean(axis=0)
