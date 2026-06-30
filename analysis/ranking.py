"""
ranking.py

Ranking analysis module.
Provides average ranking computation and
critical difference calculation for Nemenyi test.
"""

import numpy as np
from scipy import stats
from typing import List, Tuple


def compute_average_ranks(
    data: np.ndarray,
    algorithm_names: List[str],
) -> List[Tuple[str, float]]:
    """
    Compute and sort algorithms by average rank.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_problems, n_algorithms).
        Lower values = better.
    algorithm_names : list of str

    Returns
    -------
    list of (algorithm_name, average_rank)
        Sorted from best (lowest rank) to worst.
    """

    n_problems, n_algs = data.shape

    ranks = np.zeros_like(data, dtype=float)

    for i in range(n_problems):
        ranks[i] = stats.rankdata(data[i])

    avg_ranks = ranks.mean(axis=0)

    result = list(zip(algorithm_names, avg_ranks))
    result.sort(key=lambda x: x[1])

    return result


def critical_difference(
    n_algorithms: int,
    n_problems: int,
    alpha: float = 0.05,
) -> float:
    """
    Compute the Nemenyi critical difference.

    CD = q_alpha * sqrt(k(k+1) / (6*N))

    where k = number of algorithms, N = number of problems,
    q_alpha is the critical value from the studentized range
    distribution.

    Parameters
    ----------
    n_algorithms : int
    n_problems : int
    alpha : float

    Returns
    -------
    float
        Critical difference value.
    """

    k = n_algorithms
    N = n_problems

    # Approximate q_alpha values for Nemenyi test
    # For common cases (alpha=0.05)
    q_alpha_table = {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
        11: 3.219,
        12: 3.268,
    }

    q_alpha = q_alpha_table.get(k, 2.0 + 0.1 * k)

    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))

    return cd


def rank_comparison_table(
    data: np.ndarray,
    algorithm_names: List[str],
    dimension_labels: List[str] = None,
) -> dict:
    """
    Create a rank comparison table.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_problems, n_algorithms).
    algorithm_names : list of str
    dimension_labels : list of str, optional

    Returns
    -------
    dict
        Table with per-problem ranks and averages.
    """

    n_problems, n_algs = data.shape

    ranks = np.zeros_like(data, dtype=float)
    for i in range(n_problems):
        ranks[i] = stats.rankdata(data[i])

    table = {
        "algorithms": algorithm_names,
        "ranks_per_problem": ranks.tolist(),
        "average_ranks": ranks.mean(axis=0).tolist(),
    }

    if dimension_labels:
        table["problem_labels"] = dimension_labels

    return table
