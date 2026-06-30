"""
plots.py

Visualization module for the population analysis framework.
Generates publication-quality plots using matplotlib and seaborn.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional


# Publication-quality defaults
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_convergence(
    curves: Dict[str, List[float]],
    title: str = "Convergence Curve",
    save_path: Optional[str] = None,
    log_scale: bool = True,
):
    """
    Plot convergence curves for multiple algorithms.

    Parameters
    ----------
    curves : dict
        {algorithm_name: [fitness_values_per_iteration]}
    title : str
    save_path : str, optional
    log_scale : bool
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, curve in curves.items():
        ax.plot(curve, label=name, linewidth=1.5)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    ax.set_title(title)

    if log_scale:
        ax.set_yscale("log")

    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.close(fig)


def plot_population_vs_fitness(
    populations: List[int],
    mean_fitness: Dict[str, List[float]],
    title: str = "Population Size vs Mean Fitness",
    save_path: Optional[str] = None,
    log_scale: bool = True,
):
    """
    Plot mean fitness vs population size for each algorithm.

    Parameters
    ----------
    populations : list of int
    mean_fitness : dict
        {algorithm_name: [mean_fitness_per_pop_size]}
    title : str
    save_path : str, optional
    log_scale : bool
    """

    fig, ax = plt.subplots(figsize=(12, 6))

    for name, values in mean_fitness.items():
        ax.plot(
            populations[:len(values)],
            values,
            marker="o",
            label=name,
            linewidth=1.5,
            markersize=5,
        )

    ax.set_xlabel("Population Size")
    ax.set_ylabel("Mean Fitness")
    ax.set_title(title)

    if log_scale:
        ax.set_yscale("log")

    ax.set_xscale("log")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.close(fig)


def plot_dimension_vs_fitness(
    dimensions: List[int],
    mean_fitness: Dict[str, List[float]],
    title: str = "Dimension vs Mean Fitness",
    save_path: Optional[str] = None,
    log_scale: bool = True,
):
    """
    Plot mean fitness vs dimension for each algorithm.
    """

    fig, ax = plt.subplots(figsize=(12, 6))

    for name, values in mean_fitness.items():
        ax.plot(
            dimensions[:len(values)],
            values,
            marker="s",
            label=name,
            linewidth=1.5,
            markersize=5,
        )

    ax.set_xlabel("Dimension")
    ax.set_ylabel("Mean Fitness")
    ax.set_title(title)

    if log_scale:
        ax.set_yscale("log")

    ax.set_xscale("log")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.close(fig)


def plot_heatmap(
    data: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str = "Heatmap",
    save_path: Optional[str] = None,
    fmt: str = ".2e",
    cmap: str = "YlOrRd_r",
):
    """
    Plot a heatmap (e.g., Population × Dimension).

    Parameters
    ----------
    data : np.ndarray
        2D array of values.
    row_labels : list of str
    col_labels : list of str
    title : str
    save_path : str, optional
    fmt : str
        Number format.
    cmap : str
        Colormap name.
    """

    fig, ax = plt.subplots(
        figsize=(max(8, len(col_labels) * 0.8),
                 max(6, len(row_labels) * 0.5))
    )

    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        xticklabels=col_labels,
        yticklabels=row_labels,
        cmap=cmap,
        ax=ax,
        linewidths=0.5,
    )

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.close(fig)


def plot_boxplot(
    data: Dict[str, List[float]],
    title: str = "Fitness Distribution",
    save_path: Optional[str] = None,
    log_scale: bool = False,
):
    """
    Plot box plots comparing algorithm distributions.

    Parameters
    ----------
    data : dict
        {algorithm_name: [fitness_values_across_runs]}
    title : str
    save_path : str, optional
    """

    fig, ax = plt.subplots(figsize=(12, 6))

    labels = list(data.keys())
    values = [data[k] for k in labels]

    bp = ax.boxplot(
        values,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 6},
    )

    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Fitness")
    ax.set_title(title)

    if log_scale:
        ax.set_yscale("log")

    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.close(fig)


def plot_violin(
    data: Dict[str, List[float]],
    title: str = "Fitness Distribution (Violin)",
    save_path: Optional[str] = None,
):
    """
    Plot violin plots comparing algorithm distributions.
    """

    fig, ax = plt.subplots(figsize=(12, 6))

    labels = list(data.keys())
    values = [data[k] for k in labels]

    parts = ax.violinplot(
        values,
        positions=range(len(labels)),
        showmeans=True,
        showmedians=True,
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Fitness")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.close(fig)


def plot_ranking_chart(
    algorithm_names: List[str],
    average_ranks: List[float],
    title: str = "Average Ranking",
    save_path: Optional[str] = None,
):
    """
    Plot a horizontal bar chart of average ranks.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by rank
    sorted_pairs = sorted(
        zip(algorithm_names, average_ranks),
        key=lambda x: x[1],
    )

    names = [p[0] for p in sorted_pairs]
    ranks = [p[1] for p in sorted_pairs]

    colors = plt.cm.viridis(
        np.linspace(0.2, 0.8, len(names))
    )

    bars = ax.barh(names, ranks, color=colors, edgecolor="gray")

    # Add rank values
    for bar, rank in zip(bars, ranks):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{rank:.2f}",
            va="center",
            fontsize=10,
        )

    ax.set_xlabel("Average Rank")
    ax.set_title(title)
    ax.invert_yaxis()  # Best (lowest rank) on top
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.close(fig)
