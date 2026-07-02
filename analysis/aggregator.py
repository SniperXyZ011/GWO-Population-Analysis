"""
aggregator.py

Result aggregation engine.

Bridges the checkpoint database with the statistical analysis
and visualization modules. Queries raw results from SQLite,
aggregates across runs, and produces analysis-ready data structures.

This is the central analysis API:
    aggregator = ResultAggregator(checkpoint_db)
    summary = aggregator.summary_table("CEC2020", dimension=30)
    friedman = aggregator.friedman_analysis("CEC2020", dimension=30)
    wtl = aggregator.win_tie_loss("GWO", "CEC2020", dimension=30)
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from analysis.statistics import (
    compute_summary,
    friedman_test,
    pairwise_wilcoxon,
    holm_posthoc,
    average_rank,
)
from analysis.ranking import (
    compute_average_ranks,
    critical_difference,
)

logger = logging.getLogger(__name__)


class ResultAggregator:
    """
    Aggregates raw results from CheckpointDB into analysis-ready tables.

    All methods accept filter parameters (benchmark, dimension, etc.)
    and return structured dictionaries or numpy arrays suitable for
    direct input to statistics and visualization functions.

    Parameters
    ----------
    checkpoint_db : CheckpointDB
        The checkpoint database to read results from.
    """

    def __init__(self, checkpoint_db):
        self.db = checkpoint_db

    # =========================================================
    # Summary Statistics
    # =========================================================

    def summary_table(
        self,
        benchmark: str,
        dimension: int,
        population_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a summary statistics table.

        Groups results by (optimizer, function) and computes
        mean, std, median, best, worst across runs.

        Parameters
        ----------
        benchmark : str
        dimension : int
        population_size : int, optional
            If None, uses all population sizes.

        Returns
        -------
        dict
            {
                "optimizers": [...],
                "functions": [...],
                "data": {optimizer: {function: {mean, std, ...}}},
                "overall": {optimizer: {mean, std, ...}},
            }
        """

        results = self.db.query_results(
            benchmark=benchmark,
            dimension=dimension,
            population_size=population_size,
        )

        if not results:
            logger.warning(
                f"No results for {benchmark} D{dimension}"
            )
            return {"optimizers": [], "functions": [], "data": {}, "overall": {}}

        # Group by (optimizer, function)
        grouped = defaultdict(lambda: defaultdict(list))
        for r in results:
            grouped[r["optimizer"]][r["function"]].append(
                r["best_score"]
            )

        optimizers = sorted(grouped.keys())
        functions = sorted({r["function"] for r in results})

        data = {}
        overall = {}

        for opt in optimizers:
            data[opt] = {}
            all_scores = []

            for func in functions:
                scores = grouped[opt].get(func, [])
                if scores:
                    data[opt][func] = compute_summary(scores)
                    all_scores.extend(scores)

            if all_scores:
                overall[opt] = compute_summary(all_scores)

        return {
            "optimizers": optimizers,
            "functions": functions,
            "data": data,
            "overall": overall,
        }

    # =========================================================
    # Friedman Analysis
    # =========================================================

    def friedman_analysis(
        self,
        benchmark: str,
        dimension: int,
        population_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run Friedman test + Holm post-hoc across all optimizers.

        Uses mean fitness per function as the data matrix.

        Parameters
        ----------
        benchmark : str
        dimension : int
        population_size : int, optional

        Returns
        -------
        dict
            {
                "statistic": float,
                "p_value": float,
                "significant": bool,
                "rankings": [(name, avg_rank), ...],
                "critical_difference": float,
                "holm_results": [...],
                "data_matrix": np.ndarray,
                "optimizers": [...],
                "functions": [...],
            }
        """

        summary = self.summary_table(benchmark, dimension, population_size)

        if len(summary["optimizers"]) < 3:
            logger.warning(
                "Friedman test requires at least 3 algorithms. "
                f"Found {len(summary['optimizers'])}."
            )
            return {"error": "Requires at least 3 algorithms"}

        optimizers = summary["optimizers"]
        functions = summary["functions"]

        # Build data matrix: (n_functions, n_optimizers)
        data_matrix = np.full(
            (len(functions), len(optimizers)), np.inf
        )

        for j, opt in enumerate(optimizers):
            for i, func in enumerate(functions):
                if func in summary["data"].get(opt, {}):
                    data_matrix[i, j] = summary["data"][opt][func]["mean"]

        # Friedman test
        try:
            stat, p_value = friedman_test(data_matrix)
        except Exception as e:
            logger.error(f"Friedman test failed: {e}")
            return {"error": str(e)}

        # Rankings
        rankings = compute_average_ranks(data_matrix, optimizers)

        # Critical difference
        cd = critical_difference(len(optimizers), len(functions))

        # Pairwise Wilcoxon + Holm
        pw = pairwise_wilcoxon(data_matrix, optimizers)

        # Collect p-values for Holm correction
        p_values = []
        pairs = []
        for i, opt_a in enumerate(optimizers):
            for j, opt_b in enumerate(optimizers):
                if i < j:
                    p_values.append(pw[opt_a][opt_b][1])
                    pairs.append((opt_a, opt_b))

        holm_results = []
        if p_values:
            raw_holm = holm_posthoc(p_values)
            for idx, p_val, adj_alpha, sig in raw_holm:
                holm_results.append({
                    "pair": pairs[idx],
                    "p_value": p_val,
                    "adjusted_alpha": adj_alpha,
                    "significant": sig,
                })

        return {
            "statistic": stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "rankings": rankings,
            "critical_difference": cd,
            "pairwise_wilcoxon": pw,
            "holm_results": holm_results,
            "data_matrix": data_matrix,
            "optimizers": optimizers,
            "functions": functions,
        }

    # =========================================================
    # Win / Tie / Loss
    # =========================================================

    def win_tie_loss(
        self,
        reference: str,
        benchmark: str,
        dimension: int,
        population_size: Optional[int] = None,
        alpha: float = 0.05,
    ) -> Dict[str, Dict[str, int]]:
        """
        Compute Win/Tie/Loss counts for each optimizer vs a reference.

        A "Win" means the reference is significantly better.
        A "Loss" means the competitor is significantly better.
        Uses Wilcoxon signed-rank test per function.

        Parameters
        ----------
        reference : str
            Reference optimizer name.
        benchmark : str
        dimension : int
        population_size : int, optional
        alpha : float
            Significance level.

        Returns
        -------
        dict
            {optimizer: {"win": n, "tie": n, "loss": n}}
        """

        results = self.db.query_results(
            benchmark=benchmark,
            dimension=dimension,
            population_size=population_size,
        )

        if not results:
            return {}

        # Group by (optimizer, function) -> list of best_scores
        grouped = defaultdict(lambda: defaultdict(list))
        for r in results:
            grouped[r["optimizer"]][r["function"]].append(
                r["best_score"]
            )

        if reference not in grouped:
            logger.warning(f"Reference optimizer '{reference}' not found")
            return {}

        functions = sorted({r["function"] for r in results})
        ref_data = grouped[reference]

        wtl = {}

        for opt in sorted(grouped.keys()):
            if opt == reference:
                continue

            win, tie, loss = 0, 0, 0

            for func in functions:
                ref_scores = ref_data.get(func, [])
                opt_scores = grouped[opt].get(func, [])

                if not ref_scores or not opt_scores:
                    tie += 1
                    continue

                ref_arr = np.array(ref_scores)
                opt_arr = np.array(opt_scores)

                # Ensure same length (take min)
                min_len = min(len(ref_arr), len(opt_arr))
                ref_arr = ref_arr[:min_len]
                opt_arr = opt_arr[:min_len]

                # Wilcoxon test: is reference better?
                diff = ref_arr - opt_arr
                nonzero = diff != 0

                if nonzero.sum() < 1:
                    tie += 1
                    continue

                from scipy.stats import wilcoxon as scipy_wilcoxon

                try:
                    _, p_val = scipy_wilcoxon(
                        ref_arr[nonzero], opt_arr[nonzero]
                    )

                    if p_val < alpha:
                        ref_mean = np.mean(ref_arr)
                        opt_mean = np.mean(opt_arr)

                        if ref_mean < opt_mean:
                            win += 1    # Reference is better
                        else:
                            loss += 1   # Competitor is better
                    else:
                        tie += 1
                except Exception:
                    tie += 1

            wtl[opt] = {"win": win, "tie": tie, "loss": loss}

        return wtl

    # =========================================================
    # Vargha-Delaney Effect Size (A-measure)
    # =========================================================

    @staticmethod
    def vargha_delaney(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the Vargha-Delaney A-measure effect size.

        A = P(X < Y) + 0.5 * P(X == Y)

        Interpretation:
            A = 0.5 → no effect (equivalent)
            A > 0.5 → x tends to be smaller (better for minimization)
            A < 0.5 → y tends to be smaller

        Magnitude thresholds (Vargha & Delaney 2000):
            |A - 0.5| < 0.06  → negligible
            |A - 0.5| < 0.14  → small
            |A - 0.5| < 0.21  → medium
            |A - 0.5| >= 0.21 → large

        Parameters
        ----------
        x : np.ndarray
            Scores from algorithm A.
        y : np.ndarray
            Scores from algorithm B.

        Returns
        -------
        float
            A-measure in [0, 1].
        """

        m, n = len(x), len(y)

        if m == 0 or n == 0:
            return 0.5

        count = 0.0

        for xi in x:
            for yj in y:
                if xi < yj:
                    count += 1.0
                elif xi == yj:
                    count += 0.5

        return count / (m * n)

    @staticmethod
    def effect_size_magnitude(a_measure: float) -> str:
        """
        Classify the Vargha-Delaney A-measure magnitude.

        Parameters
        ----------
        a_measure : float

        Returns
        -------
        str
            "negligible", "small", "medium", or "large"
        """

        diff = abs(a_measure - 0.5)

        if diff < 0.06:
            return "negligible"
        elif diff < 0.14:
            return "small"
        elif diff < 0.21:
            return "medium"
        else:
            return "large"

    def effect_size_table(
        self,
        benchmark: str,
        dimension: int,
        population_size: Optional[int] = None,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Compute pairwise Vargha-Delaney A-measure for all optimizers.

        Parameters
        ----------
        benchmark : str
        dimension : int
        population_size : int, optional

        Returns
        -------
        dict
            {opt_a: {opt_b: {"a_measure": float, "magnitude": str}}}
        """

        results = self.db.query_results(
            benchmark=benchmark,
            dimension=dimension,
            population_size=population_size,
        )

        if not results:
            return {}

        # Group by optimizer -> list of mean scores per function
        grouped = defaultdict(lambda: defaultdict(list))
        for r in results:
            grouped[r["optimizer"]][r["function"]].append(
                r["best_score"]
            )

        # Compute mean per function for each optimizer
        opt_means = {}
        for opt in grouped:
            means = []
            for func in sorted(grouped[opt].keys()):
                scores = grouped[opt][func]
                means.append(np.mean(scores))
            opt_means[opt] = np.array(means)

        optimizers = sorted(opt_means.keys())
        table = {}

        for opt_a in optimizers:
            table[opt_a] = {}
            for opt_b in optimizers:
                if opt_a == opt_b:
                    table[opt_a][opt_b] = {
                        "a_measure": 0.5,
                        "magnitude": "negligible",
                    }
                    continue

                a = self.vargha_delaney(
                    opt_means[opt_a], opt_means[opt_b]
                )
                table[opt_a][opt_b] = {
                    "a_measure": a,
                    "magnitude": self.effect_size_magnitude(a),
                }

        return table

    # =========================================================
    # Population Sensitivity
    # =========================================================

    def population_sensitivity(
        self,
        benchmark: str,
        optimizer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute mean fitness per (optimizer, population_size, dimension).

        Used for population sensitivity plots and heatmaps.

        Parameters
        ----------
        benchmark : str
        optimizer : str, optional
            If None, returns data for all optimizers.

        Returns
        -------
        dict
            {
                "optimizers": [...],
                "dimensions": [...],
                "population_sizes": [...],
                "data": {optimizer: {dimension: {pop: mean_fitness}}}
            }
        """

        results = self.db.query_results(
            benchmark=benchmark,
            optimizer=optimizer,
        )

        if not results:
            return {
                "optimizers": [], "dimensions": [],
                "population_sizes": [], "data": {}
            }

        # Group by (optimizer, dimension, population) -> list of scores
        grouped = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for r in results:
            grouped[r["optimizer"]][r["dimension"]][
                r["population_size"]
            ].append(r["best_score"])

        optimizers = sorted(grouped.keys())
        dimensions = sorted({r["dimension"] for r in results})
        populations = sorted({r["population_size"] for r in results})

        data = {}
        for opt in optimizers:
            data[opt] = {}
            for dim in dimensions:
                data[opt][dim] = {}
                for pop in populations:
                    scores = grouped[opt][dim].get(pop, [])
                    if scores:
                        data[opt][dim][pop] = float(np.mean(scores))

        return {
            "optimizers": optimizers,
            "dimensions": dimensions,
            "population_sizes": populations,
            "data": data,
        }

    # =========================================================
    # Convergence Data
    # =========================================================

    def get_convergence_curves(
        self,
        benchmark: str,
        function: int,
        dimension: int,
        population_size: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Get average convergence curves per optimizer.

        Averages across all runs for each optimizer.

        Parameters
        ----------
        benchmark : str
        function : int
        dimension : int
        population_size : int, optional

        Returns
        -------
        dict
            {optimizer: [avg_best_fitness_per_iteration]}
        """

        results = self.db.query_results(
            benchmark=benchmark,
            dimension=dimension,
            function=function,
            population_size=population_size,
        )

        if not results:
            return {}

        # For each result, fetch its convergence from DB
        curves_by_opt = defaultdict(list)

        with self.db._connection() as conn:
            for r in results:
                rows = conn.execute(
                    """SELECT iteration, best_fitness
                       FROM convergence
                       WHERE result_id = ?
                       ORDER BY iteration""",
                    (r["id"],),
                ).fetchall()

                if rows:
                    curve = [row["best_fitness"] for row in rows]
                    curves_by_opt[r["optimizer"]].append(curve)

        # Average curves per optimizer
        avg_curves = {}
        for opt, all_curves in curves_by_opt.items():
            if not all_curves:
                continue

            # Pad shorter curves with their last value
            max_len = max(len(c) for c in all_curves)
            padded = []
            for c in all_curves:
                if len(c) < max_len:
                    c = c + [c[-1]] * (max_len - len(c))
                padded.append(c)

            avg_curves[opt] = np.mean(padded, axis=0).tolist()

        return avg_curves

    # =========================================================
    # Full Analysis Pipeline
    # =========================================================

    def full_analysis(
        self,
        benchmark: str,
        dimension: int,
        population_size: Optional[int] = None,
        reference_optimizer: str = "GWO",
    ) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.

        Produces all statistical results in one call:
            - Summary table
            - Friedman test + rankings
            - Win/Tie/Loss
            - Effect sizes
            - Population sensitivity

        Parameters
        ----------
        benchmark : str
        dimension : int
        population_size : int, optional
        reference_optimizer : str

        Returns
        -------
        dict
            Complete analysis results.
        """

        logger.info(
            f"Running full analysis: {benchmark} D{dimension}"
        )

        analysis = {
            "benchmark": benchmark,
            "dimension": dimension,
            "population_size": population_size,
        }

        # Summary statistics
        analysis["summary"] = self.summary_table(
            benchmark, dimension, population_size
        )

        # Friedman test
        if len(analysis["summary"]["optimizers"]) >= 3:
            analysis["friedman"] = self.friedman_analysis(
                benchmark, dimension, population_size
            )
        else:
            analysis["friedman"] = None

        # Win/Tie/Loss
        if reference_optimizer in analysis["summary"]["optimizers"]:
            analysis["win_tie_loss"] = self.win_tie_loss(
                reference_optimizer, benchmark, dimension,
                population_size,
            )
        else:
            analysis["win_tie_loss"] = None

        # Effect sizes
        analysis["effect_sizes"] = self.effect_size_table(
            benchmark, dimension, population_size
        )

        # Population sensitivity
        analysis["population_sensitivity"] = self.population_sensitivity(
            benchmark
        )

        logger.info(
            f"Analysis complete: "
            f"{len(analysis['summary']['optimizers'])} optimizers, "
            f"{len(analysis['summary']['functions'])} functions"
        )

        return analysis
