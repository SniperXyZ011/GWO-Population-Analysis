"""
runner.py

Executes a single optimization experiment.
Contains no algorithm-specific logic.
"""

import logging

from core.optimizer_factory import OptimizerFactory
from core.benchmark_factory import BenchmarkFactory
from core.result_manager import ResultManager

from configs.config import RESULTS_DIR, OVERWRITE_EXISTING


logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Orchestrates the entire optimization process:
        1. Create benchmark problem
        2. Create optimizer
        3. Run optimization
        4. Save results
    """

    def __init__(self, results_dir=None, overwrite=None):

        self.result_manager = ResultManager(
            results_dir or RESULTS_DIR
        )

        self.overwrite = (
            overwrite if overwrite is not None
            else OVERWRITE_EXISTING
        )

    # ---------------------------------------------------------
    # Execute a single experiment
    # ---------------------------------------------------------

    def run(self, experiment):
        """
        Execute one optimization experiment.

        Parameters
        ----------
        experiment : Experiment

        Returns
        -------
        dict
            Result dictionary from the optimizer.
        """

        logger.info(f"START: {experiment}")

        # 1. Create benchmark problem
        problem = BenchmarkFactory.create(experiment)

        # 2. Create optimizer
        optimizer = OptimizerFactory.create(
            experiment, problem
        )

        # 3. Run optimization (timing handled inside optimize)
        result = optimizer.optimize()

        # Pop the massive convergence curve array to avoid IPC overhead
        curve = result.pop("convergence_curve", [])

        # 4. Save results
        self.result_manager.save_result(experiment, result)

        self.result_manager.save_convergence(experiment, curve)

        logger.info(
            f"DONE: {experiment} | "
            f"Score={result['best_score']:.6e} | "
            f"FE={result['function_evaluations']} | "
            f"Time={result['execution_time']:.2f}s | "
            f"FE/s={result.get('fe_per_second', 0):.0f}"
        )

        return result
