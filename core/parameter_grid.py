"""
parameter_grid.py

Generates all experiment combinations from config.
Uses BENCHMARK_FUNCTIONS dict — no hardcoded if/elif chains.
"""

from itertools import product

from configs.config import (
    BENCHMARKS,
    BENCHMARK_FUNCTIONS,
    OPTIMIZERS,
    DIMENSIONS,
    POPULATION_SIZES,
    RUNS,
    MAX_FE_MULTIPLIER,
)

from core.experiment import Experiment


class ParameterGrid:
    """
    Generates every possible Experiment from the config.

    The runner should never manually generate combinations.
    """

    def __init__(
        self,
        benchmarks=None,
        optimizers=None,
        dimensions=None,
        populations=None,
        runs=None,
    ):
        self.benchmarks = benchmarks or BENCHMARKS
        self.optimizers = optimizers or OPTIMIZERS
        self.dimensions = dimensions or DIMENSIONS
        self.populations = populations or POPULATION_SIZES
        self.runs = runs or RUNS

    def generate(self):
        """
        Generate every experiment.

        Yields
        ------
        Experiment
            One Experiment per unique combination.
        """

        for (
            benchmark,
            optimizer,
            dimension,
            population,
            run,
        ) in product(
            self.benchmarks,
            self.optimizers,
            self.dimensions,
            self.populations,
            range(1, self.runs + 1),
        ):

            max_fe = dimension * MAX_FE_MULTIPLIER

            num_functions = BENCHMARK_FUNCTIONS.get(benchmark)

            if num_functions is None:
                raise ValueError(
                    f"Unknown benchmark: {benchmark}. "
                    f"Add it to BENCHMARK_FUNCTIONS in config.py."
                )

            for function in range(1, num_functions + 1):

                yield Experiment(
                    benchmark=benchmark,
                    function=function,
                    optimizer=optimizer,
                    dimension=dimension,
                    population_size=population,
                    max_function_evaluations=max_fe,
                    run=run,
                    seed=run,
                )

    def __len__(self):

        total = 0

        for benchmark in self.benchmarks:

            num_functions = BENCHMARK_FUNCTIONS.get(benchmark, 0)

            total += (
                num_functions
                * len(self.optimizers)
                * len(self.dimensions)
                * len(self.populations)
                * self.runs
            )

        return total

    def __repr__(self):
        return (
            f"ParameterGrid("
            f"benchmarks={len(self.benchmarks)}, "
            f"optimizers={len(self.optimizers)}, "
            f"dimensions={len(self.dimensions)}, "
            f"populations={len(self.populations)}, "
            f"runs={self.runs}, "
            f"total={len(self)})"
        )
