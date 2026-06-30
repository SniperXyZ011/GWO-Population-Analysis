"""
benchmark_factory.py

Factory for creating benchmark instances from the registry.
"""

from core.registry import benchmark_registry


class BenchmarkFactory:

    @staticmethod
    def create(experiment):
        """
        Create a benchmark problem instance for the given experiment.

        Parameters
        ----------
        experiment : Experiment

        Returns
        -------
        BaseProblem subclass instance
        """

        benchmark_cls = benchmark_registry.get(
            experiment.benchmark
        )

        problem = benchmark_cls(
            function=experiment.function,
            dimension=experiment.dimension,
        )

        return problem
