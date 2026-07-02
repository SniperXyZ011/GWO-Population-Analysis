"""
benchmark_factory.py

Factory for creating benchmark instances from the registry.

Includes an optional LRU cache to avoid reloading opfunu
data files for repeated (benchmark, function, dimension) keys.
"""

import logging
from typing import Dict, Tuple, Any

from core.registry import benchmark_registry

logger = logging.getLogger(__name__)


class BenchmarkFactory:

    # Cache keyed by (benchmark_name, function, dimension)
    _cache: Dict[Tuple[str, int, int], Any] = {}

    @staticmethod
    def create(experiment):
        """
        Create a benchmark problem instance for the given experiment.

        Uses a cache to avoid re-instantiating the same benchmark
        problem. This is safe because benchmark evaluate() methods
        are stateless (no internal counters or side effects).

        Parameters
        ----------
        experiment : Experiment

        Returns
        -------
        BaseProblem subclass instance
        """

        key = (
            experiment.benchmark,
            experiment.function,
            experiment.dimension,
        )

        if key in BenchmarkFactory._cache:
            return BenchmarkFactory._cache[key]

        benchmark_cls = benchmark_registry.get(
            experiment.benchmark
        )

        problem = benchmark_cls(
            function=experiment.function,
            dimension=experiment.dimension,
        )

        BenchmarkFactory._cache[key] = problem

        logger.debug(
            f"Cached benchmark: {experiment.benchmark} "
            f"F{experiment.function} D{experiment.dimension}"
        )

        return problem

    @staticmethod
    def clear_cache():
        """Clear the benchmark cache (useful in tests)."""
        BenchmarkFactory._cache.clear()

    @staticmethod
    def cache_size() -> int:
        """Return the number of cached benchmark instances."""
        return len(BenchmarkFactory._cache)
