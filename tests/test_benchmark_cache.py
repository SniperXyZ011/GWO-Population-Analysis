"""
test_benchmark_cache.py

Tests for benchmark caching in BenchmarkFactory.
"""

import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.benchmark_factory import BenchmarkFactory
from core.experiment import Experiment
from core.registry import benchmark_registry
from benchmarks.base_problem import BaseProblem


# Register a test benchmark if not already registered
class _CacheTestBenchmark(BaseProblem):
    """Minimal benchmark for cache testing."""

    def __init__(self, function: int, dimension: int):
        super().__init__(dimension)
        self.function = function

    def evaluate(self, solution):
        return float(np.sum(solution ** 2))

    def lower_bound(self):
        return np.full(self.dimension, -100.0)

    def upper_bound(self):
        return np.full(self.dimension, 100.0)

    def optimum(self):
        return 0.0

    def function_name(self):
        return f"F{self.function}"

    def benchmark_name(self):
        return "_CacheTestBenchmark"


if not benchmark_registry.exists("_CacheTestBenchmark"):
    benchmark_registry.register(_CacheTestBenchmark)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the benchmark cache before and after each test."""
    BenchmarkFactory.clear_cache()
    yield
    BenchmarkFactory.clear_cache()


def _make_experiment(function=1, dimension=5):
    """Helper to create a test experiment."""
    return Experiment(
        benchmark="_CacheTestBenchmark",
        function=function,
        dimension=dimension,
        optimizer="GWO",
        population_size=10,
        max_function_evaluations=100,
        run=1,
        seed=1,
    )


class TestBenchmarkCaching:

    def test_cache_initially_empty(self):
        assert BenchmarkFactory.cache_size() == 0

    def test_create_caches_result(self):
        exp = _make_experiment()
        problem1 = BenchmarkFactory.create(exp)
        assert BenchmarkFactory.cache_size() == 1

        # Second call should return the same instance
        problem2 = BenchmarkFactory.create(exp)
        assert problem1 is problem2
        assert BenchmarkFactory.cache_size() == 1

    def test_different_function_different_cache(self):
        exp1 = _make_experiment(function=1)
        exp2 = _make_experiment(function=2)

        BenchmarkFactory.create(exp1)
        BenchmarkFactory.create(exp2)

        assert BenchmarkFactory.cache_size() == 2

    def test_different_dimension_different_cache(self):
        exp1 = _make_experiment(dimension=5)
        exp2 = _make_experiment(dimension=10)

        p1 = BenchmarkFactory.create(exp1)
        p2 = BenchmarkFactory.create(exp2)

        assert p1 is not p2
        assert BenchmarkFactory.cache_size() == 2

    def test_same_key_different_optimizer_shares_cache(self):
        """Different optimizers on the same benchmark should share the cache."""
        exp1 = Experiment(
            benchmark="_CacheTestBenchmark",
            function=1,
            dimension=5,
            optimizer="GWO",
            population_size=10,
            max_function_evaluations=100,
            run=1,
            seed=1,
        )
        exp2 = Experiment(
            benchmark="_CacheTestBenchmark",
            function=1,
            dimension=5,
            optimizer="BBGWO",
            population_size=50,
            max_function_evaluations=100,
            run=1,
            seed=1,
        )

        p1 = BenchmarkFactory.create(exp1)
        p2 = BenchmarkFactory.create(exp2)

        assert p1 is p2  # Same benchmark instance
        assert BenchmarkFactory.cache_size() == 1

    def test_clear_cache(self):
        exp = _make_experiment()
        BenchmarkFactory.create(exp)
        assert BenchmarkFactory.cache_size() == 1

        BenchmarkFactory.clear_cache()
        assert BenchmarkFactory.cache_size() == 0
