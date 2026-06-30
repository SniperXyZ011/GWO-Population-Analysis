"""
test_factories.py

Tests for core/optimizer_factory.py and core/benchmark_factory.py.
Uses a mock benchmark registered for testing.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.registry import Registry, optimizer_registry, benchmark_registry
from core.optimizer_factory import OptimizerFactory
from core.benchmark_factory import BenchmarkFactory
from core.experiment import Experiment
from tests.conftest import MockProblem
from optimizers.base_optimizer import BaseOptimizer


# A minimal test optimizer
class _TestOptimizer(BaseOptimizer):

    def initialize(self):
        self.population = self.initialize_population()
        self.best_score = 0.0
        self.best_position = self.population[0].copy()

    def step(self):
        pass


class TestOptimizerFactory:

    def setup_method(self):
        """Register a test optimizer if not already registered."""
        if not optimizer_registry.exists("_TestOptimizer"):
            optimizer_registry.register(_TestOptimizer)

    def test_create_optimizer(self):
        problem = MockProblem(dimension=5)
        exp = Experiment(
            benchmark="MockBenchmark",
            function=1,
            dimension=5,
            optimizer="_TestOptimizer",
            population_size=10,
            max_function_evaluations=1000,
            run=1,
            seed=1,
        )
        opt = OptimizerFactory.create(exp, problem)
        assert isinstance(opt, BaseOptimizer)
        assert opt.population_size == 10
        assert opt.max_fe == 1000

    def test_create_unknown_optimizer_raises(self):
        problem = MockProblem(dimension=5)
        exp = Experiment(
            benchmark="MockBenchmark",
            function=1,
            dimension=5,
            optimizer="NonexistentOptimizer",
            population_size=10,
            max_function_evaluations=1000,
            run=1,
            seed=1,
        )
        with pytest.raises(KeyError):
            OptimizerFactory.create(exp, problem)


class TestBenchmarkFactory:

    def setup_method(self):
        """Register MockProblem as a benchmark if not registered."""
        # MockProblem needs a factory-compatible interface
        pass

    def test_create_unknown_benchmark_raises(self):
        exp = Experiment(
            benchmark="NonexistentBenchmark",
            function=1,
            dimension=5,
            optimizer="GWO",
            population_size=10,
            max_function_evaluations=1000,
            run=1,
            seed=1,
        )
        with pytest.raises(KeyError):
            BenchmarkFactory.create(exp)
