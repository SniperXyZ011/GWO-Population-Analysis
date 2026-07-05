"""
test_runner.py

Tests for core/runner.py.
Uses a mock benchmark registered for testing.
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runner import ExperimentRunner
from core.experiment import Experiment
from core.registry import optimizer_registry, benchmark_registry
from optimizers.base_optimizer import BaseOptimizer
from tests.conftest import MockProblem


# Register mock benchmark for testing
class _MockBenchmarkForRunner(MockProblem):
    """Mock benchmark that can be created by BenchmarkFactory."""
    pass


class _MockOptimizerForRunner(BaseOptimizer):
    def initialize(self):
        self.population = self.initialize_population()
        for i in range(self.population_size):
            if self.fe_count >= self.max_fe:
                break
            fitness = self.evaluate(self.population[i])
            self.update_best(self.population[i], fitness)

    def step(self):
        for i in range(self.population_size):
            if self.fe_count >= self.max_fe:
                break
            fitness = self.evaluate(self.population[i])
            self.update_best(self.population[i], fitness)


class TestRunner:

    @classmethod
    def setup_class(cls):
        if not benchmark_registry.exists("_MockBenchmarkForRunner"):
            benchmark_registry.register(_MockBenchmarkForRunner)
        if not optimizer_registry.exists("_MockOptimizerForRunner"):
            optimizer_registry.register(_MockOptimizerForRunner)

    def test_run_produces_result(self, tmp_path):
        runner = ExperimentRunner(results_dir=tmp_path)

        exp = Experiment(
            benchmark="_MockBenchmarkForRunner",
            function=1,
            dimension=5,
            optimizer="_MockOptimizerForRunner",
            population_size=5,
            max_function_evaluations=100,
            run=1,
            seed=1,
        )

        result = runner.run(exp)
        assert result is not None
        assert "best_score" in result

    def test_run_creates_files(self, tmp_path):
        runner = ExperimentRunner(results_dir=tmp_path)

        exp = Experiment(
            benchmark="_MockBenchmarkForRunner",
            function=1,
            dimension=5,
            optimizer="_MockOptimizerForRunner",
            population_size=5,
            max_function_evaluations=100,
            run=1,
            seed=1,
        )

        runner.run(exp)

        # Check result file exists
        result_file = (
            tmp_path
            / "_MockBenchmarkForRunner"
            / "_MockOptimizerForRunner"
            / "F1"
            / "D5"
            / "P5"
            / "run_1.json"
        )
        assert result_file.exists()

        # Check convergence file exists
        conv_file = (
            tmp_path
            / "_MockBenchmarkForRunner"
            / "_MockOptimizerForRunner"
            / "F1"
            / "D5"
            / "P5"
            / "convergence_run_1.csv"
        )
        assert conv_file.exists()
