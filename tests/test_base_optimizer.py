"""
test_base_optimizer.py

Tests for optimizers/base_optimizer.py template method pattern.
"""

import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizers.base_optimizer import BaseOptimizer
from tests.conftest import MockProblem


class SimpleOptimizer(BaseOptimizer):
    """Minimal optimizer for testing the template method."""

    def initialize(self):
        self.population = self.initialize_population()

        # Evaluate initial population
        for i in range(self.population_size):
            if self.fe_count >= self.max_fe:
                break
            fitness = self.evaluate(self.population[i])
            self.update_best(self.population[i], fitness)

    def step(self):
        # Just evaluate population once per step
        for i in range(self.population_size):
            if self.fe_count >= self.max_fe:
                break
            pos = self.population[i] + self.rng.randn(self.dimension) * 0.1
            pos = self.clip(pos.reshape(1, -1)).flatten()
            fitness = self.evaluate(pos)
            if fitness < self.evaluate(self.population[i]):
                self.population[i] = pos
            self.update_best(pos, fitness)


class TestBaseOptimizer:

    def test_cannot_instantiate_abstract(self):
        problem = MockProblem(dimension=5)
        with pytest.raises(TypeError):
            BaseOptimizer(problem, 10, 1000, seed=1)

    def test_optimize_returns_dict(self):
        problem = MockProblem(dimension=5)
        opt = SimpleOptimizer(problem, 5, 100, seed=42)
        result = opt.optimize()
        assert isinstance(result, dict)

    def test_result_has_required_keys(self):
        problem = MockProblem(dimension=5)
        opt = SimpleOptimizer(problem, 5, 100, seed=42)
        result = opt.optimize()

        required_keys = [
            "optimizer",
            "best_score",
            "best_position",
            "function_evaluations",
            "iterations",
            "execution_time",
            "convergence_curve",
        ]

        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_fe_count_within_budget(self):
        problem = MockProblem(dimension=5)
        opt = SimpleOptimizer(problem, 5, 100, seed=42)
        result = opt.optimize()
        assert result["function_evaluations"] <= 100 + 5  # allow small overshoot from step

    def test_convergence_curve_not_empty(self):
        problem = MockProblem(dimension=5)
        opt = SimpleOptimizer(problem, 5, 100, seed=42)
        result = opt.optimize()
        assert len(result["convergence_curve"]) > 0

    def test_best_score_is_float(self):
        problem = MockProblem(dimension=5)
        opt = SimpleOptimizer(problem, 5, 100, seed=42)
        result = opt.optimize()
        assert isinstance(result["best_score"], float)

    def test_best_position_is_list(self):
        problem = MockProblem(dimension=5)
        opt = SimpleOptimizer(problem, 5, 100, seed=42)
        result = opt.optimize()
        assert isinstance(result["best_position"], list)
        assert len(result["best_position"]) == 5

    def test_execution_time_positive(self):
        problem = MockProblem(dimension=5)
        opt = SimpleOptimizer(problem, 5, 100, seed=42)
        result = opt.optimize()
        assert result["execution_time"] > 0

    def test_reproducibility(self):
        problem1 = MockProblem(dimension=5)
        opt1 = SimpleOptimizer(problem1, 5, 100, seed=42)
        result1 = opt1.optimize()

        problem2 = MockProblem(dimension=5)
        opt2 = SimpleOptimizer(problem2, 5, 100, seed=42)
        result2 = opt2.optimize()

        assert result1["best_score"] == result2["best_score"]

    def test_initialize_population_shape(self):
        problem = MockProblem(dimension=5)
        opt = SimpleOptimizer(problem, 10, 1000, seed=1)
        pop = opt.initialize_population()
        assert pop.shape == (10, 5)

    def test_initialize_population_within_bounds(self):
        problem = MockProblem(dimension=5)
        opt = SimpleOptimizer(problem, 10, 1000, seed=1)
        pop = opt.initialize_population()
        assert np.all(pop >= -100.0)
        assert np.all(pop <= 100.0)

    def test_clip(self):
        problem = MockProblem(dimension=5)
        opt = SimpleOptimizer(problem, 5, 100, seed=1)
        out_of_bounds = np.full((2, 5), 200.0)
        clipped = opt.clip(out_of_bounds)
        assert np.all(clipped <= 100.0)

    def test_repr(self):
        problem = MockProblem(dimension=5)
        opt = SimpleOptimizer(problem, 10, 1000, seed=1)
        r = repr(opt)
        assert "SimpleOptimizer" in r
        assert "Pop=10" in r

    def test_evaluate_beyond_budget_returns_inf(self):
        problem = MockProblem(dimension=5)
        opt = SimpleOptimizer(problem, 5, 5, seed=1)
        # Exhaust budget
        for _ in range(5):
            opt.evaluate(np.zeros(5))
        # Should return inf
        result = opt.evaluate(np.zeros(5))
        assert result == np.inf
