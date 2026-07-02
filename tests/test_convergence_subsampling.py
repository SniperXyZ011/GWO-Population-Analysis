"""
test_convergence_subsampling.py

Tests for convergence curve subsampling in BaseOptimizer.
"""

import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizers.base_optimizer import BaseOptimizer


class _SubsamplingTestOptimizer(BaseOptimizer):
    """Minimal optimizer for testing convergence subsampling."""

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


class _MockProblemForSubsampling:
    """Minimal mock problem."""

    def __init__(self, dimension=10):
        self.dimension = dimension

    def evaluate(self, x):
        return np.sum(x ** 2)

    def lower_bound(self):
        return np.full(self.dimension, -100.0)

    def upper_bound(self):
        return np.full(self.dimension, 100.0)


class TestConvergenceSubsampling:

    def test_small_run_records_all(self):
        """For small runs, convergence curve should have reasonable size."""
        problem = _MockProblemForSubsampling(dimension=5)
        optimizer = _SubsamplingTestOptimizer(
            problem=problem,
            population_size=5,
            max_function_evaluations=50,
            seed=42,
        )

        result = optimizer.optimize()
        curve = result["convergence_curve"]

        # Should have recorded data
        assert len(curve) > 0
        # For 50 FE, 5 pop -> ~10 iterations, all recorded
        assert len(curve) <= 15

    def test_large_run_subsampled(self):
        """For large runs, convergence curve should be capped near 1000."""
        problem = _MockProblemForSubsampling(dimension=5)
        optimizer = _SubsamplingTestOptimizer(
            problem=problem,
            population_size=3,
            max_function_evaluations=30000,
            seed=42,
        )

        result = optimizer.optimize()
        curve = result["convergence_curve"]

        # 30000 / 3 = 10000 iterations, but subsampled to ~1000
        assert len(curve) <= 1100
        assert len(curve) >= 500  # Should still have substantial data

    def test_convergence_monotonic(self):
        """Convergence curve should be non-increasing (minimization)."""
        problem = _MockProblemForSubsampling(dimension=5)
        optimizer = _SubsamplingTestOptimizer(
            problem=problem,
            population_size=5,
            max_function_evaluations=500,
            seed=42,
        )

        result = optimizer.optimize()
        curve = result["convergence_curve"]

        for i in range(1, len(curve)):
            assert curve[i] <= curve[i - 1], (
                f"Convergence not monotonic at index {i}: "
                f"{curve[i]} > {curve[i - 1]}"
            )

    def test_fe_per_second_in_results(self):
        """Result dict should contain fe_per_second metric."""
        problem = _MockProblemForSubsampling(dimension=5)
        optimizer = _SubsamplingTestOptimizer(
            problem=problem,
            population_size=5,
            max_function_evaluations=50,
            seed=42,
        )

        result = optimizer.optimize()

        assert "fe_per_second" in result
        assert result["fe_per_second"] > 0

    def test_fe_per_second_reasonable(self):
        """FE/sec should be positive and finite."""
        problem = _MockProblemForSubsampling(dimension=5)
        optimizer = _SubsamplingTestOptimizer(
            problem=problem,
            population_size=5,
            max_function_evaluations=500,
            seed=42,
        )

        result = optimizer.optimize()

        assert np.isfinite(result["fe_per_second"])
        assert result["fe_per_second"] > 0
