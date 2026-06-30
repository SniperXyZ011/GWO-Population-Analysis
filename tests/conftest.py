"""
conftest.py

Shared fixtures for tests.
Provides a lightweight mock benchmark for fast testing
without opfunu dependency.
"""

import sys
from pathlib import Path

import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.base_problem import BaseProblem


class MockProblem(BaseProblem):
    """
    Lightweight sphere function for testing.
    f(x) = sum(x_i^2), optimum at 0.
    """

    def __init__(self, dimension: int = 10, function: int = 1):
        super().__init__(dimension)
        self.function = function
        self._lb = np.full(dimension, -100.0)
        self._ub = np.full(dimension, 100.0)

    def evaluate(self, solution: np.ndarray) -> float:
        return float(np.sum(solution ** 2))

    def lower_bound(self):
        return self._lb

    def upper_bound(self):
        return self._ub

    def optimum(self):
        return 0.0

    def function_name(self):
        return f"F{self.function}"

    def benchmark_name(self):
        return "MockBenchmark"


@pytest.fixture
def mock_problem():
    """Create a 10-dimensional mock problem."""
    return MockProblem(dimension=10)


@pytest.fixture
def mock_problem_small():
    """Create a 2-dimensional mock problem for fast tests."""
    return MockProblem(dimension=2)
