"""
test_base_problem.py

Tests for benchmarks/base_problem.py interface.
"""

import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.base_problem import BaseProblem
from tests.conftest import MockProblem


class TestBaseProblem:

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseProblem(dimension=10)

    def test_mock_problem_evaluate(self, mock_problem):
        x = np.zeros(10)
        assert mock_problem.evaluate(x) == 0.0

    def test_mock_problem_evaluate_nonzero(self, mock_problem):
        x = np.ones(10)
        assert mock_problem.evaluate(x) == 10.0

    def test_dimension(self, mock_problem):
        assert mock_problem.dimension == 10

    def test_lower_bound(self, mock_problem):
        lb = mock_problem.lower_bound()
        assert len(lb) == 10
        assert all(v == -100.0 for v in lb)

    def test_upper_bound(self, mock_problem):
        ub = mock_problem.upper_bound()
        assert len(ub) == 10
        assert all(v == 100.0 for v in ub)

    def test_optimum(self, mock_problem):
        assert mock_problem.optimum() == 0.0

    def test_function_name(self, mock_problem):
        assert mock_problem.function_name() == "F1"

    def test_benchmark_name(self, mock_problem):
        assert mock_problem.benchmark_name() == "MockBenchmark"

    def test_bounds(self, mock_problem):
        lb, ub = mock_problem.bounds()
        assert len(lb) == 10
        assert len(ub) == 10

    def test_clip(self, mock_problem):
        x = np.full((2, 10), 200.0)
        clipped = mock_problem.clip(x)
        assert np.all(clipped <= 100.0)

    def test_repr(self, mock_problem):
        r = repr(mock_problem)
        assert "MockBenchmark" in r
        assert "F1" in r
        assert "D=10" in r
