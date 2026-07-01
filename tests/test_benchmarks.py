"""
test_benchmarks.py

Tests for benchmark wrappers (CEC2017, CEC2020, CEC2022).
Requires opfunu to be installed.
"""

import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# These tests require opfunu
opfunu = pytest.importorskip("opfunu", reason="opfunu required")

from benchmarks.cec2017.benchmark import CEC2017
from benchmarks.cec2020.benchmark import CEC2020
from benchmarks.cec2022.benchmark import CEC2022


class TestCEC2017:

    def test_create_f1(self):
        problem = CEC2017(function=1, dimension=10)
        assert problem.dimension == 10
        assert problem.function_name() == "F1"
        assert problem.benchmark_name() == "CEC2017"

    def test_evaluate_returns_float(self):
        problem = CEC2017(function=1, dimension=10)
        x = np.zeros(10)
        result = problem.evaluate(x)
        assert isinstance(result, (float, np.floating))

    def test_bounds_correct_shape(self):
        problem = CEC2017(function=1, dimension=10)
        lb = problem.lower_bound()
        ub = problem.upper_bound()
        assert len(lb) == 10
        assert len(ub) == 10

    def test_invalid_function_raises(self):
        with pytest.raises(ValueError, match="not available"):
            CEC2017(function=99, dimension=10)

    def test_all_functions_exist(self):
        for f in range(1, 30):
            problem = CEC2017(function=f, dimension=10)
            assert problem is not None


class TestCEC2020:

    def test_create_f1(self):
        problem = CEC2020(function=1, dimension=10)
        assert problem.dimension == 10
        assert problem.function_name() == "F1"
        assert problem.benchmark_name() == "CEC2020"

    def test_evaluate_returns_float(self):
        problem = CEC2020(function=1, dimension=10)
        x = np.zeros(10)
        result = problem.evaluate(x)
        assert isinstance(result, (float, np.floating))

    def test_bounds_correct_shape(self):
        problem = CEC2020(function=1, dimension=10)
        lb = problem.lower_bound()
        ub = problem.upper_bound()
        assert len(lb) == 10
        assert len(ub) == 10

    def test_invalid_function_raises(self):
        with pytest.raises(ValueError, match="not available"):
            CEC2020(function=99, dimension=10)

    def test_all_functions_exist(self):
        for f in range(1, 11):
            problem = CEC2020(function=f, dimension=10)
            assert problem is not None


class TestCEC2022:

    def test_create_f1(self):
        problem = CEC2022(function=1, dimension=10)
        assert problem.dimension == 10
        assert problem.function_name() == "F1"
        assert problem.benchmark_name() == "CEC2022"

    def test_evaluate_returns_float(self):
        problem = CEC2022(function=1, dimension=10)
        x = np.zeros(10)
        result = problem.evaluate(x)
        assert isinstance(result, (float, np.floating))

    def test_bounds_correct_shape(self):
        problem = CEC2022(function=1, dimension=10)
        lb = problem.lower_bound()
        ub = problem.upper_bound()
        assert len(lb) == 10
        assert len(ub) == 10

    def test_invalid_function_raises(self):
        with pytest.raises(ValueError, match="not available"):
            CEC2022(function=99, dimension=10)

    def test_all_functions_exist(self):
        for f in range(1, 13):
            problem = CEC2022(function=f, dimension=10)
            assert problem is not None
