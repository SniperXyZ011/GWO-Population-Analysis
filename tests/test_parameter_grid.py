"""
test_parameter_grid.py

Tests for core/parameter_grid.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.parameter_grid import ParameterGrid
from core.experiment import Experiment


class TestParameterGrid:

    def test_generate_single(self):
        grid = ParameterGrid(
            benchmarks=["CEC2020"],
            optimizers=["GWO"],
            dimensions=[10],
            populations=[30],
            runs=1,
        )
        experiments = list(grid.generate())
        # CEC2020 has 10 functions
        assert len(experiments) == 10

    def test_generate_correct_type(self):
        grid = ParameterGrid(
            benchmarks=["CEC2020"],
            optimizers=["GWO"],
            dimensions=[10],
            populations=[30],
            runs=1,
        )
        for exp in grid.generate():
            assert isinstance(exp, Experiment)

    def test_len_matches_generation(self):
        grid = ParameterGrid(
            benchmarks=["CEC2020"],
            optimizers=["GWO"],
            dimensions=[10, 30],
            populations=[10, 50],
            runs=2,
        )
        expected = 10 * 1 * 2 * 2 * 2  # funcs * optim * dims * pops * runs
        assert len(grid) == expected
        assert len(list(grid.generate())) == expected

    def test_max_fe_calculation(self):
        grid = ParameterGrid(
            benchmarks=["CEC2020"],
            optimizers=["GWO"],
            dimensions=[10],
            populations=[30],
            runs=1,
        )
        for exp in grid.generate():
            assert exp.max_function_evaluations == 10 * 10000

    def test_seed_equals_run(self):
        grid = ParameterGrid(
            benchmarks=["CEC2020"],
            optimizers=["GWO"],
            dimensions=[10],
            populations=[30],
            runs=3,
        )
        for exp in grid.generate():
            assert exp.seed == exp.run

    def test_multiple_benchmarks(self):
        grid = ParameterGrid(
            benchmarks=["CEC2017", "CEC2020"],
            optimizers=["GWO"],
            dimensions=[10],
            populations=[30],
            runs=1,
        )
        # CEC2017: 29 funcs + CEC2020: 10 funcs = 39
        assert len(grid) == 39

    def test_unknown_benchmark_raises(self):
        grid = ParameterGrid(
            benchmarks=["CEC9999"],
            optimizers=["GWO"],
            dimensions=[10],
            populations=[30],
            runs=1,
        )
        import pytest
        with pytest.raises(ValueError, match="Unknown benchmark"):
            list(grid.generate())

    def test_repr(self):
        grid = ParameterGrid(
            benchmarks=["CEC2020"],
            optimizers=["GWO"],
            dimensions=[10],
            populations=[30],
            runs=1,
        )
        r = repr(grid)
        assert "ParameterGrid" in r
        assert "total=" in r
