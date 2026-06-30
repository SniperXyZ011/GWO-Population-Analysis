"""
test_experiment.py

Tests for core/experiment.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.experiment import Experiment


class TestExperiment:

    def test_creation(self):
        exp = Experiment(
            benchmark="CEC2020",
            function=1,
            dimension=10,
            optimizer="GWO",
            population_size=30,
            max_function_evaluations=100000,
            run=1,
            seed=1,
        )
        assert exp.benchmark == "CEC2020"
        assert exp.function == 1
        assert exp.dimension == 10
        assert exp.optimizer == "GWO"
        assert exp.population_size == 30
        assert exp.max_function_evaluations == 100000
        assert exp.run == 1
        assert exp.seed == 1

    def test_frozen(self):
        exp = Experiment(
            benchmark="CEC2020",
            function=1,
            dimension=10,
            optimizer="GWO",
            population_size=30,
            max_function_evaluations=100000,
            run=1,
            seed=1,
        )
        with pytest.raises(AttributeError):
            exp.dimension = 20

    def test_experiment_name(self):
        exp = Experiment(
            benchmark="CEC2020",
            function=5,
            dimension=30,
            optimizer="BBGWO",
            population_size=50,
            max_function_evaluations=300000,
            run=3,
            seed=3,
        )
        expected = "CEC2020_F5_BBGWO_D30_P50_FE300000_Run3"
        assert exp.experiment_name == expected

    def test_str(self):
        exp = Experiment(
            benchmark="CEC2017",
            function=1,
            dimension=10,
            optimizer="GWO",
            population_size=10,
            max_function_evaluations=100000,
            run=1,
            seed=1,
        )
        assert str(exp) == exp.experiment_name

    def test_hashable(self):
        exp1 = Experiment(
            benchmark="CEC2020",
            function=1,
            dimension=10,
            optimizer="GWO",
            population_size=30,
            max_function_evaluations=100000,
            run=1,
            seed=1,
        )
        exp2 = Experiment(
            benchmark="CEC2020",
            function=1,
            dimension=10,
            optimizer="GWO",
            population_size=30,
            max_function_evaluations=100000,
            run=1,
            seed=1,
        )
        assert exp1 == exp2
        assert hash(exp1) == hash(exp2)

    def test_different_experiments_not_equal(self):
        exp1 = Experiment(
            benchmark="CEC2020",
            function=1,
            dimension=10,
            optimizer="GWO",
            population_size=30,
            max_function_evaluations=100000,
            run=1,
            seed=1,
        )
        exp2 = Experiment(
            benchmark="CEC2020",
            function=2,
            dimension=10,
            optimizer="GWO",
            population_size=30,
            max_function_evaluations=100000,
            run=1,
            seed=1,
        )
        assert exp1 != exp2
