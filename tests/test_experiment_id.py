"""
test_experiment_id.py

Tests for Experiment.experiment_id property.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.experiment import Experiment


class TestExperimentId:

    @pytest.fixture
    def sample_experiment(self):
        return Experiment(
            benchmark="CEC2020",
            function=1,
            dimension=30,
            optimizer="GWO",
            population_size=50,
            max_function_evaluations=300000,
            run=5,
            seed=5,
        )

    def test_experiment_id_format(self, sample_experiment):
        """experiment_id should be colon-separated and compact."""
        eid = sample_experiment.experiment_id
        assert eid == "CEC2020:F1:GWO:D30:P50:R5"

    def test_experiment_id_is_string(self, sample_experiment):
        assert isinstance(sample_experiment.experiment_id, str)

    def test_experiment_name_unchanged(self, sample_experiment):
        """experiment_name should still work as before."""
        name = sample_experiment.experiment_name
        assert "CEC2020" in name
        assert "FE300000" in name
        assert "Run5" in name

    def test_experiment_id_differs_by_run(self):
        exp1 = Experiment(
            benchmark="CEC2020", function=1, dimension=10,
            optimizer="GWO", population_size=30,
            max_function_evaluations=100000, run=1, seed=1,
        )
        exp2 = Experiment(
            benchmark="CEC2020", function=1, dimension=10,
            optimizer="GWO", population_size=30,
            max_function_evaluations=100000, run=2, seed=2,
        )
        assert exp1.experiment_id != exp2.experiment_id

    def test_experiment_id_differs_by_optimizer(self):
        exp1 = Experiment(
            benchmark="CEC2020", function=1, dimension=10,
            optimizer="GWO", population_size=30,
            max_function_evaluations=100000, run=1, seed=1,
        )
        exp2 = Experiment(
            benchmark="CEC2020", function=1, dimension=10,
            optimizer="BBGWO", population_size=30,
            max_function_evaluations=100000, run=1, seed=1,
        )
        assert exp1.experiment_id != exp2.experiment_id

    def test_experiment_id_same_fe_same_id(self):
        """experiment_id should NOT include FE (it's derived from dimension)."""
        exp1 = Experiment(
            benchmark="CEC2020", function=1, dimension=10,
            optimizer="GWO", population_size=30,
            max_function_evaluations=100000, run=1, seed=1,
        )
        exp2 = Experiment(
            benchmark="CEC2020", function=1, dimension=10,
            optimizer="GWO", population_size=30,
            max_function_evaluations=200000, run=1, seed=1,
        )
        # Same experiment_id because only dimension matters, not FE
        assert exp1.experiment_id == exp2.experiment_id

    def test_str_uses_experiment_name(self, sample_experiment):
        """__str__ should return experiment_name, not experiment_id."""
        assert str(sample_experiment) == sample_experiment.experiment_name
