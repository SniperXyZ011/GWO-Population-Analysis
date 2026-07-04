"""
test_result_manager.py

Tests for core/result_manager.py.
"""

import sys
import json
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.result_manager import ResultManager
from core.experiment import Experiment


@pytest.fixture
def experiment():
    return Experiment(
        benchmark="CEC2020",
        function=1,
        dimension=10,
        optimizer="GWO",
        population_size=30,
        max_function_evaluations=100000,
        run=1,
        seed=1,
    )


@pytest.fixture
def sample_result():
    return {
        "optimizer": "GWO",
        "best_score": 42.5,
        "best_position": [1.0, 2.0, 3.0],
        "function_evaluations": 100000,
        "iterations": 3333,
        "execution_time": 5.2,
        "convergence_curve": [100.0, 80.0, 60.0, 42.5],
    }


class TestResultManager:

    def test_create_experiment_folder(self, tmp_path, experiment):
        rm = ResultManager(tmp_path)
        folder = rm.create_experiment_folder(experiment)
        assert folder.exists()
        assert "CEC2020" in str(folder)
        assert "GWO" in str(folder)
        assert "F1" in str(folder)
        assert "D10" in str(folder)
        assert "P30" in str(folder)

    def test_save_result(self, tmp_path, experiment, sample_result):
        rm = ResultManager(tmp_path)
        rm.save_result(experiment, sample_result)

        result_file = (
            tmp_path / "CEC2020" / "GWO" / "F1"
            / "D10" / "P30" / "runs.jsonl"
        )
        assert result_file.exists()

        with open(result_file) as f:
            lines = f.readlines()
        assert len(lines) == 1
        loaded = json.loads(lines[0])
        assert loaded["best_score"] == 42.5
        assert loaded["_experiment_id"] == experiment.experiment_name

    def test_save_convergence(self, tmp_path, experiment):
        import os
        rm = ResultManager(tmp_path)
        curve = [100.0, 80.0, 60.0, 42.5]
        rm.save_convergence(experiment, curve)

        conv_file = tmp_path / f"convergence_worker_{os.getpid()}.csv"
        assert conv_file.exists()

    def test_save_summary(self, tmp_path, experiment):
        rm = ResultManager(tmp_path)
        summary = {"mean": 50.0, "std": 10.0}
        rm.save_summary(experiment, summary)

        summary_file = (
            tmp_path / "CEC2020" / "GWO" / "F1"
            / "D10" / "P30" / "summary.json"
        )
        assert summary_file.exists()

    def test_result_exists_true(self, tmp_path, experiment, sample_result):
        rm = ResultManager(tmp_path)
        assert not rm.result_exists(experiment)

    def test_result_exists_false(self, tmp_path, experiment):
        rm = ResultManager(tmp_path)
        assert not rm.result_exists(experiment)

    def test_load_result(self, tmp_path, experiment, sample_result):
        rm = ResultManager(tmp_path)
        assert rm.load_result(experiment) is None

    def test_load_nonexistent_result(self, tmp_path, experiment):
        rm = ResultManager(tmp_path)
        loaded = rm.load_result(experiment)
        assert loaded is None

    def test_numpy_serialization(self, tmp_path, experiment):
        rm = ResultManager(tmp_path)
        result = {
            "best_score": np.float64(42.5),
            "best_position": np.array([1.0, 2.0, 3.0]),
            "fe": np.int64(100),
        }
        rm.save_result(experiment, result)

        result_file = (
            tmp_path / "CEC2020" / "GWO" / "F1"
            / "D10" / "P30" / "runs.jsonl"
        )
        with open(result_file) as f:
            lines = f.readlines()
        loaded = json.loads(lines[0])
        
        assert loaded["best_score"] == 42.5
        assert isinstance(loaded["best_position"], list)
