"""
test_checkpoint.py

Tests for core/checkpoint.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.checkpoint import CheckpointDB
from core.experiment import Experiment
from core.environment import EnvironmentInfo


@pytest.fixture
def db(tmp_path):
    """Create a fresh checkpoint DB in a temp directory."""
    return CheckpointDB(tmp_path / "test_checkpoint.db")


@pytest.fixture
def sample_experiment():
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


class TestCheckpointDB:

    def test_create_db(self, db):
        assert db.db_path.exists()

    def test_initially_empty(self, db):
        assert db.get_completed_count() == 0
        assert db.get_failed_count() == 0

    def test_is_completed_false_initially(self, db, sample_experiment):
        assert db.is_completed(sample_experiment) is False

    def test_record_result(self, db, sample_experiment, sample_result):
        db.record_result(sample_experiment, sample_result)
        assert db.is_completed(sample_experiment) is True

    def test_completed_count_increments(self, db, sample_experiment, sample_result):
        db.record_result(sample_experiment, sample_result)
        assert db.get_completed_count() == 1

    def test_get_completed_ids(self, db, sample_experiment, sample_result):
        db.record_result(sample_experiment, sample_result)
        ids = db.get_completed_ids()
        assert sample_experiment.experiment_name in ids

    def test_get_completed_ids_bulk(self, db, sample_result):
        """Test bulk completion check with multiple experiments."""
        for run in range(1, 6):
            exp = Experiment(
                benchmark="CEC2020",
                function=1,
                dimension=10,
                optimizer="GWO",
                population_size=30,
                max_function_evaluations=100000,
                run=run,
                seed=run,
            )
            db.record_result(exp, sample_result)

        ids = db.get_completed_ids()
        assert len(ids) == 5

    def test_record_failure(self, db, sample_experiment):
        db.record_failure(
            sample_experiment,
            error_message="Test error",
            error_traceback="Traceback...",
        )
        assert db.get_failed_count() == 1

    def test_retry_count_increments(self, db, sample_experiment):
        assert db.get_retry_count(sample_experiment) == 0

        db.record_failure(sample_experiment, "Error 1")
        assert db.get_retry_count(sample_experiment) == 0  # First failure = retry 0

        db.record_failure(sample_experiment, "Error 2")
        assert db.get_retry_count(sample_experiment) == 1  # Second = retry 1

    def test_store_and_get_metadata(self, db):
        db.store_metadata("test_key", "test_value")
        assert db.get_metadata("test_key") == "test_value"

    def test_get_metadata_nonexistent(self, db):
        assert db.get_metadata("nonexistent") is None

    def test_store_environment(self, db):
        env = EnvironmentInfo.capture()
        db.store_environment(env)

        raw = db.get_metadata("environment")
        assert raw is not None
        assert "hostname" in raw

    def test_query_results_empty(self, db):
        results = db.query_results(benchmark="CEC2020")
        assert results == []

    def test_query_results_with_data(self, db, sample_experiment, sample_result):
        db.record_result(sample_experiment, sample_result)

        results = db.query_results(benchmark="CEC2020")
        assert len(results) == 1
        assert results[0]["best_score"] == 42.5

    def test_query_results_filter_optimizer(self, db, sample_result):
        for opt in ["GWO", "BBGWO"]:
            exp = Experiment(
                benchmark="CEC2020",
                function=1,
                dimension=10,
                optimizer=opt,
                population_size=30,
                max_function_evaluations=100000,
                run=1,
                seed=1,
            )
            db.record_result(exp, sample_result)

        gwo_results = db.query_results(optimizer="GWO")
        assert len(gwo_results) == 1

        all_results = db.query_results(benchmark="CEC2020")
        assert len(all_results) == 2

    def test_get_campaign_stats(self, db, sample_experiment, sample_result):
        db.record_result(sample_experiment, sample_result)
        stats = db.get_campaign_stats()

        assert stats["completed"] == 1
        assert stats["avg_time"] > 0
        assert "per_optimizer" in stats
        assert "GWO" in stats["per_optimizer"]

    def test_convergence_stored(self, db, sample_experiment, sample_result):
        """Verify convergence data is stored in the DB."""
        db.record_result(sample_experiment, sample_result)

        import sqlite3
        conn = sqlite3.connect(str(db.db_path))
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            "SELECT * FROM convergence WHERE result_id = 1"
        ).fetchall()
        conn.close()

        assert len(rows) > 0

    def test_replace_on_duplicate(self, db, sample_experiment, sample_result):
        """Recording the same experiment twice should replace, not error."""
        db.record_result(sample_experiment, sample_result)

        updated_result = dict(sample_result)
        updated_result["best_score"] = 10.0
        db.record_result(sample_experiment, updated_result)

        assert db.get_completed_count() == 1

        results = db.query_results(benchmark="CEC2020")
        assert results[0]["best_score"] == 10.0

    def test_repr(self, db):
        r = repr(db)
        assert "CheckpointDB" in r
        assert "completed=0" in r
