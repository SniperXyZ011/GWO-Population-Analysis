"""
test_scheduler.py

Tests for core/scheduler.py.

Uses mock optimizer and benchmark to test scheduling
without requiring opfunu or real computation.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.checkpoint import CheckpointDB
from core.experiment import Experiment
from core.scheduler import ExperimentScheduler, _run_single_experiment
from core.registry import optimizer_registry, benchmark_registry
from optimizers.base_optimizer import BaseOptimizer
from tests.conftest import MockProblem


# =============================================================
# Mock optimizer and benchmark for scheduler tests
# =============================================================

class _SchedulerTestOptimizer(BaseOptimizer):
    """Minimal optimizer for scheduler testing."""

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


class _SchedulerTestBenchmark(MockProblem):
    """Mock benchmark that can be created by BenchmarkFactory."""
    pass


# Register mocks once
if not optimizer_registry.exists("_SchedulerTestOptimizer"):
    optimizer_registry.register(_SchedulerTestOptimizer)

if not benchmark_registry.exists("_SchedulerTestBenchmark"):
    benchmark_registry.register(_SchedulerTestBenchmark)


def _make_experiments(count: int = 5):
    """Generate a list of test experiments."""
    return [
        Experiment(
            benchmark="_SchedulerTestBenchmark",
            function=1,
            dimension=5,
            optimizer="_SchedulerTestOptimizer",
            population_size=5,
            max_function_evaluations=50,
            run=i,
            seed=i,
        )
        for i in range(1, count + 1)
    ]


# =============================================================
# Tests
# =============================================================

class TestRunSingleExperiment:

    def test_returns_dict(self, tmp_path):
        exp = _make_experiments(1)[0]
        outcome = _run_single_experiment(
            exp, str(tmp_path), False
        )
        assert isinstance(outcome, dict)
        assert "status" in outcome

    def test_completed_status(self, tmp_path):
        exp = _make_experiments(1)[0]
        outcome = _run_single_experiment(
            exp, str(tmp_path), False
        )
        assert outcome["status"] == "completed"
        assert outcome["result"] is not None

    def test_result_has_best_score(self, tmp_path):
        exp = _make_experiments(1)[0]
        outcome = _run_single_experiment(
            exp, str(tmp_path), False
        )
        assert "best_score" in outcome["result"]


class TestExperimentScheduler:

    def test_scheduler_creation(self, tmp_path):
        db = CheckpointDB(tmp_path / "test.db")
        scheduler = ExperimentScheduler(
            checkpoint_db=db,
            results_dir=tmp_path / "results",
            max_workers=1,
        )
        assert scheduler.max_workers == 1

    def test_run_campaign_sequential(self, tmp_path):
        """Run a small campaign in sequential mode (workers=1)."""
        db = CheckpointDB(tmp_path / "test.db")
        scheduler = ExperimentScheduler(
            checkpoint_db=db,
            results_dir=tmp_path / "results",
            max_workers=1,
        )

        experiments = _make_experiments(3)
        summary = scheduler.run_campaign(experiments)

        assert summary["total"] == 3
        assert summary["completed"] == 3
        assert summary["failed"] == 0

    def test_results_in_checkpoint_db(self, tmp_path):
        """Verify results are stored in the checkpoint DB."""
        db = CheckpointDB(tmp_path / "test.db")
        scheduler = ExperimentScheduler(
            checkpoint_db=db,
            results_dir=tmp_path / "results",
            max_workers=1,
        )

        experiments = _make_experiments(3)
        scheduler.run_campaign(experiments)

        assert db.get_completed_count() == 3

    def test_skip_completed_experiments(self, tmp_path):
        """Running the same campaign twice should skip all on second run."""
        db = CheckpointDB(tmp_path / "test.db")
        scheduler = ExperimentScheduler(
            checkpoint_db=db,
            results_dir=tmp_path / "results",
            max_workers=1,
        )

        experiments = _make_experiments(3)

        # First run
        summary1 = scheduler.run_campaign(experiments)
        assert summary1["completed"] == 3
        assert summary1["skipped"] == 0

        # Second run — same experiments
        summary2 = scheduler.run_campaign(experiments)
        assert summary2["completed"] == 0
        assert summary2["skipped"] == 3

    def test_overwrite_reruns(self, tmp_path):
        """With overwrite=True, completed experiments should be re-run."""
        db = CheckpointDB(tmp_path / "test.db")
        scheduler = ExperimentScheduler(
            checkpoint_db=db,
            results_dir=tmp_path / "results",
            max_workers=1,
            overwrite=True,
        )

        experiments = _make_experiments(2)

        summary1 = scheduler.run_campaign(experiments)
        assert summary1["completed"] == 2

        summary2 = scheduler.run_campaign(experiments)
        assert summary2["completed"] == 2  # re-run, not skipped

    def test_progress_callback(self, tmp_path):
        """Progress callback should be called for each experiment."""
        db = CheckpointDB(tmp_path / "test.db")

        progress_calls = []

        def callback(done, total, outcome):
            progress_calls.append((done, total))

        scheduler = ExperimentScheduler(
            checkpoint_db=db,
            results_dir=tmp_path / "results",
            max_workers=1,
            progress_callback=callback,
        )

        experiments = _make_experiments(3)
        scheduler.run_campaign(experiments)

        assert len(progress_calls) == 3

    def test_environment_stored(self, tmp_path):
        """Environment metadata should be stored in the checkpoint DB."""
        db = CheckpointDB(tmp_path / "test.db")
        scheduler = ExperimentScheduler(
            checkpoint_db=db,
            results_dir=tmp_path / "results",
            max_workers=1,
        )

        scheduler.run_campaign(_make_experiments(1))

        env_json = db.get_metadata("environment")
        assert env_json is not None
        assert "hostname" in env_json

    def test_campaign_with_no_experiments(self, tmp_path):
        """Empty experiment list should complete immediately."""
        db = CheckpointDB(tmp_path / "test.db")
        scheduler = ExperimentScheduler(
            checkpoint_db=db,
            results_dir=tmp_path / "results",
            max_workers=1,
        )

        summary = scheduler.run_campaign([])
        assert summary["total"] == 0
        assert summary["completed"] == 0

    def test_repr(self, tmp_path):
        db = CheckpointDB(tmp_path / "test.db")
        scheduler = ExperimentScheduler(
            checkpoint_db=db,
            results_dir=tmp_path / "results",
            max_workers=4,
        )
        r = repr(scheduler)
        assert "ExperimentScheduler" in r
        assert "workers=4" in r
