"""
test_optimizers.py

Tests for all 11 optimizer implementations.
Verifies each optimizer can run and produce valid results
on a mock sphere problem.
"""

import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.conftest import MockProblem

# Import all optimizers to register them
from optimizers.gwo import GWO
from optimizers.bbgwo import BBGWO
from optimizers.mengwo import MENGWO
from optimizers.mgwo import MGWO
from optimizers.rwgwo import RWGWO
from optimizers.obgwo import OBGWO
from optimizers.modgwo import modGWO
from optimizers.ebgwo import EBGWO
from optimizers.igwo_ms import IGWO_MS
from optimizers.agwo import AGWO
from optimizers.iagwo import IAGWO
from optimizers.igwo_dlh import IGWO_DLH

ALL_OPTIMIZERS = [
    GWO, BBGWO, MENGWO, MGWO,
    RWGWO, OBGWO, modGWO, EBGWO, IGWO_MS, AGWO, IAGWO, IGWO_DLH
]


class TestAllOptimizers:

    @pytest.mark.parametrize(
        "optimizer_cls",
        ALL_OPTIMIZERS,
        ids=[cls.__name__ for cls in ALL_OPTIMIZERS],
    )
    def test_optimizer_runs(self, optimizer_cls):
        """Each optimizer should complete optimization."""
        problem = MockProblem(dimension=5)
        opt = optimizer_cls(
            problem=problem,
            population_size=10,
            max_function_evaluations=500,
            seed=42,
        )
        result = opt.optimize()
        assert isinstance(result, dict)

    @pytest.mark.parametrize(
        "optimizer_cls",
        ALL_OPTIMIZERS,
        ids=[cls.__name__ for cls in ALL_OPTIMIZERS],
    )
    def test_optimizer_result_keys(self, optimizer_cls):
        """Each optimizer should return all required keys."""
        problem = MockProblem(dimension=5)
        opt = optimizer_cls(
            problem=problem,
            population_size=10,
            max_function_evaluations=500,
            seed=42,
        )
        result = opt.optimize()

        required_keys = [
            "optimizer", "best_score", "best_position",
            "function_evaluations", "iterations",
            "execution_time", "convergence_curve",
        ]
        for key in required_keys:
            assert key in result, f"{optimizer_cls.__name__} missing '{key}'"

    @pytest.mark.parametrize(
        "optimizer_cls",
        ALL_OPTIMIZERS,
        ids=[cls.__name__ for cls in ALL_OPTIMIZERS],
    )
    def test_optimizer_finite_score(self, optimizer_cls):
        """Each optimizer should produce a finite best score."""
        problem = MockProblem(dimension=5)
        opt = optimizer_cls(
            problem=problem,
            population_size=10,
            max_function_evaluations=500,
            seed=42,
        )
        result = opt.optimize()
        assert np.isfinite(result["best_score"]), (
            f"{optimizer_cls.__name__} produced non-finite score"
        )

    @pytest.mark.parametrize(
        "optimizer_cls",
        ALL_OPTIMIZERS,
        ids=[cls.__name__ for cls in ALL_OPTIMIZERS],
    )
    def test_optimizer_convergence_not_empty(self, optimizer_cls):
        """Each optimizer should have a non-empty convergence curve."""
        problem = MockProblem(dimension=5)
        opt = optimizer_cls(
            problem=problem,
            population_size=10,
            max_function_evaluations=500,
            seed=42,
        )
        result = opt.optimize()
        assert len(result["convergence_curve"]) > 0

    @pytest.mark.parametrize(
        "optimizer_cls",
        ALL_OPTIMIZERS,
        ids=[cls.__name__ for cls in ALL_OPTIMIZERS],
    )
    def test_optimizer_improves_from_random(self, optimizer_cls):
        """Each optimizer should improve from a random initialization."""
        problem = MockProblem(dimension=5)
        opt = optimizer_cls(
            problem=problem,
            population_size=10,
            max_function_evaluations=1000,
            seed=42,
        )
        result = opt.optimize()

        # A sphere function with bounds [-100, 100] has random init
        # around sum(x^2) ~ 5 * (100^2/3) = ~166667
        # Optimizer should improve significantly
        assert result["best_score"] < 50000, (
            f"{optimizer_cls.__name__} did not improve sufficiently"
        )

    @pytest.mark.parametrize(
        "optimizer_cls",
        ALL_OPTIMIZERS,
        ids=[cls.__name__ for cls in ALL_OPTIMIZERS],
    )
    def test_optimizer_reproducibility(self, optimizer_cls):
        """Same seed should produce same result."""
        problem1 = MockProblem(dimension=5)
        opt1 = optimizer_cls(
            problem=problem1,
            population_size=10,
            max_function_evaluations=200,
            seed=7,
        )
        result1 = opt1.optimize()

        problem2 = MockProblem(dimension=5)
        opt2 = optimizer_cls(
            problem=problem2,
            population_size=10,
            max_function_evaluations=200,
            seed=7,
        )
        result2 = opt2.optimize()

        assert result1["best_score"] == pytest.approx(
            result2["best_score"], rel=1e-10
        ), f"{optimizer_cls.__name__} is not reproducible"

    @pytest.mark.parametrize(
        "optimizer_cls",
        ALL_OPTIMIZERS,
        ids=[cls.__name__ for cls in ALL_OPTIMIZERS],
    )
    def test_optimizer_name_correct(self, optimizer_cls):
        """Optimizer name in result should match class name."""
        problem = MockProblem(dimension=5)
        opt = optimizer_cls(
            problem=problem,
            population_size=10,
            max_function_evaluations=200,
            seed=42,
        )
        result = opt.optimize()
        assert result["optimizer"] == optimizer_cls.__name__

    @pytest.mark.parametrize(
        "optimizer_cls",
        ALL_OPTIMIZERS,
        ids=[cls.__name__ for cls in ALL_OPTIMIZERS],
    )
    def test_optimizer_min_population(self, optimizer_cls):
        """Should work with very small population (3)."""
        problem = MockProblem(dimension=3)
        # MENGWO needs at least 4 for NPSR, others need at least 3
        pop = 5 if optimizer_cls.__name__ == "MENGWO" else 3
        opt = optimizer_cls(
            problem=problem,
            population_size=pop,
            max_function_evaluations=200,
            seed=42,
        )
        result = opt.optimize()
        assert np.isfinite(result["best_score"])
