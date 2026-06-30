"""
test_config.py

Tests for configs/config.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    ROOT_DIR,
    RESULTS_DIR,
    LOG_DIR,
    BENCHMARKS,
    BENCHMARK_FUNCTIONS,
    DIMENSIONS,
    POPULATION_SIZES,
    RUNS,
    MAX_FE_MULTIPLIER,
    OPTIMIZERS,
)


class TestConfig:

    def test_root_dir_exists(self):
        assert ROOT_DIR.exists()

    def test_results_dir_is_path(self):
        assert isinstance(RESULTS_DIR, Path)

    def test_log_dir_is_path(self):
        assert isinstance(LOG_DIR, Path)

    def test_benchmarks_not_empty(self):
        assert len(BENCHMARKS) > 0

    def test_benchmarks_are_strings(self):
        for b in BENCHMARKS:
            assert isinstance(b, str)

    def test_benchmark_functions_match_benchmarks(self):
        for b in BENCHMARKS:
            assert b in BENCHMARK_FUNCTIONS
            assert isinstance(BENCHMARK_FUNCTIONS[b], int)
            assert BENCHMARK_FUNCTIONS[b] > 0

    def test_dimensions_sorted(self):
        assert DIMENSIONS == sorted(DIMENSIONS)

    def test_dimensions_positive(self):
        for d in DIMENSIONS:
            assert d > 0

    def test_population_sizes_sorted(self):
        assert POPULATION_SIZES == sorted(POPULATION_SIZES)

    def test_population_sizes_positive(self):
        for p in POPULATION_SIZES:
            assert p > 0

    def test_runs_positive(self):
        assert RUNS > 0

    def test_max_fe_multiplier_positive(self):
        assert MAX_FE_MULTIPLIER > 0

    def test_optimizers_not_empty(self):
        assert len(OPTIMIZERS) > 0

    def test_optimizers_are_strings(self):
        for o in OPTIMIZERS:
            assert isinstance(o, str)

    def test_expected_dimensions(self):
        expected = [10, 30, 50, 100, 200, 500, 1000]
        assert DIMENSIONS == expected

    def test_expected_population_sizes(self):
        expected = [3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 1500, 2000]
        assert POPULATION_SIZES == expected

    def test_runs_is_30(self):
        assert RUNS == 30

    def test_max_fe_multiplier_is_10000(self):
        assert MAX_FE_MULTIPLIER == 10000
