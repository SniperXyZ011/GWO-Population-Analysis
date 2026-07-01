"""
test_resource_monitor.py

Tests for core/resource_monitor.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.resource_monitor import ResourceMonitor, SystemResources


class TestSystemResources:

    def test_detect_returns_system_resources(self):
        resources = ResourceMonitor.detect()
        assert isinstance(resources, SystemResources)

    def test_cpu_count_positive(self):
        resources = ResourceMonitor.detect()
        assert resources.cpu_count > 0

    def test_total_memory_non_negative(self):
        resources = ResourceMonitor.detect()
        assert resources.total_memory_gb >= 0.0

    def test_available_memory_non_negative(self):
        resources = ResourceMonitor.detect()
        assert resources.available_memory_gb >= 0.0

    def test_available_memory_lte_total(self):
        resources = ResourceMonitor.detect()
        if resources.total_memory_gb > 0:
            assert resources.available_memory_gb <= resources.total_memory_gb


class TestRecommendWorkers:

    def test_recommend_returns_positive_int(self):
        workers = ResourceMonitor.recommend_workers()
        assert isinstance(workers, int)
        assert workers >= 1

    def test_recommend_with_explicit_resources(self):
        resources = SystemResources(
            cpu_count=256,
            total_memory_gb=1024.0,
            available_memory_gb=900.0,
            load_average_1m=None,
        )
        workers = ResourceMonitor.recommend_workers(
            resources=resources,
            per_worker_memory_mb=150.0,
            max_utilization=0.5,
        )
        # CPU-based: 256 * 0.5 = 128
        # Memory-based: 900 * 1024 / 150 = 6144
        # Should be min(128, 6144) = 128
        assert workers == 128

    def test_recommend_respects_max_workers(self):
        resources = SystemResources(
            cpu_count=256,
            total_memory_gb=1024.0,
            available_memory_gb=900.0,
            load_average_1m=None,
        )
        workers = ResourceMonitor.recommend_workers(
            resources=resources,
            max_workers=10,
        )
        assert workers <= 10

    def test_recommend_respects_min_workers(self):
        resources = SystemResources(
            cpu_count=1,
            total_memory_gb=0.5,
            available_memory_gb=0.1,
            load_average_1m=None,
        )
        workers = ResourceMonitor.recommend_workers(
            resources=resources,
            min_workers=2,
        )
        assert workers >= 2

    def test_recommend_memory_limited(self):
        resources = SystemResources(
            cpu_count=256,
            total_memory_gb=2.0,
            available_memory_gb=1.0,
            load_average_1m=None,
        )
        workers = ResourceMonitor.recommend_workers(
            resources=resources,
            per_worker_memory_mb=500.0,
            max_utilization=1.0,
        )
        # Memory-based: 1.0 * 1024 / 500 = 2
        assert workers <= 2

    def test_recommend_load_aware(self):
        # System under heavy load
        resources_heavy = SystemResources(
            cpu_count=100,
            total_memory_gb=100.0,
            available_memory_gb=80.0,
            load_average_1m=80.0,  # 80% loaded
        )
        workers_heavy = ResourceMonitor.recommend_workers(
            resources=resources_heavy,
            max_utilization=0.5,
        )

        # System idle
        resources_idle = SystemResources(
            cpu_count=100,
            total_memory_gb=100.0,
            available_memory_gb=80.0,
            load_average_1m=0.0,
        )
        workers_idle = ResourceMonitor.recommend_workers(
            resources=resources_idle,
            max_utilization=0.5,
        )

        assert workers_heavy < workers_idle

    def test_recommend_with_none_resources(self):
        """Auto-detect when resources=None."""
        workers = ResourceMonitor.recommend_workers(resources=None)
        assert workers >= 1

    def test_recommend_zero_memory_fallback(self):
        """When psutil is missing, memory is (0, 0)."""
        resources = SystemResources(
            cpu_count=8,
            total_memory_gb=0.0,
            available_memory_gb=0.0,
            load_average_1m=None,
        )
        workers = ResourceMonitor.recommend_workers(
            resources=resources,
            max_utilization=0.5,
        )
        # Should fall back to CPU-based: 8 * 0.5 = 4
        assert workers == 4
