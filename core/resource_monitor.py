"""
resource_monitor.py

System resource detection and worker recommendation.

Ensures the framework respects shared HPC environments
by auto-detecting CPU count and available memory, then
recommending a safe number of parallel workers.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SystemResources:
    """Snapshot of available system resources."""

    cpu_count: int
    total_memory_gb: float
    available_memory_gb: float
    load_average_1m: Optional[float]  # None on Windows


class ResourceMonitor:
    """
    Detect system resources and recommend worker count.

    Usage
    -----
    >>> resources = ResourceMonitor.detect()
    >>> workers = ResourceMonitor.recommend_workers(resources)
    """

    @staticmethod
    def detect() -> SystemResources:
        """
        Detect current system resources.

        Returns
        -------
        SystemResources
        """

        cpu_count = ResourceMonitor._get_cpu_count()
        total_mem, avail_mem = ResourceMonitor._get_memory()
        load_avg = ResourceMonitor._get_load_average()

        return SystemResources(
            cpu_count=cpu_count,
            total_memory_gb=total_mem,
            available_memory_gb=avail_mem,
            load_average_1m=load_avg,
        )

    @staticmethod
    def recommend_workers(
        resources: Optional[SystemResources] = None,
        per_worker_memory_mb: float = 150.0,
        max_utilization: float = 0.5,
        min_workers: int = 1,
        max_workers: Optional[int] = None,
    ) -> int:
        """
        Recommend a safe number of parallel workers.

        Parameters
        ----------
        resources : SystemResources, optional
            Auto-detected if None.
        per_worker_memory_mb : float
            Estimated memory per worker (opfunu + optimizer).
        max_utilization : float
            Fraction of CPUs to use (default 50% for shared HPC).
        min_workers : int
            Minimum workers to recommend.
        max_workers : int, optional
            Hard upper limit.

        Returns
        -------
        int
            Recommended worker count.
        """

        if resources is None:
            resources = ResourceMonitor.detect()

        # CPU-based limit
        cpu_based = max(1, int(resources.cpu_count * max_utilization))

        # Memory-based limit
        if resources.available_memory_gb > 0 and per_worker_memory_mb > 0:
            memory_based = max(
                1,
                int(
                    (resources.available_memory_gb * 1024)
                    / per_worker_memory_mb
                ),
            )
        else:
            memory_based = cpu_based

        # Load-aware reduction (if system is already under load)
        load_factor = 1.0
        if resources.load_average_1m is not None:
            current_load_ratio = (
                resources.load_average_1m / max(resources.cpu_count, 1)
            )
            if current_load_ratio > 0.3:
                # Reduce recommendation proportionally to existing load
                load_factor = max(0.2, 1.0 - current_load_ratio)

        recommended = int(
            min(cpu_based, memory_based) * load_factor
        )

        # Apply bounds
        recommended = max(recommended, min_workers)

        if max_workers is not None:
            recommended = min(recommended, max_workers)

        logger.info(
            f"Resource detection: "
            f"CPUs={resources.cpu_count}, "
            f"RAM={resources.available_memory_gb:.1f}GB avail, "
            f"Load={'%.1f' % resources.load_average_1m if resources.load_average_1m is not None else 'N/A'} | "
            f"Recommended workers: {recommended} "
            f"(cpu_limit={cpu_based}, mem_limit={memory_based}, "
            f"load_factor={load_factor:.2f})"
        )

        return recommended

    # ====================================================
    # Internal helpers
    # ====================================================

    @staticmethod
    def _get_cpu_count() -> int:
        """Get physical CPU count, avoiding hyper-threading oversubscription."""

        try:
            import psutil
            # Try to get physical cores first
            physical = psutil.cpu_count(logical=False)
            if physical:
                try:
                    # Respect cgroups (e.g. docker) if restricted
                    allowed = len(os.sched_getaffinity(0))
                    return min(physical, allowed)
                except AttributeError:
                    return physical
            
            # Fallback if physical is None
            try:
                return len(os.sched_getaffinity(0))
            except AttributeError:
                return os.cpu_count() or 1
        except ImportError:
            try:
                return len(os.sched_getaffinity(0))
            except AttributeError:
                return os.cpu_count() or 1

    @staticmethod
    def _get_memory() -> tuple:
        """
        Get total and available memory in GB.

        Returns (total_gb, available_gb).
        Falls back to (0, 0) if psutil is not installed.
        """

        try:
            import psutil
            mem = psutil.virtual_memory()
            return (
                mem.total / (1024 ** 3),
                mem.available / (1024 ** 3),
            )
        except ImportError:
            logger.warning(
                "psutil not installed — cannot detect memory. "
                "Install with: pip install psutil"
            )
            return (0.0, 0.0)

    @staticmethod
    def _get_load_average() -> Optional[float]:
        """
        Get 1-minute load average.

        Returns None on Windows.
        """

        try:
            load = os.getloadavg()
            return load[0]
        except (AttributeError, OSError):
            # Windows does not support getloadavg
            return None
