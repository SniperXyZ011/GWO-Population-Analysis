"""
dashboard.py

Terminal-based progress dashboard for experiment campaigns.

Provides real-time monitoring of:
    - Completed / remaining / failed experiments
    - ETA estimation
    - Per-optimizer breakdown
    - CPU and RAM usage (when psutil available)
    - Worker status
    - Throughput (experiments/sec)

Uses the `rich` library for beautiful terminal output when
available. Falls back to simple print-based updates if `rich`
is not installed.
"""

import logging
import os
import time
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CampaignDashboard:
    """
    Real-time terminal dashboard for experiment campaigns.

    Parameters
    ----------
    total_experiments : int
        Total number of experiments in the campaign.
    checkpoint_db : CheckpointDB
        For querying real-time campaign stats.
    update_interval : float
        Seconds between dashboard refreshes.
    """

    def __init__(
        self,
        total_experiments: int,
        checkpoint_db=None,
        update_interval: float = 5.0,
    ):
        self.total = total_experiments
        self.db = checkpoint_db
        self.update_interval = update_interval

        # Counters (updated via callbacks)
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.start_time = time.time()

        # Background update thread
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

        # Recent timing for ETA
        self._recent_times = []

    # =========================================================
    # Public API
    # =========================================================

    def start(self):
        """Start the dashboard background thread."""

        self.start_time = time.time()
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._background_loop,
            daemon=True,
            name="dashboard",
        )
        self._thread.start()

        logger.info("Campaign dashboard started")

    def stop(self):
        """Stop the dashboard background thread."""

        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

        # Print final summary
        self._print_final_summary()

    def update(self, completed: int, total: int, outcome: dict):
        """
        Callback for the scheduler — called after each experiment.

        Parameters
        ----------
        completed : int
            Total done (completed + skipped + failed).
        total : int
        outcome : dict
            From _run_single_experiment.
        """

        with self._lock:
            status = outcome.get("status", "unknown")

            if status == "completed":
                self.completed += 1
                exec_time = (
                    outcome.get("result", {}).get("execution_time", 0)
                    if outcome.get("result") else 0
                )
                self._recent_times.append(exec_time)

                # Keep only last 100 for rolling average
                if len(self._recent_times) > 100:
                    self._recent_times = self._recent_times[-100:]

            elif status == "skipped":
                self.skipped += 1
            elif status == "failed":
                self.failed += 1

    # =========================================================
    # Progress Callback (for scheduler integration)
    # =========================================================

    def progress_callback(self, done: int, total: int, outcome: dict):
        """
        Scheduler-compatible progress callback.

        Pass this method to ExperimentScheduler(progress_callback=...).
        """
        self.update(done, total, outcome)

    # =========================================================
    # Display Logic
    # =========================================================

    def _background_loop(self):
        """Background thread that periodically prints progress."""

        while not self._stop_event.is_set():
            self._print_progress()
            self._stop_event.wait(self.update_interval)

    def _print_progress(self):
        """Print current progress to terminal."""

        with self._lock:
            elapsed = time.time() - self.start_time
            done = self.completed + self.failed + self.skipped
            pending = self.total - done

            # Throughput
            rate = (
                (self.completed + self.failed) / max(elapsed, 1e-9)
            )

            # ETA
            if rate > 0 and pending > 0:
                eta_seconds = pending / rate
                eta_str = _format_duration(eta_seconds)
            else:
                eta_str = "N/A"

            # Average time per experiment
            avg_time = (
                sum(self._recent_times) / len(self._recent_times)
                if self._recent_times else 0
            )

            # System resources
            cpu_str, ram_str = _get_system_usage()

            # Format output
            pct = done / max(self.total, 1) * 100
            bar = _progress_bar(pct, width=30)

        # Print outside lock to minimize hold time
        lines = [
            "",
            "═" * 62,
            "  GWO Population Analysis — Campaign Progress",
            "═" * 62,
            f"  {bar}  {pct:.1f}%",
            f"  Total: {self.total:,} | Done: {done:,} | "
            f"Remaining: {pending:,}",
            f"  Completed: {self.completed:,} | "
            f"Skipped: {self.skipped:,} | "
            f"Failed: {self.failed:,}",
            f"  Elapsed: {_format_duration(elapsed)} | "
            f"ETA: {eta_str}",
            f"  Rate: {rate:.1f} exp/s | "
            f"Avg time: {avg_time:.2f}s/exp",
            f"  {cpu_str} | {ram_str}",
            "═" * 62,
        ]

        print("\n".join(lines), flush=True)

    def _print_final_summary(self):
        """Print final campaign summary."""

        elapsed = time.time() - self.start_time
        done = self.completed + self.failed + self.skipped

        rate = (
            (self.completed + self.failed) / max(elapsed, 1e-9)
        )

        lines = [
            "",
            "╔" + "═" * 60 + "╗",
            "║  CAMPAIGN COMPLETE" + " " * 41 + "║",
            "╠" + "═" * 60 + "╣",
            f"║  Total:     {self.total:>10,}" + " " * 37 + "║",
            f"║  Completed: {self.completed:>10,}" + " " * 37 + "║",
            f"║  Skipped:   {self.skipped:>10,}" + " " * 37 + "║",
            f"║  Failed:    {self.failed:>10,}" + " " * 37 + "║",
            f"║  Elapsed:   {_format_duration(elapsed):>10}" + " " * 37 + "║",
            f"║  Rate:      {rate:>9.1f}/s" + " " * 37 + "║",
            "╚" + "═" * 60 + "╝",
        ]

        print("\n".join(lines), flush=True)

    # =========================================================
    # Query DB for per-optimizer stats
    # =========================================================

    def get_optimizer_progress(self) -> dict:
        """
        Get per-optimizer completion counts from checkpoint DB.

        Returns
        -------
        dict
            {optimizer: {"completed": n, "avg_time": float}}
        """

        if self.db is None:
            return {}

        stats = self.db.get_campaign_stats()
        return stats.get("per_optimizer", {})


# =============================================================
# Utility functions
# =============================================================

def _progress_bar(pct: float, width: int = 30) -> str:
    """Create a text-based progress bar."""

    filled = int(width * pct / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""

    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m}m"


def _get_system_usage() -> tuple:
    """
    Get CPU and RAM usage strings.

    Returns ("CPU: XX%", "RAM: XX.X GB / YY.Y GB")
    """

    try:
        import psutil

        cpu = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory()

        cpu_str = f"CPU: {cpu:.0f}%"
        ram_str = (
            f"RAM: {mem.used / (1024**3):.1f} GB / "
            f"{mem.total / (1024**3):.0f} GB "
            f"({mem.percent:.0f}%)"
        )
        return cpu_str, ram_str

    except ImportError:
        return "CPU: N/A", "RAM: N/A (install psutil)"
