"""
scheduler.py

Parallel experiment execution engine.

Uses concurrent.futures.ProcessPoolExecutor for embarrassingly
parallel execution of independent optimization experiments.

Key design decisions:
    - ExperimentRunner.run() remains the unit of work
    - Workers are fully isolated (separate processes)
    - Results are written to checkpoint DB from the main process
    - Failed experiments are tracked and optionally retried
    - Progress is reported via callback
"""

import logging
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed, FIRST_COMPLETED, wait
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from core.checkpoint import CheckpointDB
from core.experiment import Experiment
from core.environment import EnvironmentInfo
from core.resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)


# =============================================================
# Top-level worker function (must be picklable)
# =============================================================

def _run_single_experiment(
    experiment: Experiment,
    results_dir: str,
    overwrite: bool = False,
) -> dict:
    """
    Execute a single experiment in a worker process.

    This is the top-level function submitted to ProcessPoolExecutor.
    It must be defined at module level to be picklable.

    Parameters
    ----------
    experiment : Experiment
    results_dir : str
    overwrite : bool

    Returns
    -------
    dict
        Contains 'experiment', 'result', and 'status'.
    """

    # Lazy imports inside the worker to avoid shared state
    import sys
    import os

    # Ensure project root is on path for the worker
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from core.runner import ExperimentRunner

    try:
        runner = ExperimentRunner(results_dir=results_dir, overwrite=overwrite)
        result = runner.run(experiment)

        return {
            "experiment": experiment,
            "result": result,
            "status": "completed" if result is not None else "skipped",
            "error": None,
        }

    except Exception as e:
        return {
            "experiment": experiment,
            "result": None,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def _worker_init():
    """Pin the worker to a specific CPU core to optimize NUMA access."""
    import os
    try:
        import psutil
        p = psutil.Process()
        cores = p.cpu_affinity()
        if cores:
            # Distribute workers across available cores using PID
            core = cores[os.getpid() % len(cores)]
            p.cpu_affinity([core])
    except (AttributeError, ImportError, NotImplementedError):
        pass


class ExperimentScheduler:
    """
    Parallel experiment execution engine.

    Orchestrates:
        1. Filtering already-completed experiments via CheckpointDB
        2. Submitting pending experiments to ProcessPoolExecutor
        3. Recording results as they complete
        4. Tracking failures and optional retries
        5. Progress reporting via callback

    Parameters
    ----------
    checkpoint_db : CheckpointDB
        SQLite checkpoint database.
    results_dir : Path
        Directory for file-based result storage (JSON + CSV).
    max_workers : int
        Number of parallel worker processes.
        0 = auto-detect via ResourceMonitor.
    max_retries : int
        Maximum retry count for failed experiments.
    overwrite : bool
        If True, re-run already-completed experiments.
    progress_callback : callable, optional
        Called with (completed, total, last_result_dict) after each completion.
    """

    def __init__(
        self,
        checkpoint_db: CheckpointDB,
        results_dir: Path,
        max_workers: int = 0,
        max_retries: int = 0,
        overwrite: bool = False,
        progress_callback: Optional[Callable] = None,
    ):

        self.db = checkpoint_db
        self.results_dir = Path(results_dir)
        self.max_retries = max_retries
        self.overwrite = overwrite
        self.progress_callback = progress_callback

        # Resolve worker count
        if max_workers <= 0:
            self.max_workers = ResourceMonitor.recommend_workers()
        else:
            self.max_workers = max_workers

        # Capture environment once for the campaign
        self._env = EnvironmentInfo.capture()

        logger.info(f"Scheduler initialized: workers={self.max_workers}")

    # =========================================================
    # Main campaign execution
    # =========================================================

    def run_campaign(
        self,
        experiments: Iterable[Experiment],
    ) -> dict:
        """
        Execute all experiments with parallel workers.

        Parameters
        ----------
        experiments : iterable of Experiment
            All experiments to run (including already-completed ones).

        Returns
        -------
        dict
            Campaign summary: completed, skipped, failed, elapsed.
        """

        # Store campaign environment metadata
        self.db.store_environment(self._env)

        # 1. Determine pending experiments
        all_experiments = list(experiments)
        total_all = len(all_experiments)

        if self.overwrite:
            pending = all_experiments
        else:
            completed_ids = self.db.get_completed_ids()
            pending = [
                exp for exp in all_experiments
                if exp.experiment_name not in completed_ids
            ]

        skipped = total_all - len(pending)
        total_pending = len(pending)

        logger.info(
            f"Campaign: {total_all} total, "
            f"{skipped} already completed, "
            f"{total_pending} pending"
        )

        if total_pending == 0:
            logger.info("All experiments already completed. Nothing to do.")
            return {
                "total": total_all,
                "completed": 0,
                "skipped": skipped,
                "failed": 0,
                "elapsed": 0.0,
            }

        # 2. Execute in parallel
        completed = 0
        failed = 0
        start_time = time.time()
        results_dir_str = str(self.results_dir)

        if self.max_workers == 1:
            # Sequential mode — useful for debugging
            batch_outcomes = []
            for exp in pending:
                outcome = _run_single_experiment(
                    exp, results_dir_str, self.overwrite
                )
                
                outcome["hostname"] = self._env.hostname
                outcome["git_hash"] = self._env.git_hash or ""
                outcome["python_version"] = self._env.python_version

                self._handle_outcome(outcome)

                if outcome["status"] == "completed":
                    completed += 1
                    batch_outcomes.append(outcome)
                    if len(batch_outcomes) >= 500:
                        self.db.record_results_batch(batch_outcomes)
                        batch_outcomes.clear()
                elif outcome["status"] == "failed":
                    failed += 1

                if self.progress_callback:
                    self.progress_callback(
                        completed + skipped + failed,
                        total_all,
                        outcome,
                    )
                    
            if batch_outcomes:
                self.db.record_results_batch(batch_outcomes)
                batch_outcomes.clear()
        else:
            # Parallel mode
            with ProcessPoolExecutor(
                max_workers=self.max_workers,
                max_tasks_per_child=1000,
                initializer=_worker_init
            ) as executor:

                future_to_exp = {}
                active_futures = set()
                pending_iter = iter(pending)
                batch_outcomes = []
                
                # Initial fill
                for _ in range(self.max_workers * 2):
                    try:
                        exp = next(pending_iter)
                        future = executor.submit(
                            _run_single_experiment,
                            exp,
                            results_dir_str,
                            self.overwrite,
                        )
                        future_to_exp[future] = exp
                        active_futures.add(future)
                    except StopIteration:
                        break

                while active_futures:
                    done, active_futures = wait(active_futures, return_when=FIRST_COMPLETED)
                    
                    for future in done:
                        try:
                            outcome = future.result(timeout=3600)
                        except Exception as e:
                            exp = future_to_exp[future]
                            outcome = {
                                "experiment": exp,
                                "result": None,
                                "status": "failed",
                                "error": str(e),
                                "traceback": traceback.format_exc(),
                            }

                        outcome["hostname"] = self._env.hostname
                        outcome["git_hash"] = self._env.git_hash or ""
                        outcome["python_version"] = self._env.python_version

                        self._handle_outcome(outcome)

                        if outcome["status"] == "completed":
                            completed += 1
                            batch_outcomes.append(outcome)
                            if len(batch_outcomes) >= 500:
                                self.db.record_results_batch(batch_outcomes)
                                batch_outcomes.clear()
                        elif outcome["status"] == "failed":
                            failed += 1

                        done_count = completed + skipped + failed
                        if self.progress_callback:
                            self.progress_callback(done_count, total_all, outcome)
                        elif done_count % max(1, total_pending // 20) == 0:
                            elapsed = time.time() - start_time
                            rate = done_count / max(elapsed, 1e-9)
                            eta = (total_pending - (completed + failed)) / max(rate, 1e-9)

                            logger.info(
                                f"Progress: {done_count}/{total_all} "
                                f"({done_count / total_all * 100:.1f}%) | "
                                f"Completed={completed + skipped} "
                                f"Failed={failed} | "
                                f"Rate={rate:.1f}/s | "
                                f"ETA={eta / 3600:.1f}h"
                            )
                        
                        del future_to_exp[future]

                        # Replenish queue
                        try:
                            exp = next(pending_iter)
                            new_future = executor.submit(
                                _run_single_experiment,
                                exp,
                                results_dir_str,
                                self.overwrite,
                            )
                            future_to_exp[new_future] = exp
                            active_futures.add(new_future)
                        except StopIteration:
                            pass

                if batch_outcomes:
                    self.db.record_results_batch(batch_outcomes)
                    batch_outcomes.clear()

        elapsed = time.time() - start_time

        summary = {
            "total": total_all,
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "elapsed": elapsed,
            "experiments_per_second": (
                (completed + failed) / max(elapsed, 1e-9)
            ),
        }

        logger.info("=" * 60)
        logger.info("CAMPAIGN COMPLETE")
        logger.info(f"  Total:     {total_all}")
        logger.info(f"  Completed: {completed}")
        logger.info(f"  Skipped:   {skipped}")
        logger.info(f"  Failed:    {failed}")
        logger.info(f"  Time:      {elapsed / 3600:.2f} hours")
        logger.info(
            f"  Throughput: {summary['experiments_per_second']:.1f} exp/s"
        )
        logger.info("=" * 60)

        return summary

    # =========================================================
    # Outcome handling
    # =========================================================

    def _handle_outcome(self, outcome: dict):
        """Process a single experiment outcome (failures only for DB)."""

        experiment = outcome["experiment"]
        status = outcome["status"]

        # Successful results are now batched in run_campaign
        if status == "failed":

            error_msg = outcome.get("error", "Unknown error")
            tb = outcome.get("traceback", "")

            logger.error(
                f"FAILED: {experiment} | {error_msg}"
            )

            self.db.record_failure(
                experiment=experiment,
                error_message=error_msg,
                error_traceback=tb,
            )

            # Retry logic
            retry_count = self.db.get_retry_count(experiment)
            if retry_count < self.max_retries:
                logger.info(
                    f"RETRY ({retry_count + 1}/{self.max_retries}): "
                    f"{experiment}"
                )
                # The experiment will be retried on next campaign run
                # since it won't be in the completed set

    # =========================================================
    # Convenience
    # =========================================================

    def __repr__(self) -> str:
        return (
            f"ExperimentScheduler("
            f"workers={self.max_workers}, "
            f"retries={self.max_retries}, "
            f"db={self.db.db_path})"
        )
