"""
checkpoint.py

SQLite-backed checkpoint database for experiment tracking.

Provides:
    - Atomic result recording (crash-safe)
    - O(log n) completion checks (indexed queries)
    - Instant resume after interruption
    - Campaign metadata storage
    - Failure tracking with retry support

Uses WAL mode for concurrent read access from monitoring tools.
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from core.experiment import Experiment


class CheckpointDB:
    """
    SQLite checkpoint database.

    Thread-safe: uses a threading lock around all write operations.
    WAL mode enables concurrent reads from monitoring/analysis tools.
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Path):

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._write_lock = threading.Lock()

        self._init_db()

    # =========================================================
    # Connection management
    # =========================================================

    @contextmanager
    def _connection(self):
        """Yield a SQLite connection with proper cleanup."""

        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row

        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # =========================================================
    # Schema initialization
    # =========================================================

    def _init_db(self):
        """Create tables if they do not exist."""

        with self._connection() as conn:

            conn.executescript("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                );

                CREATE TABLE IF NOT EXISTS campaign_metadata (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE TABLE IF NOT EXISTS results (
                    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id           TEXT    UNIQUE NOT NULL,
                    benchmark               TEXT    NOT NULL,
                    function                INTEGER NOT NULL,
                    dimension               INTEGER NOT NULL,
                    optimizer               TEXT    NOT NULL,
                    population_size         INTEGER NOT NULL,
                    max_fe                  INTEGER NOT NULL,
                    run                     INTEGER NOT NULL,
                    seed                    INTEGER NOT NULL,
                    best_score              REAL,
                    function_evaluations    INTEGER,
                    iterations              INTEGER,
                    execution_time          REAL,
                    fe_per_second           REAL,
                    best_position           TEXT,
                    timestamp               TEXT,
                    hostname                TEXT,
                    git_hash                TEXT,
                    python_version          TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_results_experiment_id
                    ON results(experiment_id);

                CREATE INDEX IF NOT EXISTS idx_results_benchmark
                    ON results(benchmark, optimizer, function, dimension, population_size);

                CREATE INDEX IF NOT EXISTS idx_results_optimizer
                    ON results(optimizer);

                CREATE TABLE IF NOT EXISTS convergence (
                    result_id    INTEGER NOT NULL,
                    iteration    INTEGER NOT NULL,
                    best_fitness REAL    NOT NULL,
                    PRIMARY KEY (result_id, iteration),
                    FOREIGN KEY (result_id) REFERENCES results(id)
                );

                CREATE TABLE IF NOT EXISTS failures (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id   TEXT    NOT NULL,
                    error_message   TEXT,
                    error_traceback TEXT,
                    timestamp       TEXT,
                    retry_count     INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_failures_experiment_id
                    ON failures(experiment_id);
            """)

            # Store schema version
            existing = conn.execute(
                "SELECT version FROM schema_version"
            ).fetchone()

            if existing is None:
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (self.SCHEMA_VERSION,),
                )

    # =========================================================
    # Campaign metadata
    # =========================================================

    def store_metadata(self, key: str, value: str):
        """Store a campaign-level metadata key-value pair."""

        with self._write_lock:
            with self._connection() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO campaign_metadata
                       (key, value) VALUES (?, ?)""",
                    (key, value),
                )

    def get_metadata(self, key: str) -> Optional[str]:
        """Retrieve a campaign metadata value."""

        with self._connection() as conn:
            row = conn.execute(
                "SELECT value FROM campaign_metadata WHERE key = ?",
                (key,),
            ).fetchone()

            return row["value"] if row else None

    def store_environment(self, env_info):
        """
        Store environment info as campaign metadata.

        Parameters
        ----------
        env_info : EnvironmentInfo
        """

        self.store_metadata(
            "environment",
            json.dumps(env_info.to_dict(), indent=2),
        )

    # =========================================================
    # Completion checks
    # =========================================================

    def is_completed(self, experiment: Experiment) -> bool:
        """
        Check if an experiment has been completed.

        O(log n) via indexed lookup on experiment_id.
        """

        exp_id = experiment.experiment_name

        with self._connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM results WHERE experiment_id = ?",
                (exp_id,),
            ).fetchone()

            return row is not None

    def get_completed_ids(self) -> Set[str]:
        """
        Get all completed experiment IDs in one query.

        Much faster than calling is_completed() in a loop.

        Returns
        -------
        set of str
            All experiment_id values in the results table.
        """

        with self._connection() as conn:
            rows = conn.execute(
                "SELECT experiment_id FROM results"
            ).fetchall()

            return {row["experiment_id"] for row in rows}

    def get_completed_count(self) -> int:
        """Return the total number of completed experiments."""

        with self._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM results"
            ).fetchone()

            return row["cnt"]

    def get_failed_count(self) -> int:
        """Return the total number of failed experiments."""

        with self._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT experiment_id) as cnt FROM failures"
            ).fetchone()

            return row["cnt"]

    # =========================================================
    # Result recording
    # =========================================================

    def record_result(
        self,
        experiment: Experiment,
        result: Dict[str, Any],
        hostname: str = "",
        git_hash: str = "",
        python_version: str = "",
    ):
        """
        Atomically record a completed experiment result.

        Parameters
        ----------
        experiment : Experiment
        result : dict
            Output from BaseOptimizer.get_results().
        hostname : str
        git_hash : str
        python_version : str
        """

        from datetime import datetime, timezone

        exp_id = experiment.experiment_name
        now = datetime.now(timezone.utc).isoformat()

        best_position = result.get("best_position")
        if best_position is not None:
            best_position = json.dumps(best_position)

        fe = result.get("function_evaluations", 0)
        exec_time = result.get("execution_time", 0.0)
        fe_per_sec = fe / max(exec_time, 1e-9)

        with self._write_lock:
            with self._connection() as conn:

                cursor = conn.execute(
                    """INSERT OR REPLACE INTO results (
                        experiment_id, benchmark, function, dimension,
                        optimizer, population_size, max_fe, run, seed,
                        best_score, function_evaluations, iterations,
                        execution_time, fe_per_second, best_position,
                        timestamp, hostname, git_hash, python_version
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )""",
                    (
                        exp_id,
                        experiment.benchmark,
                        experiment.function,
                        experiment.dimension,
                        experiment.optimizer,
                        experiment.population_size,
                        experiment.max_function_evaluations,
                        experiment.run,
                        experiment.seed,
                        result.get("best_score"),
                        fe,
                        result.get("iterations"),
                        exec_time,
                        fe_per_sec,
                        best_position,
                        now,
                        hostname,
                        git_hash,
                        python_version,
                    ),
                )

                result_id = cursor.lastrowid

                # Store convergence curve
                curve = result.get("convergence_curve", [])
                if curve:
                    # Subsample to max 1000 points for storage
                    step = max(1, len(curve) // 1000)
                    sampled = [
                        (result_id, i, float(curve[i]))
                        for i in range(0, len(curve), step)
                    ]
                    # Always include the last point
                    if sampled and sampled[-1][1] != len(curve) - 1:
                        sampled.append(
                            (result_id, len(curve) - 1, float(curve[-1]))
                        )

                    conn.executemany(
                        """INSERT INTO convergence
                           (result_id, iteration, best_fitness)
                           VALUES (?, ?, ?)""",
                        sampled,
                    )

    # =========================================================
    # Failure tracking
    # =========================================================

    def record_failure(
        self,
        experiment: Experiment,
        error_message: str,
        error_traceback: str = "",
    ):
        """Record a failed experiment for retry tracking."""

        from datetime import datetime, timezone

        exp_id = experiment.experiment_name
        now = datetime.now(timezone.utc).isoformat()

        with self._write_lock:
            with self._connection() as conn:

                # Increment retry count
                existing = conn.execute(
                    """SELECT MAX(retry_count) as max_retry
                       FROM failures
                       WHERE experiment_id = ?""",
                    (exp_id,),
                ).fetchone()

                retry = 0
                if existing and existing["max_retry"] is not None:
                    retry = existing["max_retry"] + 1

                conn.execute(
                    """INSERT INTO failures
                       (experiment_id, error_message, error_traceback,
                        timestamp, retry_count)
                       VALUES (?, ?, ?, ?, ?)""",
                    (exp_id, error_message, error_traceback, now, retry),
                )

    def get_retry_count(self, experiment: Experiment) -> int:
        """Get the number of times an experiment has been retried."""

        exp_id = experiment.experiment_name

        with self._connection() as conn:
            row = conn.execute(
                """SELECT MAX(retry_count) as max_retry
                   FROM failures
                   WHERE experiment_id = ?""",
                (exp_id,),
            ).fetchone()

            if row and row["max_retry"] is not None:
                return row["max_retry"]

            return 0

    # =========================================================
    # Query helpers (for analysis / monitoring)
    # =========================================================

    def query_results(
        self,
        benchmark: Optional[str] = None,
        optimizer: Optional[str] = None,
        dimension: Optional[int] = None,
        function: Optional[int] = None,
        population_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query results with optional filters.

        Returns list of result dicts.
        """

        clauses = []
        params = []

        if benchmark is not None:
            clauses.append("benchmark = ?")
            params.append(benchmark)

        if optimizer is not None:
            clauses.append("optimizer = ?")
            params.append(optimizer)

        if dimension is not None:
            clauses.append("dimension = ?")
            params.append(dimension)

        if function is not None:
            clauses.append("function = ?")
            params.append(function)

        if population_size is not None:
            clauses.append("population_size = ?")
            params.append(population_size)

        where = ""
        if clauses:
            where = "WHERE " + " AND ".join(clauses)

        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM results {where} ORDER BY id",
                params,
            ).fetchall()

            return [dict(row) for row in rows]

    def get_campaign_stats(self) -> Dict[str, Any]:
        """
        Get campaign-level statistics for monitoring.

        Returns
        -------
        dict
            completed, failed, avg_time, total_time, etc.
        """

        with self._connection() as conn:

            stats = {}

            row = conn.execute(
                """SELECT
                     COUNT(*) as completed,
                     AVG(execution_time) as avg_time,
                     SUM(execution_time) as total_time,
                     MIN(execution_time) as min_time,
                     MAX(execution_time) as max_time,
                     AVG(fe_per_second) as avg_fe_per_sec
                   FROM results"""
            ).fetchone()

            stats["completed"] = row["completed"]
            stats["avg_time"] = row["avg_time"] or 0.0
            stats["total_time"] = row["total_time"] or 0.0
            stats["min_time"] = row["min_time"] or 0.0
            stats["max_time"] = row["max_time"] or 0.0
            stats["avg_fe_per_sec"] = row["avg_fe_per_sec"] or 0.0

            fail_row = conn.execute(
                "SELECT COUNT(DISTINCT experiment_id) as cnt FROM failures"
            ).fetchone()
            stats["failed"] = fail_row["cnt"]

            # Per-optimizer breakdown
            opt_rows = conn.execute(
                """SELECT optimizer, COUNT(*) as cnt,
                          AVG(execution_time) as avg_time
                   FROM results
                   GROUP BY optimizer
                   ORDER BY optimizer"""
            ).fetchall()

            stats["per_optimizer"] = {
                r["optimizer"]: {
                    "completed": r["cnt"],
                    "avg_time": r["avg_time"],
                }
                for r in opt_rows
            }

            return stats

    # =========================================================
    # Convenience
    # =========================================================

    def __repr__(self) -> str:
        completed = self.get_completed_count()
        failed = self.get_failed_count()
        return (
            f"CheckpointDB(path={self.db_path}, "
            f"completed={completed}, failed={failed})"
        )
