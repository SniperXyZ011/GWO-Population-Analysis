"""
main.py

Entry point for the GWO Population Analysis Framework.

Subcommands:
    run      — Execute experiments (parallel by default)
    status   — Show campaign progress
    analyze  — Run statistical analysis (Phase 3)
    report   — Generate publication reports (Phase 4)

Usage:
    python main.py run --config configs/experiment.yaml
    python main.py run --benchmark CEC2020 --optimizer GWO --dimension 10 --workers 64
    python main.py run --workers 1  (sequential / debug mode)
    python main.py status
"""

import argparse
import json
import sys
import time

# --- Register all benchmarks and optimizers ---
import benchmarks.cec2017  # noqa: F401
import benchmarks.cec2020  # noqa: F401
import benchmarks.cec2022  # noqa: F401
import optimizers           # noqa: F401

from core.parameter_grid import ParameterGrid
from core.registry import optimizer_registry, benchmark_registry
from core.checkpoint import CheckpointDB
from core.scheduler import ExperimentScheduler
from core.environment import EnvironmentInfo
from configs.yaml_config import ExperimentConfig
from utils.logger import setup_logger


# =============================================================
# CLI Argument Parser
# =============================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""

    parser = argparse.ArgumentParser(
        prog="gwo-framework",
        description="GWO Population Analysis Framework — HPC Edition",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    # ---------------------------------------------------------
    # Subcommand: run
    # ---------------------------------------------------------

    run_parser = subparsers.add_parser(
        "run",
        help="Execute optimization experiments",
    )

    run_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )

    run_parser.add_argument(
        "--benchmark",
        nargs="*",
        default=None,
        help="Benchmark suites (e.g., CEC2017 CEC2020)",
    )

    run_parser.add_argument(
        "--optimizer",
        nargs="*",
        default=None,
        help="Optimizers (e.g., GWO BBGWO REGWO)",
    )

    run_parser.add_argument(
        "--dimension",
        nargs="*",
        type=int,
        default=None,
        help="Dimensions (e.g., 10 30 50)",
    )

    run_parser.add_argument(
        "--population",
        nargs="*",
        type=int,
        default=None,
        help="Population sizes (e.g., 10 50 100)",
    )

    run_parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Number of independent runs",
    )

    run_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (0 = auto, 1 = sequential)",
    )

    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-run already-completed experiments",
    )

    # ---------------------------------------------------------
    # Subcommand: status
    # ---------------------------------------------------------

    status_parser = subparsers.add_parser(
        "status",
        help="Show campaign progress from checkpoint database",
    )

    status_parser.add_argument(
        "--db",
        type=str,
        default="checkpoint.db",
        help="Path to checkpoint database",
    )

    # ---------------------------------------------------------
    # Backward-compatible: no subcommand = run
    # ---------------------------------------------------------

    # Also add run-level args to the top-level parser for backward compat
    parser.add_argument(
        "--benchmark",
        nargs="*",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--optimizer",
        nargs="*",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dimension",
        nargs="*",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--population",
        nargs="*",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )

    return parser


# =============================================================
# Subcommand: run
# =============================================================

def cmd_run(args):
    """Execute the experiment campaign."""

    logger = setup_logger()

    # --- Display registered components ---
    logger.info(
        f"Registered Optimizers: {optimizer_registry.list()}"
    )
    logger.info(
        f"Registered Benchmarks: {benchmark_registry.list()}"
    )

    # --- Build configuration ---
    config = ExperimentConfig.from_cli_args(args)

    logger.info(f"Configuration: {config}")
    logger.info(f"Config details: {json.dumps(config.to_dict(), indent=2)}")

    # --- Capture environment ---
    env = EnvironmentInfo.capture()
    logger.info(f"Environment: {env.hostname} | {env.python_version.split()[0]} | git={env.git_hash}")

    # --- Build parameter grid ---
    grid = ParameterGrid(
        benchmarks=config.benchmarks,
        optimizers=config.optimizers,
        dimensions=config.dimensions,
        populations=config.population_sizes,
        runs=config.runs,
    )

    total = len(grid)
    logger.info(f"Total experiments in grid: {total}")

    if total == 0:
        logger.warning("No experiments to run. Check your configuration.")
        return

    # --- Initialize checkpoint DB ---
    db = CheckpointDB(config.checkpoint_db)

    logger.info(f"Checkpoint DB: {config.checkpoint_db}")
    logger.info(f"Previously completed: {db.get_completed_count()}")

    # Store config in DB for reproducibility
    db.store_metadata(
        "config",
        json.dumps(config.to_dict(), indent=2),
    )

    # --- Create scheduler and run ---
    scheduler = ExperimentScheduler(
        checkpoint_db=db,
        results_dir=config.results_dir,
        max_workers=config.workers,
        max_retries=config.max_retries,
        overwrite=config.overwrite,
    )

    summary = scheduler.run_campaign(grid.generate())

    return summary


# =============================================================
# Subcommand: status
# =============================================================

def cmd_status(args):
    """Display campaign progress from checkpoint database."""

    from pathlib import Path

    db_path = Path(args.db)

    if not db_path.exists():
        print(f"Checkpoint database not found: {db_path}")
        print("Run 'python main.py run' first to create it.")
        return

    db = CheckpointDB(db_path)
    stats = db.get_campaign_stats()

    print("\n" + "=" * 60)
    print("    GWO Population Analysis — Campaign Status")
    print("=" * 60)
    print(f"  Database:    {db_path}")
    print(f"  Completed:   {stats['completed']}")
    print(f"  Failed:      {stats['failed']}")
    print(f"  Avg time:    {stats['avg_time']:.2f}s")
    print(f"  Total time:  {stats['total_time'] / 3600:.2f}h")
    print(f"  Avg FE/sec:  {stats['avg_fe_per_sec']:.0f}")

    if stats["per_optimizer"]:
        print("\n  Per-optimizer breakdown:")
        print(f"  {'Optimizer':<15} {'Completed':>10} {'Avg Time':>10}")
        print(f"  {'-' * 15} {'-' * 10} {'-' * 10}")

        for name, info in sorted(stats["per_optimizer"].items()):
            print(
                f"  {name:<15} "
                f"{info['completed']:>10} "
                f"{info['avg_time']:>9.2f}s"
            )

    # Show environment info
    env_json = db.get_metadata("environment")
    if env_json:
        env = json.loads(env_json)
        print(f"\n  Hostname:    {env.get('hostname', 'N/A')}")
        print(f"  Python:      {env.get('python_version', 'N/A').split()[0]}")
        print(f"  NumPy:       {env.get('numpy_version', 'N/A')}")
        print(f"  Git:         {env.get('git_hash', 'N/A')}")

    print("=" * 60 + "\n")


# =============================================================
# Entry point
# =============================================================

def main():

    parser = build_parser()
    args = parser.parse_args()

    # Default to "run" for backward compatibility
    command = args.command or "run"

    if command == "run":
        cmd_run(args)
    elif command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
