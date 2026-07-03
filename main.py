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
import benchmarks.cec2013  # noqa: F401
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
    import datetime
    import json
    import os
    import psutil
    from core.parameter_grid import ParameterGrid
    from core.resource_monitor import ResourceMonitor

    db_path = Path(args.db)

    if not db_path.exists():
        print(f"Checkpoint database not found: {db_path}")
        print("Run 'python main.py run' first to create it.")
        return

    db = CheckpointDB(db_path)
    stats = db.get_campaign_stats()

    # Attempt to calculate total progress from stored config
    config_json = db.get_metadata("config")
    total_experiments = 0
    num_workers = 0
    results_dir = "results"

    if config_json:
        try:
            config_dict = json.loads(config_json)
            grid = ParameterGrid(
                benchmarks=config_dict.get("benchmarks", []),
                optimizers=config_dict.get("optimizers", []),
                dimensions=config_dict.get("dimensions", []),
                populations=config_dict.get("population_sizes", []),
                runs=config_dict.get("runs", 1)
            )
            total_experiments = len(grid)
            num_workers = config_dict.get("workers", 0)
            results_dir = config_dict.get("results_dir", "results")
        except Exception:
            pass

    if not num_workers:
        num_workers = ResourceMonitor.recommend_workers()

    completed = stats["completed"]
    failed = stats["failed"]
    total_processed = completed + failed
    pending = max(0, total_experiments - total_processed) if total_experiments > 0 else 0

    avg_time = stats["avg_time"]
    total_cpu_hours = stats["total_time"] / 3600.0
    
    start_time_str = stats.get("start_time")
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    
    elapsed_wall_hours = 0.0
    speedup = 0.0
    campaign_id = "N/A"
    
    if start_time_str:
        try:
            # Handle ISO format strings gracefully
            if start_time_str.endswith('Z'):
                start_time_str = start_time_str[:-1] + '+00:00'
            start_time = datetime.datetime.fromisoformat(start_time_str)
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=datetime.timezone.utc)
            
            campaign_id = start_time.strftime("%Y-%m-%d_%H-%M")
            elapsed_wall_delta = now_utc - start_time
            elapsed_wall_hours = elapsed_wall_delta.total_seconds() / 3600.0
            if elapsed_wall_hours > 0:
                speedup = total_cpu_hours / elapsed_wall_hours
        except Exception:
            pass

    status_str = "RUNNING" if pending > 0 else "COMPLETED"
    if total_experiments == 0:
        status_str = "UNKNOWN"

    progress_pct = (total_processed / total_experiments * 100) if total_experiments > 0 else 0.0

    exp_per_hour = total_processed / elapsed_wall_hours if elapsed_wall_hours > 0 else 0
    throughput_fe_sec = stats["avg_fe_per_sec"] * speedup

    eta_hours = (pending / exp_per_hour) if exp_per_hour > 0 else 0
    eta_delta = datetime.timedelta(hours=eta_hours)
    expected_finish = (now_utc + eta_delta).strftime("%Y-%m-%d %H:%M") if pending > 0 else "N/A"
    
    def format_hours(h):
        if h == 0: return "0h 0m"
        hours = int(h)
        mins = int((h - hours) * 60)
        return f"{hours}h {mins:02d}m"

    # Resource collection
    cpu_usage = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    
    def get_dir_size(path):
        total = 0
        try:
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        total += os.path.getsize(fp)
        except Exception:
            pass
        return total

    db_size = os.path.getsize(db_path) if db_path.exists() else 0
    results_size = get_dir_size(results_dir)
    disk_used_gb = (db_size + results_size) / (1024**3)

    print("\n" + "=" * 63)
    print("            GWO Population Analysis - Campaign Status")
    print("=" * 63)
    
    print("\nCampaign")
    print("-" * 63)
    print(f"Campaign ID        : {campaign_id}")
    print(f"Database           : {db_path.name}")
    print(f"Status             : {status_str}")

    print("\nProgress")
    print("-" * 63)
    print(f"Completed          : {completed:,}")
    print(f"Failed             : {failed:,}")
    print(f"Pending            : {pending:,}")
    print(f"Total Experiments  : {total_experiments:,}")
    print(f"\nProgress           : {progress_pct:.1f}%")

    print("\nExecution")
    print("-" * 63)
    print(f"Elapsed Wall Time  : {format_hours(elapsed_wall_hours)}")
    print(f"CPU Time Used      : {total_cpu_hours:.2f}h")
    print(f"Parallel Speedup   : {speedup:.1f}x")
    print(f"\nAverage Runtime    : {avg_time:.2f} s")
    print(f"Average FE/sec     : {stats['avg_fe_per_sec']:,.0f}")

    print("\nThroughput")
    print("-" * 63)
    print(f"Experiments/hour   : {exp_per_hour:,.0f}")
    if throughput_fe_sec > 1e6:
        print(f"Function Eval/sec  : {throughput_fe_sec/1e6:.2f}e6")
    else:
        print(f"Function Eval/sec  : {throughput_fe_sec:,.0f}")

    print("\nEstimated Remaining")
    print("-" * 63)
    print(f"ETA                : {format_hours(eta_hours) if pending > 0 else '0h 0m'}")
    print(f"Expected Finish    : {expected_finish}")

    print("\nResources")
    print("-" * 63)
    print(f"Workers            : {num_workers} / {psutil.cpu_count()}")
    print(f"CPU Usage          : {cpu_usage:.0f}%")
    print(f"RAM Usage          : {mem.used/(1024**3):.0f} GB / {mem.total/(1024**3):.0f} GB")
    print(f"Disk Used          : {disk_used_gb:.1f} GB")

    if stats["per_optimizer"]:
        print("\nPer Optimizer")
        print("-" * 63)
        print(f"{'Optimizer':<11} {'Completed':>9} {'Avg Time':>10} {'Avg FE/s':>11}")
        print("-" * 63)
        for name, info in sorted(stats["per_optimizer"].items()):
            comp = info['completed']
            atime = info['avg_time']
            afe = info.get('avg_fe_sec') or 0.0
            print(f"{name:<11} {comp:>9,} {atime:>8.2f} s {afe:>11,.0f}")

    # Show environment info
    env_json = db.get_metadata("environment")
    if env_json:
        env = json.loads(env_json)
        print("\nEnvironment")
        print("-" * 63)
        print(f"Hostname     : {env.get('hostname', 'N/A')}")
        print(f"Python       : {env.get('python_version', 'N/A').split()[0]}")
        print(f"NumPy        : {env.get('numpy_version', 'N/A')}")
        print(f"Git Commit   : {env.get('git_hash', 'N/A')}")

    print("=" * 63 + "\n")

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
