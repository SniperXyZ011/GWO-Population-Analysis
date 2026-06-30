"""
main.py

Entry point for the GWO Population Analysis Framework.
Orchestrates the full experimental pipeline.

Usage:
    python main.py
    python main.py --benchmark CEC2020 --optimizer GWO --dimension 10
    python main.py --optimizer GWO BBGWO --dimension 10 30 --population 10 50
"""

import argparse
import sys
import time

# --- Register all benchmarks and optimizers ---
import benchmarks.cec2017  # noqa: F401
import benchmarks.cec2020  # noqa: F401
import benchmarks.cec2022  # noqa: F401
import optimizers           # noqa: F401

from core.parameter_grid import ParameterGrid
from core.runner import ExperimentRunner
from core.registry import optimizer_registry, benchmark_registry
from utils.logger import setup_logger


def parse_args():
    """Parse command-line arguments for experiment filtering."""

    parser = argparse.ArgumentParser(
        description="GWO Population Analysis Framework"
    )

    parser.add_argument(
        "--benchmark",
        nargs="*",
        default=None,
        help="Benchmark suites to run (e.g., CEC2017 CEC2020).",
    )

    parser.add_argument(
        "--optimizer",
        nargs="*",
        default=None,
        help="Optimizers to run (e.g., GWO BBGWO REGWO).",
    )

    parser.add_argument(
        "--dimension",
        nargs="*",
        type=int,
        default=None,
        help="Dimensions to run (e.g., 10 30 50).",
    )

    parser.add_argument(
        "--population",
        nargs="*",
        type=int,
        default=None,
        help="Population sizes to run (e.g., 10 50 100).",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Number of independent runs.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    logger = setup_logger()

    # --- Display registered components ---
    logger.info(
        f"Registered Optimizers: {optimizer_registry.list()}"
    )
    logger.info(
        f"Registered Benchmarks: {benchmark_registry.list()}"
    )

    # --- Build parameter grid ---
    grid = ParameterGrid(
        benchmarks=args.benchmark,
        optimizers=args.optimizer,
        dimensions=args.dimension,
        populations=args.population,
        runs=args.runs,
    )

    total = len(grid)
    logger.info(f"Total experiments: {total}")
    logger.info(f"Grid: {grid}")

    if total == 0:
        logger.warning("No experiments to run. Check your configuration.")
        return

    # --- Run experiments ---
    runner = ExperimentRunner()

    completed = 0
    skipped = 0
    failed = 0

    start_time = time.time()

    for experiment in grid.generate():

        try:
            result = runner.run(experiment)

            if result is None:
                skipped += 1
            else:
                completed += 1

        except Exception as e:
            failed += 1
            logger.error(
                f"FAILED: {experiment} | Error: {e}",
                exc_info=True,
            )

        # Progress report every 100 experiments
        done = completed + skipped + failed
        if done % 100 == 0 and done > 0:
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0

            logger.info(
                f"Progress: {done}/{total} "
                f"({done/total*100:.1f}%) | "
                f"Completed={completed} "
                f"Skipped={skipped} "
                f"Failed={failed} | "
                f"ETA={eta/3600:.1f}h"
            )

    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"  Total:     {total}")
    logger.info(f"  Completed: {completed}")
    logger.info(f"  Skipped:   {skipped}")
    logger.info(f"  Failed:    {failed}")
    logger.info(f"  Time:      {elapsed/3600:.2f} hours")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
