"""
yaml_config.py

YAML configuration loader.

Loads a declarative experiment configuration from a YAML file
and overrides Python defaults from config.py.

Falls back gracefully to config.py defaults when:
    - No YAML file is specified
    - A field is missing from the YAML file
    - PyYAML is not installed
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from configs.config import (
    BENCHMARKS,
    BENCHMARK_FUNCTIONS,
    OPTIMIZERS,
    DIMENSIONS,
    POPULATION_SIZES,
    RUNS,
    MAX_FE_MULTIPLIER,
    RESULTS_DIR,
    LOG_DIR,
)

logger = logging.getLogger(__name__)


class ExperimentConfig:
    """
    Merged configuration from YAML overrides + Python defaults.

    Attributes
    ----------
    name : str
        Campaign name.
    benchmarks : list of str
    optimizers : list of str
    dimensions : list of int
    population_sizes : list of int
    runs : int
    max_fe_multiplier : int
    workers : int
        0 = auto-detect.
    max_retries : int
    results_dir : Path
    logs_dir : Path
    checkpoint_db : Path
    save_convergence : bool
    overwrite : bool
    """

    def __init__(
        self,
        name: str = "default_campaign",
        benchmarks: Optional[List[str]] = None,
        optimizers: Optional[List[str]] = None,
        dimensions: Optional[List[int]] = None,
        population_sizes: Optional[List[int]] = None,
        runs: Optional[int] = None,
        max_fe_multiplier: Optional[int] = None,
        workers: int = 0,
        max_retries: int = 0,
        results_dir: Optional[str] = None,
        logs_dir: Optional[str] = None,
        checkpoint_db: Optional[str] = None,
        save_convergence: bool = True,
        overwrite: bool = False,
    ):

        self.name = name

        self.benchmarks = benchmarks or list(BENCHMARKS)
        self.optimizers = optimizers or list(OPTIMIZERS)
        self.dimensions = dimensions or list(DIMENSIONS)
        self.population_sizes = population_sizes or list(POPULATION_SIZES)
        self.runs = runs or RUNS
        self.max_fe_multiplier = max_fe_multiplier or MAX_FE_MULTIPLIER

        self.workers = workers
        self.max_retries = max_retries

        self.results_dir = Path(results_dir) if results_dir else RESULTS_DIR
        self.logs_dir = Path(logs_dir) if logs_dir else LOG_DIR
        self.checkpoint_db = (
            Path(checkpoint_db) if checkpoint_db
            else self.results_dir.parent / "checkpoint.db"
        )

        self.save_convergence = save_convergence
        self.overwrite = overwrite

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """
        Load configuration from a YAML file.

        Missing fields fall back to Python defaults.

        Parameters
        ----------
        yaml_path : str
            Path to the YAML configuration file.

        Returns
        -------
        ExperimentConfig

        Raises
        ------
        ImportError
            If PyYAML is not installed.
        FileNotFoundError
            If the YAML file doesn't exist.
        """

        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config files. "
                "Install with: pip install pyyaml"
            )

        path = Path(yaml_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {yaml_path}"
            )

        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            logger.warning(f"Empty YAML file: {yaml_path}. Using defaults.")
            return cls()

        # Extract nested sections
        experiment = raw.get("experiment", {})
        execution = raw.get("execution", {})
        output = raw.get("output", {})

        return cls(
            name=experiment.get("name", "default_campaign"),
            benchmarks=raw.get("benchmarks"),
            optimizers=raw.get("optimizers"),
            dimensions=raw.get("dimensions"),
            population_sizes=raw.get("population_sizes"),
            runs=raw.get("runs"),
            max_fe_multiplier=raw.get("max_fe_multiplier"),
            workers=execution.get("workers", 0),
            max_retries=execution.get("retry_failed", 0),
            results_dir=output.get("results_dir"),
            logs_dir=output.get("logs_dir"),
            checkpoint_db=output.get("checkpoint_db"),
            save_convergence=output.get("save_convergence", True),
            overwrite=output.get("overwrite", False),
        )

    @classmethod
    def from_cli_args(cls, args) -> "ExperimentConfig":
        """
        Create config from argparse Namespace.

        CLI args override YAML if a --config file is specified.

        Parameters
        ----------
        args : argparse.Namespace

        Returns
        -------
        ExperimentConfig
        """

        # Start from YAML if provided
        config_path = getattr(args, "config", None)
        if config_path:
            config = cls.from_yaml(config_path)
        else:
            config = cls()

        # CLI overrides
        if getattr(args, "benchmark", None):
            config.benchmarks = args.benchmark

        if getattr(args, "optimizer", None):
            config.optimizers = args.optimizer

        if getattr(args, "dimension", None):
            config.dimensions = args.dimension

        if getattr(args, "population", None):
            config.population_sizes = args.population

        if getattr(args, "runs", None):
            config.runs = args.runs

        if getattr(args, "workers", None) is not None:
            config.workers = args.workers

        if getattr(args, "overwrite", False):
            config.overwrite = True

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary (for logging/storage)."""

        return {
            "name": self.name,
            "benchmarks": self.benchmarks,
            "optimizers": self.optimizers,
            "dimensions": self.dimensions,
            "population_sizes": self.population_sizes,
            "runs": self.runs,
            "max_fe_multiplier": self.max_fe_multiplier,
            "workers": self.workers,
            "max_retries": self.max_retries,
            "results_dir": str(self.results_dir),
            "logs_dir": str(self.logs_dir),
            "checkpoint_db": str(self.checkpoint_db),
            "save_convergence": self.save_convergence,
            "overwrite": self.overwrite,
        }

    def __repr__(self) -> str:
        return (
            f"ExperimentConfig("
            f"name={self.name!r}, "
            f"benchmarks={len(self.benchmarks)}, "
            f"optimizers={len(self.optimizers)}, "
            f"dims={self.dimensions}, "
            f"pops={len(self.population_sizes)}, "
            f"runs={self.runs}, "
            f"workers={self.workers})"
        )
