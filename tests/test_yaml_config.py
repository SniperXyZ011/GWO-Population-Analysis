"""
test_yaml_config.py

Tests for configs/yaml_config.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.yaml_config import ExperimentConfig


class TestExperimentConfig:

    def test_default_config(self):
        config = ExperimentConfig()
        assert len(config.benchmarks) > 0
        assert len(config.optimizers) > 0
        assert len(config.dimensions) > 0
        assert len(config.population_sizes) > 0
        assert config.runs > 0

    def test_custom_config(self):
        config = ExperimentConfig(
            name="test",
            benchmarks=["CEC2020"],
            optimizers=["GWO"],
            dimensions=[10],
            population_sizes=[30],
            runs=5,
            workers=4,
        )
        assert config.name == "test"
        assert config.benchmarks == ["CEC2020"]
        assert config.optimizers == ["GWO"]
        assert config.dimensions == [10]
        assert config.population_sizes == [30]
        assert config.runs == 5
        assert config.workers == 4

    def test_to_dict(self):
        config = ExperimentConfig(
            benchmarks=["CEC2020"],
            optimizers=["GWO"],
        )
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["benchmarks"] == ["CEC2020"]
        assert d["optimizers"] == ["GWO"]
        assert "workers" in d
        assert "results_dir" in d

    def test_repr(self):
        config = ExperimentConfig(
            benchmarks=["CEC2020"],
            optimizers=["GWO"],
        )
        r = repr(config)
        assert "ExperimentConfig" in r

    def test_workers_default_zero(self):
        config = ExperimentConfig()
        assert config.workers == 0  # auto-detect

    def test_checkpoint_db_default(self):
        config = ExperimentConfig()
        assert config.checkpoint_db.name == "checkpoint.db"


class TestYAMLLoading:

    def test_from_yaml(self, tmp_path):
        """Test loading from a YAML file."""
        yaml_content = """
experiment:
  name: "test_campaign"

benchmarks:
  - CEC2020

optimizers:
  - GWO
  - BBGWO

dimensions: [10, 30]

population_sizes: [20, 50]

runs: 5
max_fe_multiplier: 5000

execution:
  workers: 8
  retry_failed: 2

output:
  results_dir: "./test_results"
  checkpoint_db: "./test.db"
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        config = ExperimentConfig.from_yaml(str(yaml_file))

        assert config.name == "test_campaign"
        assert config.benchmarks == ["CEC2020"]
        assert config.optimizers == ["GWO", "BBGWO"]
        assert config.dimensions == [10, 30]
        assert config.population_sizes == [20, 50]
        assert config.runs == 5
        assert config.max_fe_multiplier == 5000
        assert config.workers == 8
        assert config.max_retries == 2

    def test_from_yaml_missing_fields_fallback(self, tmp_path):
        """Missing YAML fields should fall back to Python defaults."""
        yaml_content = """
benchmarks:
  - CEC2020
"""
        yaml_file = tmp_path / "minimal.yaml"
        yaml_file.write_text(yaml_content)

        config = ExperimentConfig.from_yaml(str(yaml_file))

        assert config.benchmarks == ["CEC2020"]
        # Other fields should be defaults
        assert len(config.optimizers) > 0
        assert len(config.dimensions) > 0
        assert config.runs > 0

    def test_from_yaml_empty_file(self, tmp_path):
        """Empty YAML should use all defaults."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        config = ExperimentConfig.from_yaml(str(yaml_file))
        assert len(config.benchmarks) > 0

    def test_from_yaml_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ExperimentConfig.from_yaml("/nonexistent/path.yaml")


class TestCLIArgs:

    def test_from_cli_args_no_config(self):
        """CLI args without a config file."""

        class MockArgs:
            config = None
            benchmark = ["CEC2020"]
            optimizer = ["GWO"]
            dimension = [10, 30]
            population = [50]
            runs = 5
            workers = 4
            overwrite = False

        config = ExperimentConfig.from_cli_args(MockArgs())

        assert config.benchmarks == ["CEC2020"]
        assert config.optimizers == ["GWO"]
        assert config.dimensions == [10, 30]
        assert config.population_sizes == [50]
        assert config.runs == 5
        assert config.workers == 4

    def test_from_cli_args_with_yaml(self, tmp_path):
        """CLI args override YAML values."""
        yaml_content = """
benchmarks:
  - CEC2020

optimizers:
  - GWO

dimensions: [10]
runs: 30
"""
        yaml_file = tmp_path / "base.yaml"
        yaml_file.write_text(yaml_content)

        class MockArgs:
            config = str(yaml_file)
            benchmark = None       # Use YAML
            optimizer = ["BBGWO"]  # Override YAML
            dimension = None       # Use YAML
            population = None
            runs = None
            workers = 16           # Override
            overwrite = False

        config = ExperimentConfig.from_cli_args(MockArgs())

        assert config.benchmarks == ["CEC2020"]  # From YAML
        assert config.optimizers == ["BBGWO"]     # CLI override
        assert config.dimensions == [10]           # From YAML
        assert config.workers == 16                # CLI override

    def test_from_cli_args_overwrite_flag(self):
        class MockArgs:
            config = None
            benchmark = None
            optimizer = None
            dimension = None
            population = None
            runs = None
            workers = None
            overwrite = True

        config = ExperimentConfig.from_cli_args(MockArgs())
        assert config.overwrite is True
