"""
test_enhanced_logging.py

Tests for enhanced logging in utils/logger.py.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, log_experiment_metric, _JsonLinesHandler


class TestSetupLogger:

    def test_returns_logger(self, tmp_path, monkeypatch):
        import configs.config as config
        monkeypatch.setattr(config, "LOG_DIR", tmp_path)

        # Clear any existing handlers
        import logging
        logger = logging.getLogger("test_setup_logger_1")
        logger.handlers.clear()

        result = setup_logger(
            name="test_setup_logger_1",
            log_file="test.log",
            enable_jsonl=False,
        )
        assert result.name == "test_setup_logger_1"

    def test_creates_log_file(self, tmp_path, monkeypatch):
        import utils.logger as logger_module
        monkeypatch.setattr(logger_module, "LOG_DIR", str(tmp_path))

        import logging
        logger = logging.getLogger("test_setup_logger_2")
        logger.handlers.clear()

        setup_logger(
            name="test_setup_logger_2",
            log_file="campaign_test.log",
            enable_jsonl=False,
        )

        assert (tmp_path / "campaign_test.log").exists()

    def test_no_duplicate_handlers(self, tmp_path, monkeypatch):
        import configs.config as config
        monkeypatch.setattr(config, "LOG_DIR", tmp_path)

        import logging
        logger = logging.getLogger("test_setup_logger_3")
        logger.handlers.clear()

        setup_logger(name="test_setup_logger_3", log_file="dup.log", enable_jsonl=False)
        handler_count = len(logger.handlers)

        setup_logger(name="test_setup_logger_3", log_file="dup.log", enable_jsonl=False)
        assert len(logger.handlers) == handler_count


class TestJsonLinesHandler:

    def test_creates_jsonl_file(self, tmp_path):
        filepath = tmp_path / "test.jsonl"
        handler = _JsonLinesHandler(filepath)

        import logging
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="Test message", args=(), exc_info=None,
        )
        handler.emit(record)

        assert filepath.exists()

    def test_jsonl_is_valid_json(self, tmp_path):
        filepath = tmp_path / "test.jsonl"
        handler = _JsonLinesHandler(filepath)

        import logging
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="Test message", args=(), exc_info=None,
        )
        handler.emit(record)

        with open(filepath) as f:
            line = f.readline()
            data = json.loads(line)

            assert data["level"] == "INFO"
            assert data["message"] == "Test message"
            assert "timestamp" in data
            assert "pid" in data

    def test_jsonl_multiple_entries(self, tmp_path):
        filepath = tmp_path / "test.jsonl"
        handler = _JsonLinesHandler(filepath)

        import logging
        for i in range(5):
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="",
                lineno=0, msg=f"Message {i}", args=(), exc_info=None,
            )
            handler.emit(record)

        with open(filepath) as f:
            lines = f.readlines()
            assert len(lines) == 5

            for line in lines:
                data = json.loads(line)
                assert "message" in data

    def test_jsonl_extra_fields(self, tmp_path):
        filepath = tmp_path / "test.jsonl"
        handler = _JsonLinesHandler(filepath)

        import logging
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="Metric", args=(), exc_info=None,
        )
        record.experiment = "CEC2020_F1_GWO"
        record.metric = "fe_per_second"
        record.value = 12345.6

        handler.emit(record)

        with open(filepath) as f:
            data = json.loads(f.readline())
            assert data["experiment"] == "CEC2020_F1_GWO"
            assert data["metric"] == "fe_per_second"
            assert data["value"] == 12345.6


class TestLogExperimentMetric:

    def test_log_metric(self, tmp_path, monkeypatch):
        import configs.config as config
        monkeypatch.setattr(config, "LOG_DIR", tmp_path)

        import logging
        logger = logging.getLogger("test_metric_logger")
        logger.handlers.clear()

        jsonl_handler = _JsonLinesHandler(tmp_path / "metrics.jsonl")
        logger.addHandler(jsonl_handler)
        logger.setLevel(logging.INFO)

        log_experiment_metric(
            logger=logger,
            experiment_name="CEC2020_F1_GWO_D10",
            metric="best_score",
            value=42.5,
        )

        with open(tmp_path / "metrics.jsonl") as f:
            data = json.loads(f.readline())
            assert data["experiment"] == "CEC2020_F1_GWO_D10"
            assert data["metric"] == "best_score"
            assert data["value"] == 42.5
