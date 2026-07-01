"""
test_environment.py

Tests for core/environment.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.environment import EnvironmentInfo, _get_cpu_count, _get_package_version


class TestEnvironmentInfo:

    def test_capture_returns_environment_info(self):
        env = EnvironmentInfo.capture()
        assert isinstance(env, EnvironmentInfo)

    def test_hostname_not_empty(self):
        env = EnvironmentInfo.capture()
        assert len(env.hostname) > 0

    def test_python_version_not_empty(self):
        env = EnvironmentInfo.capture()
        assert len(env.python_version) > 0
        assert "." in env.python_version  # e.g., "3.11.4"

    def test_cpu_count_positive(self):
        env = EnvironmentInfo.capture()
        assert env.cpu_count > 0

    def test_numpy_version_not_empty(self):
        env = EnvironmentInfo.capture()
        assert env.numpy_version != "not installed"

    def test_scipy_version_not_empty(self):
        env = EnvironmentInfo.capture()
        assert env.scipy_version != "not installed"

    def test_timestamp_is_iso_format(self):
        env = EnvironmentInfo.capture()
        assert "T" in env.timestamp
        assert ":" in env.timestamp

    def test_os_info_not_empty(self):
        env = EnvironmentInfo.capture()
        assert len(env.os_info) > 0

    def test_to_dict_returns_dict(self):
        env = EnvironmentInfo.capture()
        d = env.to_dict()
        assert isinstance(d, dict)
        assert "hostname" in d
        assert "python_version" in d
        assert "cpu_count" in d

    def test_frozen(self):
        env = EnvironmentInfo.capture()
        with pytest.raises(AttributeError):
            env.hostname = "something_else"

    def test_git_hash_is_string_or_none(self):
        env = EnvironmentInfo.capture()
        assert env.git_hash is None or isinstance(env.git_hash, str)


class TestHelpers:

    def test_get_cpu_count_positive(self):
        assert _get_cpu_count() > 0

    def test_get_package_version_numpy(self):
        version = _get_package_version("numpy")
        assert version != "not installed"
        assert "." in version

    def test_get_package_version_nonexistent(self):
        version = _get_package_version("this_package_does_not_exist_xyz")
        assert version == "not installed"
