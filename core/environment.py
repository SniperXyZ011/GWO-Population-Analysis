"""
environment.py

Captures reproducibility metadata about the execution environment.
Stored once per campaign in the checkpoint database.
"""

import platform
import socket
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class EnvironmentInfo:
    """Immutable snapshot of the execution environment."""

    hostname: str
    python_version: str
    os_info: str
    cpu_count: int
    numpy_version: str
    scipy_version: str
    opfunu_version: str
    git_hash: Optional[str]
    timestamp: str

    @classmethod
    def capture(cls) -> "EnvironmentInfo":
        """
        Auto-capture all environment metadata at the moment
        this method is called.

        Returns
        -------
        EnvironmentInfo
            Frozen dataclass with all fields populated.
        """

        return cls(
            hostname=socket.gethostname(),
            python_version=sys.version,
            os_info=f"{platform.system()} {platform.release()} ({platform.machine()})",
            cpu_count=_get_cpu_count(),
            numpy_version=_get_package_version("numpy"),
            scipy_version=_get_package_version("scipy"),
            opfunu_version=_get_package_version("opfunu"),
            git_hash=_get_git_hash(),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return asdict(self)


# =====================================================
# Internal helpers
# =====================================================

def _get_cpu_count() -> int:
    """Get the number of logical CPUs available."""

    import os

    try:
        # Prefer os.sched_getaffinity (respects cgroups / taskset)
        return len(os.sched_getaffinity(0))
    except AttributeError:
        # Windows / macOS fallback
        return os.cpu_count() or 1


def _get_package_version(package_name: str) -> str:
    """
    Get the installed version of a Python package.

    Returns 'not installed' if the package is missing.
    """

    try:
        from importlib.metadata import version
        return version(package_name)
    except Exception:
        return "not installed"


def _get_git_hash() -> Optional[str]:
    """
    Get the current git commit hash if inside a repository.

    Returns None if git is unavailable or not in a repo.
    """

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            return result.stdout.strip()

    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None
