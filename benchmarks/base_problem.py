"""
base_problem.py

Abstract base class for all optimization benchmark problems.
Every benchmark suite (CEC2017, CEC2020, CEC2022, etc.)
must inherit from this class.

Optimizers interact only through this interface.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseProblem(ABC):
    """
    Abstract interface for optimization benchmark problems.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension

    # ----------------------------------------------------
    # Required Methods
    # ----------------------------------------------------

    @abstractmethod
    def evaluate(self, solution: np.ndarray) -> float:
        """
        Evaluate a candidate solution.

        Parameters
        ----------
        solution : np.ndarray
            Candidate solution vector of length `dimension`.

        Returns
        -------
        float
            Objective function value.
        """
        pass

    @abstractmethod
    def lower_bound(self) -> np.ndarray:
        """Return lower search bound (array or scalar)."""
        pass

    @abstractmethod
    def upper_bound(self) -> np.ndarray:
        """Return upper search bound (array or scalar)."""
        pass

    @abstractmethod
    def optimum(self) -> float:
        """Global optimum value of the function."""
        pass

    @abstractmethod
    def function_name(self) -> str:
        """
        Example: "F1", "F2"
        """
        pass

    @abstractmethod
    def benchmark_name(self) -> str:
        """
        Example: "CEC2020"
        """
        pass

    # ----------------------------------------------------
    # Optional Utility Methods
    # ----------------------------------------------------

    def bounds(self) -> Tuple:
        """
        Returns
        -------
        (lower_bound, upper_bound)
        """
        return self.lower_bound(), self.upper_bound()

    def clip(self, population: np.ndarray) -> np.ndarray:
        """
        Clip candidate solutions within search bounds.
        """

        return np.clip(
            population,
            self.lower_bound(),
            self.upper_bound()
        )

    def __repr__(self):

        return (
            f"{self.benchmark_name()} | "
            f"{self.function_name()} | "
            f"D={self.dimension}"
        )
