"""
CEC2022 Benchmark Wrapper

Wraps opfunu CEC2022 functions through the BaseProblem interface.
12 functions: F1–F12.
"""

from opfunu.cec_based.cec2022 import (
    F12022, F22022, F32022, F42022, F52022,
    F62022, F72022, F82022, F92022, F102022,
    F112022, F122022,
)

from benchmarks.base_problem import BaseProblem
from core.registry import benchmark_registry


@benchmark_registry.register
class CEC2022(BaseProblem):

    FUNCTION_MAP = {
        1: F12022,
        2: F22022,
        3: F32022,
        4: F42022,
        5: F52022,
        6: F62022,
        7: F72022,
        8: F82022,
        9: F92022,
        10: F102022,
        11: F112022,
        12: F122022,
    }

    def __init__(self, function: int, dimension: int):

        super().__init__(dimension)

        self.function = function

        if function not in self.FUNCTION_MAP:
            raise ValueError(
                f"CEC2022 function F{function} not available. "
                f"Valid range: 1–12."
            )

        self._problem = self.FUNCTION_MAP[function](ndim=dimension)

    # ==================================================
    # Required Methods
    # ==================================================

    def evaluate(self, solution):

        return self._problem.evaluate(solution)

    def lower_bound(self):

        return self._problem.lb

    def upper_bound(self):

        return self._problem.ub

    def optimum(self):

        return self._problem.f_global

    def function_name(self):

        return f"F{self.function}"

    def benchmark_name(self):

        return "CEC2022"
