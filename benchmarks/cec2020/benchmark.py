"""
CEC2020 Benchmark Wrapper

Wraps opfunu CEC2020 functions through the BaseProblem interface.
10 functions: F1–F10.
"""

from opfunu.cec_based.cec2020 import (
    F12020, F22020, F32020, F42020, F52020,
    F62020, F72020, F82020, F92020, F102020,
)

from benchmarks.base_problem import BaseProblem
from core.registry import benchmark_registry


@benchmark_registry.register
class CEC2020(BaseProblem):

    FUNCTION_MAP = {
        1: F12020,
        2: F22020,
        3: F32020,
        4: F42020,
        5: F52020,
        6: F62020,
        7: F72020,
        8: F82020,
        9: F92020,
        10: F102020,
    }

    def __init__(self, function: int, dimension: int):

        super().__init__(dimension)

        self.function = function

        if function not in self.FUNCTION_MAP:
            raise ValueError(
                f"CEC2020 function F{function} not available. "
                f"Valid range: 1–10."
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

        return "CEC2020"
