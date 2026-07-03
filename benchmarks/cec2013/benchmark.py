"""
CEC2013 Benchmark Wrapper

Wraps opfunu CEC2013 functions through the BaseProblem interface.
28 functions: F1–F28.
"""

from opfunu.cec_based.cec2013 import (
    F12013, F22013, F32013, F42013, F52013, F62013, F72013, F82013, F92013, F102013,
    F112013, F122013, F132013, F142013, F152013, F162013, F172013, F182013, F192013, F202013,
    F212013, F222013, F232013, F242013, F252013, F262013, F272013, F282013
)

from benchmarks.base_problem import BaseProblem
from core.registry import benchmark_registry


@benchmark_registry.register
class CEC2013(BaseProblem):

    FUNCTION_MAP = {
        1: F12013, 2: F22013, 3: F32013, 4: F42013, 5: F52013, 6: F62013, 7: F72013, 8: F82013, 9: F92013, 10: F102013,
        11: F112013, 12: F122013, 13: F132013, 14: F142013, 15: F152013, 16: F162013, 17: F172013, 18: F182013, 19: F192013, 20: F202013,
        21: F212013, 22: F222013, 23: F232013, 24: F242013, 25: F252013, 26: F262013, 27: F272013, 28: F282013,
    }

    def __init__(self, function: int, dimension: int):

        super().__init__(dimension)

        self.function = function

        if function not in self.FUNCTION_MAP:
            raise ValueError(
                f"CEC2013 function F{function} not available. "
                f"Valid range: 1–28."
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

        return "CEC2013"
