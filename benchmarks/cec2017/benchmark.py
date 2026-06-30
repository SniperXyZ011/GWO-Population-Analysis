"""
CEC2017 Benchmark Wrapper

Wraps opfunu CEC2017 functions through the BaseProblem interface.
29 functions: F1–F29.
"""

from opfunu.cec_based.cec2017 import (
    F12017, F22017, F32017, F42017, F52017,
    F62017, F72017, F82017, F92017, F102017,
    F112017, F122017, F132017, F142017, F152017,
    F162017, F172017, F182017, F192017, F202017,
    F212017, F222017, F232017, F242017, F252017,
    F262017, F272017, F282017, F292017,
)

from benchmarks.base_problem import BaseProblem
from core.registry import benchmark_registry


@benchmark_registry.register
class CEC2017(BaseProblem):

    FUNCTION_MAP = {
        1: F12017,
        2: F22017,
        3: F32017,
        4: F42017,
        5: F52017,
        6: F62017,
        7: F72017,
        8: F82017,
        9: F92017,
        10: F102017,
        11: F112017,
        12: F122017,
        13: F132017,
        14: F142017,
        15: F152017,
        16: F162017,
        17: F172017,
        18: F182017,
        19: F192017,
        20: F202017,
        21: F212017,
        22: F222017,
        23: F232017,
        24: F242017,
        25: F252017,
        26: F262017,
        27: F272017,
        28: F282017,
        29: F292017,
    }

    def __init__(self, function: int, dimension: int):

        super().__init__(dimension)

        self.function = function

        if function not in self.FUNCTION_MAP:
            raise ValueError(
                f"CEC2017 function F{function} not available. "
                f"Valid range: 1–29."
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

        return "CEC2017"
