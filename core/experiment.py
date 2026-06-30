"""
experiment.py

Contains the Experiment dataclass.
Each object represents ONE optimization experiment.
Frozen (immutable) to ensure reproducibility.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:

    # ----------------------------
    # Benchmark
    # ----------------------------

    benchmark: str

    function: int

    dimension: int

    # ----------------------------
    # Optimizer
    # ----------------------------

    optimizer: str

    # ----------------------------
    # Parameters
    # ----------------------------

    population_size: int

    max_function_evaluations: int

    # ----------------------------
    # Reproducibility
    # ----------------------------

    run: int

    seed: int

    # ----------------------------
    # Utility
    # ----------------------------

    @property
    def experiment_name(self) -> str:

        return (
            f"{self.benchmark}_"
            f"F{self.function}_"
            f"{self.optimizer}_"
            f"D{self.dimension}_"
            f"P{self.population_size}_"
            f"FE{self.max_function_evaluations}_"
            f"Run{self.run}"
        )

    def __str__(self):

        return self.experiment_name
