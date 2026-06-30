"""
optimizer_factory.py

Factory for creating optimizer instances from the registry.
"""

from core.registry import optimizer_registry


class OptimizerFactory:

    @staticmethod
    def create(experiment, problem):
        """
        Create an optimizer instance for the given experiment.

        Parameters
        ----------
        experiment : Experiment
        problem : BaseProblem

        Returns
        -------
        BaseOptimizer subclass instance
        """

        optimizer_cls = optimizer_registry.get(
            experiment.optimizer
        )

        optimizer = optimizer_cls(
            problem=problem,
            population_size=experiment.population_size,
            max_function_evaluations=experiment.max_function_evaluations,
            seed=experiment.seed,
        )

        return optimizer
