"""
modgwo.py

Modified Grey Wolf Optimizer (modGWO).

Key modifications:
    1. Exponential a decay: a = 2 * (1 - t/T)^2
    2. Fitness-weighted leader influence:
       Weights inversely proportional to leader fitness
       so that better leaders have stronger pull.
"""

from optimizers.base_optimizer import BaseOptimizer
from core.registry import optimizer_registry

import numpy as np


@optimizer_registry.register
class modGWO(BaseOptimizer):

    def initialize(self):
        """Initialize population and leaders."""

        self.population = self.initialize_population()

        self.alpha = np.zeros(self.dimension)
        self.beta = np.zeros(self.dimension)
        self.delta = np.zeros(self.dimension)

        self.alpha_score = np.inf
        self.beta_score = np.inf
        self.delta_score = np.inf

        for i in range(self.population_size):

            if self.fe_count >= self.max_fe:
                break

            fitness = self.evaluate(self.population[i])

            self.update_leaders(self.population[i], fitness)

        self.best_score = self.alpha_score
        self.best_position = self.alpha.copy()

    def step(self):
        """One iteration of modGWO."""

        fe_ratio = self.fe_count / self.max_fe

        # Quadratic decay of a
        a = 2.0 * (1.0 - fe_ratio) ** 2

        # Compute fitness-based weights for leader influence
        scores = np.array([
            self.alpha_score,
            self.beta_score,
            self.delta_score,
        ])

        # Inverse fitness (lower is better)
        # Add epsilon to avoid division by zero
        inv_scores = 1.0 / (np.abs(scores) + 1e-30)
        total = inv_scores.sum()

        w_alpha = inv_scores[0] / total
        w_beta = inv_scores[1] / total
        w_delta = inv_scores[2] / total

        for i in range(self.population_size):

            if self.fe_count >= self.max_fe:
                break

            for j in range(self.dimension):

                r1, r2 = self.rng.random(), self.rng.random()
                A1 = 2.0 * a * r1 - a
                C1 = 2.0 * r2
                D_alpha = abs(C1 * self.alpha[j] - self.population[i, j])
                X1 = self.alpha[j] - A1 * D_alpha

                r1, r2 = self.rng.random(), self.rng.random()
                A2 = 2.0 * a * r1 - a
                C2 = 2.0 * r2
                D_beta = abs(C2 * self.beta[j] - self.population[i, j])
                X2 = self.beta[j] - A2 * D_beta

                r1, r2 = self.rng.random(), self.rng.random()
                A3 = 2.0 * a * r1 - a
                C3 = 2.0 * r2
                D_delta = abs(C3 * self.delta[j] - self.population[i, j])
                X3 = self.delta[j] - A3 * D_delta

                # Weighted average instead of simple average
                self.population[i, j] = (
                    w_alpha * X1
                    + w_beta * X2
                    + w_delta * X3
                )

            # Clip and evaluate
            self.population[i] = np.clip(
                self.population[i], self.lb, self.ub
            )

            fitness = self.evaluate(self.population[i])

            self.update_leaders(self.population[i], fitness)

        self.best_score = self.alpha_score
        self.best_position = self.alpha.copy()
