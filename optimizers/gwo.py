"""
gwo.py

Original Grey Wolf Optimizer (GWO).

Reference:
    Mirjalili, S., Mirjalili, S.M. & Lewis, A. (2014).
    "Grey Wolf Optimizer." Advances in Engineering Software, 69, 46-61.
"""

from optimizers.base_optimizer import BaseOptimizer
from core.registry import optimizer_registry

import numpy as np


@optimizer_registry.register
class GWO(BaseOptimizer):

    def initialize(self):
        """Initialize population and alpha/beta/delta leaders."""

        self.population = self.initialize_population()

        # Leader positions
        self.alpha = np.zeros(self.dimension)
        self.beta = np.zeros(self.dimension)
        self.delta = np.zeros(self.dimension)

        # Leader scores
        self.alpha_score = np.inf
        self.beta_score = np.inf
        self.delta_score = np.inf

        # Evaluate initial population
        for i in range(self.population_size):

            if self.fe_count >= self.max_fe:
                break

            fitness = self.evaluate(self.population[i])

            self.update_leaders(self.population[i], fitness)

        self.best_score = self.alpha_score
        self.best_position = self.alpha.copy()

    def step(self):
        """One iteration of the standard GWO."""

        # Linearly decreasing parameter: a = 2 → 0
        a = 2.0 - 2.0 * (self.fe_count / self.max_fe)

        for i in range(self.population_size):

            if self.fe_count >= self.max_fe:
                break

            for j in range(self.dimension):

                # Alpha component
                r1 = self.rng.random()
                r2 = self.rng.random()

                A1 = 2.0 * a * r1 - a
                C1 = 2.0 * r2

                D_alpha = abs(C1 * self.alpha[j] - self.population[i, j])
                X1 = self.alpha[j] - A1 * D_alpha

                # Beta component
                r1 = self.rng.random()
                r2 = self.rng.random()

                A2 = 2.0 * a * r1 - a
                C2 = 2.0 * r2

                D_beta = abs(C2 * self.beta[j] - self.population[i, j])
                X2 = self.beta[j] - A2 * D_beta

                # Delta component
                r1 = self.rng.random()
                r2 = self.rng.random()

                A3 = 2.0 * a * r1 - a
                C3 = 2.0 * r2

                D_delta = abs(C3 * self.delta[j] - self.population[i, j])
                X3 = self.delta[j] - A3 * D_delta

                # Eq. (3.7) — average of three leader-guided positions
                self.population[i, j] = (X1 + X2 + X3) / 3.0

            # Clip to bounds
            self.population[i] = self.clip(
                self.population[i].reshape(1, -1)
            ).flatten()

            # Evaluate
            fitness = self.evaluate(self.population[i])

            # Update leaders
            self.update_leaders(self.population[i], fitness)

        self.best_score = self.alpha_score
        self.best_position = self.alpha.copy()
