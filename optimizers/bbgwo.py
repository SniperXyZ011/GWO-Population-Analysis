"""
bbgwo.py

Bare Bones Grey Wolf Optimizer (BBGWO).

Reference:
    Wang, S. & Shi, Y.
    "A Bare Bones Grey Wolf Optimizer for Global Numerical Optimization."

Position update via Gaussian sampling:
    x_new ~ N(mu, sigma^2)
    mu    = (alpha + beta + delta) / 3
    sigma = (a / (3*sqrt(3))) * sqrt(sum of variance terms)
"""

from optimizers.base_optimizer import BaseOptimizer
from core.registry import optimizer_registry

import numpy as np


@optimizer_registry.register
class BBGWO(BaseOptimizer):

    def initialize(self):
        """Initialize population and leaders."""

        self.population = self.initialize_population()

        self.alpha = np.zeros(self.dimension)
        self.beta = np.zeros(self.dimension)
        self.delta = np.zeros(self.dimension)

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
        """One iteration of BBGWO — Gaussian sampling."""

        # Linearly decreasing a: 2 → 0
        # Using (iteration) / (max_iterations) approximation
        # based on FE ratio, matching MATLAB: a = 2*(1 - (l-1)/Max_iter)
        a = 2.0 * (1.0 - self.fe_count / self.max_fe)

        for i in range(self.population_size):

            if self.fe_count >= self.max_fe:
                break

            for j in range(self.dimension):

                # Mean (Eq. 18)
                mu = (
                    self.alpha[j]
                    + self.beta[j]
                    + self.delta[j]
                ) / 3.0

                # Variance terms for each leader (Eq. 18)
                term1 = (
                    (self.population[i, j] - self.alpha[j]) ** 2
                    + (1.0 / 3.0) * self.alpha[j] ** 2
                )
                term2 = (
                    (self.population[i, j] - self.beta[j]) ** 2
                    + (1.0 / 3.0) * self.beta[j] ** 2
                )
                term3 = (
                    (self.population[i, j] - self.delta[j]) ** 2
                    + (1.0 / 3.0) * self.delta[j] ** 2
                )

                # Standard deviation (Eq. 18)
                sigma = (
                    (a / (3.0 * np.sqrt(3.0)))
                    * np.sqrt(term1 + term2 + term3)
                )

                # Sample new position from N(mu, sigma^2) (Eq. 17)
                self.population[i, j] = mu + sigma * self.rng.randn()

            # Clip to bounds
            self.population[i] = self.clip(
                self.population[i].reshape(1, -1)
            ).flatten()

            # Evaluate
            fitness = self.evaluate(self.population[i])

            self.update_leaders(self.population[i], fitness)

        self.best_score = self.alpha_score
        self.best_position = self.alpha.copy()
