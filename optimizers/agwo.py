"""
agwo.py

Adaptive Grey Wolf Optimizer (AGWO).

Key modifications:
    1. Adaptive 'a' parameter based on fitness improvement rate
    2. Gaussian random walk perturbation of the alpha wolf
       to enhance exploitation capability
"""

from optimizers.base_optimizer import BaseOptimizer
from core.registry import optimizer_registry

import numpy as np


@optimizer_registry.register
class AGWO(BaseOptimizer):

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

        # Track previous best for improvement rate
        self._prev_best = self.alpha_score

    def step(self):
        """One iteration of AGWO."""

        fe_ratio = self.fe_count / self.max_fe

        # Adaptive a: based on improvement rate
        improvement_rate = abs(
            (self._prev_best - self.alpha_score)
            / (abs(self._prev_best) + 1e-30)
        )

        # When improvement is large, reduce a less (keep exploring)
        # When improvement stalls, reduce a more (exploit harder)
        a_base = 2.0 * (1.0 - fe_ratio)

        if improvement_rate > 0.01:
            # Good improvement — maintain exploration
            a = a_base * (1.0 + 0.5 * improvement_rate)
        else:
            # Stagnation — intensify exploitation
            a = a_base * (1.0 - 0.3 * (1.0 - improvement_rate))

        a = max(min(a, 2.0), 0.0)

        self._prev_best = self.alpha_score

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

                self.population[i, j] = (X1 + X2 + X3) / 3.0

            # Clip
            self.population[i] = np.clip(
                self.population[i], self.lb, self.ub
            )

            fitness = self.evaluate(self.population[i])

            self.update_leaders(self.population[i], fitness)

        # Gaussian perturbation of alpha (exploitation enhancement)
        if self.fe_count < self.max_fe:

            sigma = (self.ub - self.lb) * 0.01 * (1.0 - fe_ratio)
            perturbed_alpha = self.alpha + sigma * self.rng.randn(self.dimension)
            perturbed_alpha = np.clip(perturbed_alpha, self.lb, self.ub)

            perturbed_fitness = self.evaluate(perturbed_alpha)

            self.update_leaders(perturbed_alpha, perturbed_fitness)

        self.best_score = self.alpha_score
        self.best_position = self.alpha.copy()
