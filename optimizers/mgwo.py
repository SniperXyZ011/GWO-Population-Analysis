"""
mgwo.py

Modified Grey Wolf Optimizer (MGWO).

Key modifications over original GWO:
    1. Exponential decay of 'a': a = 2 * exp(-4 * (t/T)^2)
       (provides slower decay early, faster late)
    2. Lévy flight for exploration when |A| > 1

Reference:
    Commonly cited "Modified GWO" with Lévy flight enhancement.
"""

from optimizers.base_optimizer import BaseOptimizer
from core.registry import optimizer_registry

import numpy as np
from scipy.special import gamma as gamma_func


@optimizer_registry.register
class MGWO(BaseOptimizer):

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

    def _levy_flight(self, dim, beta_lf=1.5):
        """Generate Lévy flight step vector."""

        sigma_u = (
            gamma_func(1 + beta_lf)
            * np.sin(np.pi * beta_lf / 2.0)
            / (
                gamma_func((1 + beta_lf) / 2.0)
                * beta_lf
                * 2.0 ** ((beta_lf - 1.0) / 2.0)
            )
        ) ** (1.0 / beta_lf)

        u = self.rng.randn(dim) * sigma_u
        v = self.rng.randn(dim)

        step = u / (np.abs(v) ** (1.0 / beta_lf))

        return step

    def step(self):
        """One iteration of MGWO."""

        fe_ratio = self.fe_count / self.max_fe

        # Exponential decay of a
        a = 2.0 * np.exp(-4.0 * fe_ratio ** 2)

        for i in range(self.population_size):

            if self.fe_count >= self.max_fe:
                break

            new_pos = np.zeros(self.dimension)

            # Check if we should use Lévy flight
            r_sw = self.rng.random()
            A_sw = 2.0 * a * r_sw - a

            if abs(A_sw) >= 1.0:
                # Exploration: Lévy flight
                levy = self._levy_flight(self.dimension)
                new_pos = (
                    self.population[i]
                    + 0.01 * levy * (self.population[i] - self.alpha)
                )
            else:
                # Standard GWO exploitation with modified a
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

                    new_pos[j] = (X1 + X2 + X3) / 3.0

            # Clip and evaluate
            new_pos = np.clip(new_pos, self.lb, self.ub)

            fitness = self.evaluate(new_pos)

            # Greedy update
            self.population[i] = new_pos

            self.update_leaders(new_pos, fitness)

        self.best_score = self.alpha_score
        self.best_position = self.alpha.copy()
