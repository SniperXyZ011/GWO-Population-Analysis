"""
ia_gwo.py

Improved Adaptive Grey Wolf Optimizer (IA-GWO).

Reference:
    IA-GWO combines:
    - PSO-inspired velocity update
    - Standard GWO hunting
    - Adaptive contraction toward alpha
    - IMF-weighted leadership update
"""

from optimizers.base_optimizer import BaseOptimizer
from core.registry import optimizer_registry

import numpy as np


@optimizer_registry.register
class IAGWO(BaseOptimizer):

    def __init__(self, *args, theta=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta = theta

    def imf_weight(self, t):
        return 0.6 * np.exp(-0.02 * np.exp(-0.05 * t)) + 0.3

    def initialize(self):
        """Initialize population, velocity, personal bests and leaders."""

        self.population = self.initialize_population()
        self.fitness = np.full(self.population_size, np.inf)

        # Velocity
        self.V = self.rng.uniform(
            -0.1 * (self.ub - self.lb),
             0.1 * (self.ub - self.lb),
            (self.population_size, self.dimension)
        )

        self.v_max = 0.2 * (self.ub - self.lb)

        # Evaluate initial population
        for i in range(self.population_size):

            if self.fe_count >= self.max_fe:
                break

            self.fitness[i] = self.evaluate(self.population[i])

        # Personal best
        self.pbest = self.population.copy()
        self.pbest_fit = self.fitness.copy()

        # Sort
        idx = np.argsort(self.fitness)
        self.population = self.population[idx]
        self.fitness = self.fitness[idx]

        self.alpha = self.population[0].copy()
        self.beta = self.population[1].copy()
        self.delta = self.population[2].copy()

        self.alpha_score = self.fitness[0]

        self.best_score = self.alpha_score
        self.best_position = self.alpha.copy()

        self.iteration = 1

    def step(self):
        """One iteration of IA-GWO."""

        if self.fe_count >= self.max_fe:
            return

        T = max(1, self.max_iterations)
        t = self.iteration

        a = 2.0 - 2.0 * (t / T)

        # -------------------------------------------------------
        # Phase 1: PSO velocity update
        # -------------------------------------------------------

        phi = self.rng.random()

        self.V += (
            phi * (self.alpha - self.population)
            + phi * (self.pbest - self.population)
        )

        self.V = np.clip(self.V, -self.v_max, self.v_max)

        self.population = np.clip(
            self.population + self.V,
            self.lb,
            self.ub,
        )

        # -------------------------------------------------------
        # Phase 2: Standard GWO update
        # -------------------------------------------------------

        r1 = self.rng.random((self.population_size, self.dimension))
        r2 = self.rng.random((self.population_size, self.dimension))
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        X1 = self.alpha - A1 * np.abs(C1 * self.alpha - self.population)

        r1 = self.rng.random((self.population_size, self.dimension))
        r2 = self.rng.random((self.population_size, self.dimension))
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        X2 = self.beta - A2 * np.abs(C2 * self.beta - self.population)

        r1 = self.rng.random((self.population_size, self.dimension))
        r2 = self.rng.random((self.population_size, self.dimension))
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        X3 = self.delta - A3 * np.abs(C3 * self.delta - self.population)

        self.population = np.clip(
            (X1 + X2 + X3) / 3.0,
            self.lb,
            self.ub,
        )

        # -------------------------------------------------------
        # Evaluate
        # -------------------------------------------------------

        for i in range(self.population_size):

            if self.fe_count >= self.max_fe:
                break

            self.fitness[i] = self.evaluate(self.population[i])

        favg = np.mean(self.fitness)

        # -------------------------------------------------------
        # Phase 3: Adaptive contraction
        # -------------------------------------------------------

        phi = 1.0 / (
            1.0 + np.exp(-self.theta * self.fitness / (favg + 1e-30))
        )

        self.population = (
            self.alpha
            + phi[:, None] * (self.population - self.alpha)
        )

        self.population = np.clip(
            self.population,
            self.lb,
            self.ub,
        )

        # -------------------------------------------------------
        # Evaluate again
        # -------------------------------------------------------

        for i in range(self.population_size):

            if self.fe_count >= self.max_fe:
                break

            self.fitness[i] = self.evaluate(self.population[i])

        improved = self.fitness < self.pbest_fit
        self.pbest[improved] = self.population[improved]
        self.pbest_fit[improved] = self.fitness[improved]

        idx = np.argsort(self.fitness)

        self.population = self.population[idx]
        self.fitness = self.fitness[idx]

        self.alpha = self.population[0].copy()
        self.beta = self.population[1].copy()
        self.delta = self.population[2].copy()

        self.alpha_score = self.fitness[0]

        # -------------------------------------------------------
        # Phase 4: IMF update
        # -------------------------------------------------------

        w = self.imf_weight(t)

        r = self.rng.random((self.population_size, self.dimension))
        A1 = 2 * a * r - a
        X1 = self.alpha - w * A1 * np.abs(self.alpha - self.population)

        r = self.rng.random((self.population_size, self.dimension))
        A2 = 2 * a * r - a
        X2 = self.beta - w * A2 * np.abs(self.beta - self.population)

        r = self.rng.random((self.population_size, self.dimension))
        A3 = 2 * a * r - a
        X3 = self.delta - w * A3 * np.abs(self.delta - self.population)

        self.population = np.clip(
            (X1 + X2 + X3) / 3.0,
            self.lb,
            self.ub,
        )

        # -------------------------------------------------------
        # Final evaluation
        # -------------------------------------------------------

        for i in range(self.population_size):

            if self.fe_count >= self.max_fe:
                break

            self.fitness[i] = self.evaluate(self.population[i])

        idx = np.argmin(self.fitness)

        if self.fitness[idx] < self.alpha_score:
            self.alpha_score = self.fitness[idx]
            self.alpha = self.population[idx].copy()

        # Final sorting
        idx = np.argsort(self.fitness)

        self.population = self.population[idx]
        self.fitness = self.fitness[idx]

        self.alpha = self.population[0].copy()
        self.beta = self.population[1].copy()
        self.delta = self.population[2].copy()

        self.best_score = self.fitness[0]
        self.best_position = self.population[0].copy()

        self.iteration += 1