"""
rwgwo.py

Random Walk Grey Wolf Optimizer (RWGWO).

Reference:
    Based on RWGWO.m — top 3 wolves use Cauchy random walk
    perturbation, remaining wolves use standard GWO update.
    Greedy selection applied.
"""

from optimizers.base_optimizer import BaseOptimizer
from core.registry import optimizer_registry

import numpy as np


@optimizer_registry.register
class RWGWO(BaseOptimizer):

    def initialize(self):
        """Initialize population, sort, and identify leaders."""

        self.population = self.initialize_population()
        self.fitness = np.full(self.population_size, np.inf)

        # Evaluate initial population
        for i in range(self.population_size):

            if self.fe_count >= self.max_fe:
                break

            self.fitness[i] = self.evaluate(self.population[i])

        # Sort by fitness
        sort_idx = np.argsort(self.fitness)
        self.population = self.population[sort_idx]
        self.fitness = self.fitness[sort_idx]

        self.alpha = self.population[0].copy()
        self.alpha_score = self.fitness[0]
        self.beta = self.population[1].copy() if self.population_size > 1 else self.alpha.copy()
        self.delta = self.population[2].copy() if self.population_size > 2 else self.beta.copy()

        self.best_score = self.alpha_score
        self.best_position = self.alpha.copy()

    def step(self):
        """One iteration of RWGWO."""

        N = self.population_size
        U_positions = np.zeros_like(self.population)

        # Parameter
        a = 2.0 - 2.0 * (self.fe_count / self.max_fe)
        par = a  # Used to scale random walk

        # Top 3 wolves: Cauchy random walk perturbation
        x0 = 0.0
        gamma_rw = 1.0

        for i in range(min(3, N)):
            for j in range(self.dimension):
                y = self.rng.random()
                random_walk = x0 + gamma_rw * np.tan(np.pi * (y - 0.5))
                U_positions[i, j] = self.population[i, j] + random_walk * par

        # Remaining wolves: standard GWO update
        for i in range(3, N):
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

                U_positions[i, j] = (X1 + X2 + X3) / 3.0

        # Clip and greedy selection
        U_positions = np.clip(U_positions, self.lb, self.ub)

        for i in range(N):

            if self.fe_count >= self.max_fe:
                break

            u_fitness = self.evaluate(U_positions[i])

            # Greedy: keep better of old vs new
            if u_fitness < self.fitness[i]:
                self.fitness[i] = u_fitness
                self.population[i] = U_positions[i].copy()

        # Sort and update leaders
        sort_idx = np.argsort(self.fitness)
        self.population = self.population[sort_idx]
        self.fitness = self.fitness[sort_idx]

        self.alpha = self.population[0].copy()
        self.alpha_score = self.fitness[0]
        self.beta = self.population[1].copy() if N > 1 else self.alpha.copy()
        self.delta = self.population[2].copy() if N > 2 else self.beta.copy()

        self.best_score = self.alpha_score
        self.best_position = self.alpha.copy()
