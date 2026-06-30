"""
obgwo.py

Opposition-Based Grey Wolf Optimizer (OBGWO).

Reference:
    Based on OBGWO_1.m — opposition-based population initialization
    and OBL with jumping rate (JR=0.1) at each iteration.
    Merge + sort to keep best N wolves.
"""

from optimizers.base_optimizer import BaseOptimizer
from core.registry import optimizer_registry

import numpy as np


@optimizer_registry.register
class OBGWO(BaseOptimizer):

    def initialize(self):
        """Initialize population with opposition-based learning."""

        N = self.population_size

        self.population = self.initialize_population()
        self.fitness = np.full(N, np.inf)

        # Evaluate initial population
        for i in range(N):
            if self.fe_count >= self.max_fe:
                break
            self.fitness[i] = self.evaluate(self.population[i])

        # Opposition-based initialization
        OP = np.zeros_like(self.population)
        OP_fitness = np.full(N, np.inf)

        for i in range(N):
            OP[i] = self.lb + self.ub - self.population[i]
            OP[i] = np.clip(OP[i], self.lb, self.ub)

            if self.fe_count >= self.max_fe:
                break
            OP_fitness[i] = self.evaluate(OP[i])

        # Merge and keep best N
        combined_pos = np.vstack([self.population, OP])
        combined_fit = np.concatenate([self.fitness, OP_fitness])

        sort_idx = np.argsort(combined_fit)
        self.population = combined_pos[sort_idx[:N]].copy()
        self.fitness = combined_fit[sort_idx[:N]].copy()

        # Set leaders
        self.alpha = self.population[0].copy()
        self.alpha_score = self.fitness[0]
        self.beta = self.population[1].copy() if N > 1 else self.alpha.copy()
        self.beta_score = self.fitness[1] if N > 1 else self.alpha_score
        self.delta = self.population[2].copy() if N > 2 else self.beta.copy()
        self.delta_score = self.fitness[2] if N > 2 else self.beta_score

        self.best_score = self.alpha_score
        self.best_position = self.alpha.copy()

    def step(self):
        """One iteration of OBGWO."""

        N = self.population_size

        a = 2.0 - 2.0 * (self.fe_count / self.max_fe)

        # Standard GWO position update
        for i in range(N):

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

            self.population[i] = np.clip(self.population[i], self.lb, self.ub)
            self.fitness[i] = self.evaluate(self.population[i])

        # OBL with jumping rate JR = 0.1
        JR = 0.1
        OP = np.zeros_like(self.population)
        OP_fitness = np.full(N, np.inf)

        for i in range(N):

            if self.fe_count >= self.max_fe:
                break

            for j in range(self.dimension):
                if self.rng.random() < JR:
                    OP[i, j] = self.ub[j] if hasattr(self.ub, '__len__') else self.ub
                    OP[i, j] += self.lb[j] if hasattr(self.lb, '__len__') else self.lb
                    OP[i, j] -= self.population[i, j]
                else:
                    OP[i, j] = self.population[i, j]

            OP[i] = np.clip(OP[i], self.lb, self.ub)
            OP_fitness[i] = self.evaluate(OP[i])

        # Merge current + opposition, keep best N
        combined_pos = np.vstack([self.population, OP])
        combined_fit = np.concatenate([self.fitness, OP_fitness])

        sort_idx = np.argsort(combined_fit)
        self.population = combined_pos[sort_idx[:N]].copy()
        self.fitness = combined_fit[sort_idx[:N]].copy()

        # Update leaders from merged+sorted pool (include previous leaders)
        all_pos = np.vstack([
            self.population,
            self.alpha.reshape(1, -1),
            self.beta.reshape(1, -1),
            self.delta.reshape(1, -1),
        ])
        all_fit = np.concatenate([
            self.fitness,
            [self.alpha_score, self.beta_score, self.delta_score],
        ])

        sort_idx = np.argsort(all_fit)

        self.alpha = all_pos[sort_idx[0]].copy()
        self.alpha_score = all_fit[sort_idx[0]]
        self.beta = all_pos[sort_idx[1]].copy()
        self.beta_score = all_fit[sort_idx[1]]
        self.delta = all_pos[sort_idx[2]].copy()
        self.delta_score = all_fit[sort_idx[2]]

        self.best_score = self.alpha_score
        self.best_position = self.alpha.copy()
