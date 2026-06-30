"""
mengwo.py

Mutation, Evolution, and Non-linear Population Size Reduction
Grey Wolf Optimizer (MENGWO).

Reference:
    Based on MENGWO.m — three mechanisms:
    1. Mutation operator (Eq.12): exploration/exploitation switch
    2. EPD (Eq.14): elite perturbation of worst wolves
    3. NPSR (Eq.16): non-linear population size reduction
"""

from optimizers.base_optimizer import BaseOptimizer
from core.registry import optimizer_registry

import numpy as np


@optimizer_registry.register
class MENGWO(BaseOptimizer):

    def initialize(self):
        """Initialize population and leaders."""

        self.N = self.population_size  # Current pop size (may shrink)
        self.N0 = self.population_size  # Original pop size

        self.population = self.initialize_population()
        self.fitness = np.full(self.N, np.inf)

        self.alpha = np.zeros(self.dimension)
        self.beta = np.zeros(self.dimension)
        self.delta = np.zeros(self.dimension)

        self.alpha_score = np.inf
        self.beta_score = np.inf
        self.delta_score = np.inf

        # Global best archive
        self.global_best_score = np.inf
        self.global_best_pos = np.zeros(self.dimension)

        # Evaluate initial population
        for i in range(self.N):

            if self.fe_count >= self.max_fe:
                break

            self.fitness[i] = self.evaluate(self.population[i])

            if self.fitness[i] < self.global_best_score:
                self.global_best_score = self.fitness[i]
                self.global_best_pos = self.population[i].copy()

        self.best_score = self.global_best_score
        self.best_position = self.global_best_pos.copy()

    def step(self):
        """One iteration of MENGWO."""

        fe_ratio = self.fe_count / self.max_fe

        # Sort population
        sort_idx = np.argsort(self.fitness[:self.N])
        self.population[:self.N] = self.population[sort_idx]
        self.fitness[:self.N] = self.fitness[sort_idx]

        self.alpha = self.population[0].copy()
        self.alpha_score = self.fitness[0]
        self.beta = self.population[1].copy() if self.N > 1 else self.alpha.copy()
        self.delta = self.population[2].copy() if self.N > 2 else self.beta.copy()

        # Parameters
        a = 2.0 - 2.0 * fe_ratio
        B = 1.5 * fe_ratio

        # Mutation operator (Eq.12)
        new_positions = self.population[:self.N].copy()

        for i in range(self.N):

            if self.fe_count >= self.max_fe:
                break

            # Select two random distinct wolves
            rand1 = self.rng.randint(self.N)
            while rand1 == i:
                rand1 = self.rng.randint(self.N)

            rand2 = self.rng.randint(self.N)
            while rand2 == i or rand2 == rand1:
                rand2 = self.rng.randint(self.N)

            r3 = self.rng.random()

            # Independent switching coefficient
            r2_sw = self.rng.random()
            A_sw = 2.0 * a * r2_sw - a

            for j in range(self.dimension):

                # Alpha
                r1, r2 = self.rng.random(), self.rng.random()
                A1 = 2.0 * a * r2 - a
                C1 = 2.0 * r1
                D_alpha = abs(C1 * self.alpha[j] - self.population[i, j])
                X1 = self.alpha[j] - A1 * D_alpha

                # Beta
                r1, r2 = self.rng.random(), self.rng.random()
                A2 = 2.0 * a * r2 - a
                C2 = 2.0 * r1
                D_beta = abs(C2 * self.beta[j] - self.population[i, j])
                X2 = self.beta[j] - A2 * D_beta

                # Delta
                r1, r2 = self.rng.random(), self.rng.random()
                A3 = 2.0 * a * r2 - a
                C3 = 2.0 * r1
                D_delta = abs(C3 * self.delta[j] - self.population[i, j])
                X3 = self.delta[j] - A3 * D_delta

                X_mean = (X1 + X2 + X3) / 3.0

                # Eq.12: exploration vs exploitation
                if abs(A_sw) >= 1.0:
                    # Exploration
                    new_positions[i, j] = (
                        X_mean
                        + r3 * (
                            self.population[rand1, j]
                            - self.population[rand2, j]
                        )
                    )
                else:
                    # Exploitation
                    R = self.rng.random()
                    new_positions[i, j] = (
                        X_mean
                        + B * R * (self.alpha[j] - X_mean)
                    )

            # Clip
            new_positions[i] = np.clip(
                new_positions[i], self.lb, self.ub
            )

            # Evaluate
            self.fitness[i] = self.evaluate(new_positions[i])

            # Update global best
            if self.fitness[i] < self.global_best_score:
                self.global_best_score = self.fitness[i]
                self.global_best_pos = new_positions[i].copy()

        self.population[:self.N] = new_positions[:self.N]

        if self.fe_count >= self.max_fe:
            self.best_score = self.global_best_score
            self.best_position = self.global_best_pos.copy()
            return

        # Re-sort after mutation
        sort_idx = np.argsort(self.fitness[:self.N])
        self.population[:self.N] = self.population[sort_idx]
        self.fitness[:self.N] = self.fitness[sort_idx]

        self.alpha = self.population[0].copy()
        self.alpha_score = self.fitness[0]
        self.beta = self.population[1].copy() if self.N > 1 else self.alpha.copy()
        self.delta = self.population[2].copy() if self.N > 2 else self.beta.copy()

        # NPSR: Non-linear Population Size Reduction (Eq.16)
        N_next = round(
            self.N0 * (1.0 - 0.8 * (self.fe_count / self.max_fe) ** 3)
        )
        N_next = max(N_next, 4)

        if N_next < self.N:
            self.N = N_next

        # EPD: Elite Perturbation of worst wolves (Eq.14)
        k_count = max(1, int(0.25 * self.N))

        q_ini = 0.1
        q_end = 0.9
        q = q_ini + (q_end - q_ini) * fe_ratio

        for i in range(self.N - k_count, self.N):

            if self.fe_count >= self.max_fe:
                break

            r4 = self.rng.random()

            for j in range(self.dimension):

                if r4 < q:
                    # Move near Alpha
                    R2 = self.rng.random()
                    self.population[i, j] = (
                        self.population[i, j]
                        + (self.alpha[j] - R2 * self.population[i, j])
                    )
                else:
                    # Random reinitialization
                    R3 = self.rng.random()
                    lb_j = self.lb[j] if hasattr(self.lb, '__len__') else self.lb
                    ub_j = self.ub[j] if hasattr(self.ub, '__len__') else self.ub
                    self.population[i, j] = lb_j + (ub_j - lb_j) * R3

            # Clip
            self.population[i] = np.clip(
                self.population[i], self.lb, self.ub
            )

            # Evaluate
            self.fitness[i] = self.evaluate(self.population[i])

            # Update global best
            if self.fitness[i] < self.global_best_score:
                self.global_best_score = self.fitness[i]
                self.global_best_pos = self.population[i].copy()

        self.best_score = self.global_best_score
        self.best_position = self.global_best_pos.copy()
