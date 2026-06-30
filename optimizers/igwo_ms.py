"""
igwo_ms.py

Improved Grey Wolf Optimizer with Multi-Strategy (IGWO_MS).

Reference:
    Based on IGWO.m — three phases per iteration:
    1. ACP (Adaptive Chaotic Perturbation)
    2. Standard GWO exploitation
    3. LOBL (Linear Opposition-Based Learning)

Note: Uses 2 FEs per wolf per iteration (standard + LOBL).
"""

from optimizers.base_optimizer import BaseOptimizer
from core.registry import optimizer_registry

import numpy as np


@optimizer_registry.register
class IGWO_MS(BaseOptimizer):

    def __init__(self, problem, population_size, max_function_evaluations, seed=None):
        super().__init__(problem, population_size, max_function_evaluations, seed)
        self.k_lobl = 1.2  # LOBL parameter

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
        """One iteration of IGWO-MS: ACP + GWO + LOBL."""

        N = self.population_size
        max_iter_approx = self.max_fe / max(N * 2, 1)  # ~2 FEs per wolf
        t = self.iteration + 1

        # Parameter a
        a = 2.0 - (t - 1) * (2.0 / max(max_iter_approx, 1))
        a = max(a, 0.0)

        # ============================================
        # STEP 1: ACP (Adaptive Chaotic Perturbation)
        # ============================================

        X_mean = np.mean(self.population[:N], axis=0)

        for i in range(N):

            r3 = self.rng.random()
            r4 = self.rng.random()

            t_ratio = (max_iter_approx - t + 1) / max(max_iter_approx, 1)
            gamma = 2.0 * np.exp(r4 * t_ratio) * np.sin(2.0 * np.pi * r4)

            if r3 < 0.5:
                self.population[i] = (
                    r3 * X_mean
                    + gamma * (self.alpha - self.population[i])
                )
            else:
                self.population[i] = (
                    r3 * X_mean
                    + gamma * (
                        (self.beta + self.delta) / 2.0
                        - self.population[i]
                    )
                )

        # ============================================
        # STEP 2: Standard GWO Exploitation
        # ============================================

        for i in range(N):

            r1 = self.rng.random(self.dimension)
            r2 = self.rng.random(self.dimension)
            A1 = 2.0 * a * r1 - a
            C1 = 2.0 * r2
            D_alpha = np.abs(C1 * self.alpha - self.population[i])
            X1 = self.alpha - A1 * D_alpha

            r1 = self.rng.random(self.dimension)
            r2 = self.rng.random(self.dimension)
            A2 = 2.0 * a * r1 - a
            C2 = 2.0 * r2
            D_beta = np.abs(C2 * self.beta - self.population[i])
            X2 = self.beta - A2 * D_beta

            r1 = self.rng.random(self.dimension)
            r2 = self.rng.random(self.dimension)
            A3 = 2.0 * a * r1 - a
            C3 = 2.0 * r2
            D_delta = np.abs(C3 * self.delta - self.population[i])
            X3 = self.delta - A3 * D_delta

            self.population[i] = (X1 + X2 + X3) / 3.0

        # ============================================
        # STEP 3: LOBL + Evaluate
        # ============================================

        k = self.k_lobl

        for i in range(N):

            if self.fe_count >= self.max_fe:
                break

            self.population[i] = np.clip(
                self.population[i], self.lb, self.ub
            )

            curr_fit = self.evaluate(self.population[i])

            if self.fe_count >= self.max_fe:
                self.update_leaders(self.population[i], curr_fit)
                break

            # LOBL
            X_LOBL = (
                (self.lb + self.ub) / 2.0
                + (self.lb + self.ub) / (2.0 * k)
                - self.population[i] / k
            )
            X_LOBL = np.clip(X_LOBL, self.lb, self.ub)

            lobl_fit = self.evaluate(X_LOBL)

            if lobl_fit < curr_fit:
                self.population[i] = X_LOBL.copy()
                final_fit = lobl_fit
            else:
                final_fit = curr_fit

            # Update leaders immediately (as in MATLAB)
            self.update_leaders(self.population[i], final_fit)

        self.best_score = self.alpha_score
        self.best_position = self.alpha.copy()
