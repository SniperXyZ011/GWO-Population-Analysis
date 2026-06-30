"""
regwo.py

Refraction Learning based Enhanced Grey Wolf Optimizer (REGWO).

Reference:
    Based on REGWO.m — dual strategy (standard GWO / equilibrium-inspired)
    plus refraction learning with adaptive k parameter.
"""

from optimizers.base_optimizer import BaseOptimizer
from core.registry import optimizer_registry

import numpy as np


@optimizer_registry.register
class REGWO(BaseOptimizer):

    def initialize(self):
        """Initialize population and leaders."""

        self.population = self.initialize_population()
        self.fitness = np.full(self.population_size, np.inf)

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

            self.fitness[i] = self.evaluate(self.population[i])

            self.update_leaders(self.population[i], self.fitness[i])

        self.best_score = self.alpha_score
        self.best_position = self.alpha.copy()

        # History for stagnation detection
        self.best_history = []

    def step(self):
        """One iteration of REGWO."""

        N = self.population_size
        fe_ratio = self.fe_count / self.max_fe

        a = 2.0 - 2.0 * fe_ratio

        # Adaptive k parameter
        k_max = 1.0
        k_min = 0.0
        k = k_max - (k_max - k_min) * fe_ratio

        # Stagnation-based adaptive k
        if len(self.best_history) > 10:
            prev = self.best_history[-10]
            eta = abs(
                (self.alpha_score - prev) / (prev + 1e-30)
            )
            mu_k = (k - k_min) / (k_max - k_min + 1e-30)
            gamma = 0.05

            tau1 = self.rng.random()
            tau2 = self.rng.random()

            max_iter_approx = self.max_fe / max(N, 1)
            iter_approx = self.iteration
            t_ratio = iter_approx / max(max_iter_approx, 1)

            if (mu_k > 0.5 and eta > gamma) or (mu_k <= 0.5 and eta <= gamma):
                k = (
                    np.sqrt(max(1 - t_ratio, 0))
                    * (k_max - k_min) + k_min
                )
            elif mu_k <= 0.5 and eta > gamma:
                k = tau1 / 4.0 + (k_max + k_min) / 2.0
            elif mu_k > 0.5 and eta <= gamma:
                k = tau2 / 4.0 + k_min

        n_refract = 1.0

        for i in range(N):

            if self.fe_count >= self.max_fe:
                break

            X_new = np.zeros(self.dimension)

            if self.rng.random() > 0.5:
                # Strategy 1: Standard GWO update
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

                    X_new[j] = (X1 + X2 + X3) / 3.0
            else:
                # Strategy 2: Equilibrium-inspired update
                X_avg = (self.alpha + self.beta + self.delta) / 3.0
                pool = np.array([self.alpha, self.beta, self.delta, X_avg])
                idx = self.rng.randint(0, 4)
                X_eq = pool[idx]

                r = self.rng.random()
                lam = self.rng.random()
                m = (1.0 - fe_ratio) ** fe_ratio

                sign_val = 1.0 if r >= 0.5 else -1.0
                F = sign_val * (np.exp(-lam * m) - 1.0)

                for j in range(self.dimension):
                    if self.rng.random() >= 0.5:
                        G0 = self.rng.random() * (X_eq[j] - lam * self.population[i, j])
                    else:
                        G0 = 0.0

                    G = G0 * F
                    X_new[j] = (
                        X_eq[j]
                        + (self.population[i, j] - X_eq[j]) * F
                        + (G / (lam + 1e-8)) * (1.0 - F)
                    )

            # Clip
            X_new = np.clip(X_new, self.lb, self.ub)

            # Greedy selection for main update
            f_new = self.evaluate(X_new)

            if self.fe_count >= self.max_fe:
                self.update_leaders(X_new, f_new)
                break

            if f_new < self.fitness[i]:
                self.population[i] = X_new
                self.fitness[i] = f_new

            # Refraction learning step
            mid = (self.ub + self.lb) / 2.0
            X_star = (mid - self.population[i]) / (k * n_refract + 1e-8) + mid
            X_star = np.clip(X_star, self.lb, self.ub)

            f_star = self.evaluate(X_star)

            if f_star < self.fitness[i]:
                self.population[i] = X_star
                self.fitness[i] = f_star

        # Re-sort and update leaders
        self.alpha_score = np.inf
        self.beta_score = np.inf
        self.delta_score = np.inf

        self.alpha = np.zeros(self.dimension)
        self.beta = np.zeros(self.dimension)
        self.delta = np.zeros(self.dimension)

        for i in range(N):
            self.update_leaders(self.population[i], self.fitness[i])

        self.best_score = self.alpha_score
        self.best_position = self.alpha.copy()

        self.best_history.append(self.alpha_score)
