"""
igwo_dlh.py

Improved Grey Wolf Optimizer with Distance-based Local Hunting (DLH).

Reference:
    Nadimi-Shahraki et al. (2021)

Description:
    Standard GWO candidate generation followed by
    Distance-based Local Hunting (DLH). The better
    candidate between GWO and DLH is greedily selected.
"""

from optimizers.base_optimizer import BaseOptimizer
from core.registry import optimizer_registry

import numpy as np
from scipy.spatial.distance import cdist


@optimizer_registry.register
class IGWO_DLH(BaseOptimizer):

    def initialize(self):
        """Initialize population, fitness, personal bests and leaders."""

        self.population = self.initialize_population()
        self.fitness = np.full(self.population_size, np.inf)

        # Evaluate initial population
        for i in range(self.population_size):

            if self.fe_count >= self.max_fe:
                break

            self.fitness[i] = self.evaluate(self.population[i])

        # Personal bests
        self.pbest_pos = self.population.copy()
        self.pbest_score = self.fitness.copy()

        # Sort population
        sort_idx = np.argsort(self.fitness)
        self.population = self.population[sort_idx]
        self.fitness = self.fitness[sort_idx]

        self.alpha = self.population[0].copy()
        self.beta = self.population[1].copy()
        self.delta = self.population[2].copy()

        self.alpha_score = self.fitness[0]
        self.beta_score = self.fitness[1]
        self.delta_score = self.fitness[2]

        self.best_score = self.alpha_score
        self.best_position = self.alpha.copy()

    def step(self):
        """One iteration of IGWO-DLH."""

        if self.fe_count >= self.max_fe:
            return

        N = self.population_size

        # -----------------------------------------
        # Update Alpha / Beta / Delta
        # -----------------------------------------

        for i in range(N):

            f = float(self.fitness[i])

            if f < self.alpha_score:
                self.alpha_score = f
                self.alpha = self.population[i].copy()

            if self.alpha_score < f < self.beta_score:
                self.beta_score = f
                self.beta = self.population[i].copy()

            if self.beta_score < f < self.delta_score:
                self.delta_score = f
                self.delta = self.population[i].copy()

        progress = self.fe_count / self.max_fe
        a = 2.0 * (1.0 - progress)

        # Need two complete evaluations
        if self.fe_count + 2 * N > self.max_fe:
            return

        # -----------------------------------------
        # Standard GWO candidates
        # -----------------------------------------

        r1 = self.rng.random((N, self.dimension))
        r2 = self.rng.random((N, self.dimension))
        A1 = 2 * a * r1 - a
        X1 = self.alpha - A1 * np.abs(2 * r2 * self.alpha - self.population)

        r1 = self.rng.random((N, self.dimension))
        r2 = self.rng.random((N, self.dimension))
        A2 = 2 * a * r1 - a
        X2 = self.beta - A2 * np.abs(2 * r2 * self.beta - self.population)

        r1 = self.rng.random((N, self.dimension))
        r2 = self.rng.random((N, self.dimension))
        A3 = 2 * a * r1 - a
        X3 = self.delta - A3 * np.abs(2 * r2 * self.delta - self.population)

        x_gwo = np.clip(
            (X1 + X2 + X3) / 3.0,
            self.lb,
            self.ub,
        )

        fit_gwo = np.full(N, np.inf)

        for i in range(N):

            if self.fe_count >= self.max_fe:
                break

            fit_gwo[i] = self.evaluate(x_gwo[i])

        # -----------------------------------------
        # DLH candidates
        # -----------------------------------------

        radius = np.linalg.norm(
            self.population - x_gwo,
            axis=1
        )

        dist_mat = cdist(
            self.population,
            self.population
        )

        rand_perm = self.rng.permutation(N)

        x_dlh = self.population.copy()
        fit_dlh = self.fitness.copy()

        for i in range(N):

            neighbors = np.where(
                dist_mat[i] <= radius[i]
            )[0]

            if len(neighbors) == 0:
                continue

            rand_nb = self.rng.choice(
                neighbors,
                size=self.dimension
            )

            x_dlh[i] = (
                self.population[i]
                + self.rng.random(self.dimension)
                * (
                    self.population[rand_nb, np.arange(self.dimension)]
                    - self.population[rand_perm[i], np.arange(self.dimension)]
                )
            )

            x_dlh[i] = np.clip(
                x_dlh[i],
                self.lb,
                self.ub,
            )

        # -----------------------------------------
        # Evaluate DLH candidates
        # -----------------------------------------

        for i in range(N):

            if self.fe_count >= self.max_fe:
                break

            fit_dlh[i] = self.evaluate(x_dlh[i])

        # -----------------------------------------
        # Greedy selection
        # -----------------------------------------

        use_gwo = fit_gwo < fit_dlh

        candidate_pos = np.where(
            use_gwo[:, None],
            x_gwo,
            x_dlh,
        )

        candidate_fit = np.where(
            use_gwo,
            fit_gwo,
            fit_dlh,
        )

        improved = candidate_fit < self.pbest_score

        self.pbest_score[improved] = candidate_fit[improved]
        self.pbest_pos[improved] = candidate_pos[improved]

        self.population = self.pbest_pos.copy()
        self.fitness = self.pbest_score.copy()

        # -----------------------------------------
        # Sort population
        # -----------------------------------------

        sort_idx = np.argsort(self.fitness)

        self.population = self.population[sort_idx]
        self.fitness = self.fitness[sort_idx]

        self.pbest_pos = self.pbest_pos[sort_idx]
        self.pbest_score = self.pbest_score[sort_idx]

        self.alpha = self.population[0].copy()
        self.beta = self.population[1].copy()
        self.delta = self.population[2].copy()

        self.alpha_score = self.fitness[0]
        self.beta_score = self.fitness[1]
        self.delta_score = self.fitness[2]

        self.best_score = self.alpha_score
        self.best_position = self.alpha.copy()