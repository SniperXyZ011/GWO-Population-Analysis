"""
ebgwo.py

Enhanced Bare Bones Grey Wolf Optimizer (EBGWO).

Key enhancements over BBGWO:
    1. Adaptive sigma based on fitness rank
    2. Boundary learning: if a wolf is near a boundary,
       reflect it towards the center with a probability
    3. Gaussian walk with linearly shrinking sigma
"""

from optimizers.base_optimizer import BaseOptimizer
from core.registry import optimizer_registry

import numpy as np


@optimizer_registry.register
class EBGWO(BaseOptimizer):

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

        for i in range(self.population_size):

            if self.fe_count >= self.max_fe:
                break

            self.fitness[i] = self.evaluate(self.population[i])

            self.update_leaders(self.population[i], self.fitness[i])

        self.best_score = self.alpha_score
        self.best_position = self.alpha.copy()

    def step(self):
        """One iteration of EBGWO."""

        N = self.population_size
        fe_ratio = self.fe_count / self.max_fe

        # Linearly decreasing a: 2 → 0
        a = 2.0 * (1.0 - fe_ratio)

        # Sort to get fitness ranks
        sort_idx = np.argsort(self.fitness[:N])
        ranks = np.zeros(N, dtype=int)
        for rank, idx in enumerate(sort_idx):
            ranks[idx] = rank

        for i in range(N):

            if self.fe_count >= self.max_fe:
                break

            new_pos = np.zeros(self.dimension)

            # Adaptive sigma scaling based on rank
            # Top wolves (low rank) get smaller sigma (exploitation)
            # Bottom wolves (high rank) get larger sigma (exploration)
            rank_ratio = ranks[i] / max(N - 1, 1)
            sigma_scale = 0.5 + 0.5 * rank_ratio  # 0.5 to 1.0

            for j in range(self.dimension):

                # BBGWO mean
                mu = (
                    self.alpha[j]
                    + self.beta[j]
                    + self.delta[j]
                ) / 3.0

                # Enhanced variance terms
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

                sigma = (
                    sigma_scale
                    * (a / (3.0 * np.sqrt(3.0)))
                    * np.sqrt(term1 + term2 + term3)
                )

                new_pos[j] = mu + sigma * self.rng.randn()

            # Boundary learning: if near boundary, probabilistically
            # reflect towards center
            lb_arr = self.lb if hasattr(self.lb, '__len__') else np.full(self.dimension, self.lb)
            ub_arr = self.ub if hasattr(self.ub, '__len__') else np.full(self.dimension, self.ub)
            search_range = ub_arr - lb_arr
            center = (ub_arr + lb_arr) / 2.0

            for j in range(self.dimension):
                dist_to_lb = abs(new_pos[j] - lb_arr[j])
                dist_to_ub = abs(new_pos[j] - ub_arr[j])
                min_dist = min(dist_to_lb, dist_to_ub)

                if min_dist < 0.1 * search_range[j] and self.rng.random() < 0.3:
                    # Reflect towards center
                    new_pos[j] = (
                        new_pos[j]
                        + self.rng.random() * (center[j] - new_pos[j])
                    )

            # Clip and evaluate
            new_pos = np.clip(new_pos, self.lb, self.ub)

            new_fitness = self.evaluate(new_pos)

            # Greedy selection
            if new_fitness < self.fitness[i]:
                self.population[i] = new_pos
                self.fitness[i] = new_fitness

            self.update_leaders(self.population[i], self.fitness[i])

        self.best_score = self.alpha_score
        self.best_position = self.alpha.copy()
