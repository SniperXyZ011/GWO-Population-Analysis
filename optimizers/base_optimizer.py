"""
base_optimizer.py

Abstract base class for all optimization algorithms.
Implements the Template Method Pattern — the optimization
loop exists only here. Subclasses implement initialize() and step().

Every optimizer (GWO, REGWO, BBGWO, etc.) must inherit from
this class.
"""

from abc import ABC, abstractmethod
import numpy as np
import time


class BaseOptimizer(ABC):

    def __init__(
        self,
        problem,
        population_size,
        max_function_evaluations,
        seed=None,
    ):

        self.problem = problem

        self.population_size = population_size

        self.dimension = problem.dimension

        self.max_fe = max_function_evaluations

        self.seed = seed

        # Use a local RandomState for thread safety
        self.rng = np.random.RandomState(seed)

        # -----------------------------
        # Search Space
        # -----------------------------

        self.lb = problem.lower_bound()

        self.ub = problem.upper_bound()

        # -----------------------------
        # Runtime Statistics
        # -----------------------------

        self.fe_count = 0

        self.iteration = 0

        self.execution_time = 0.0

        # -----------------------------
        # Best Solution
        # -----------------------------

        self.best_position = None

        self.best_score = np.inf

        # -----------------------------
        # Convergence History
        # -----------------------------

        self.convergence_curve = []

    @property
    def max_iterations(self):
        """Approximate maximum number of iterations."""
        return max(1, self.max_fe // max(1, self.population_size))

    # ===================================================
    # Template Method: optimize()
    # ===================================================

    def optimize(self):
        """
        Run the full optimization process.

        This is the ONLY optimization loop in the framework.
        Subclasses must NOT override this method.

        Returns
        -------
        dict
            Result dictionary.
        """

        self.reset()

        self.start_timer()

        self.initialize()

        while self.fe_count < self.max_fe:

            self.step()

            self.iteration += 1

            self.record_progress()

        self.stop_timer()

        return self.get_results()

    # ===================================================
    # Abstract Methods (subclass responsibility)
    # ===================================================

    @abstractmethod
    def initialize(self):
        """
        Initialize optimizer-specific variables and
        evaluate the initial population.
        """
        pass

    @abstractmethod
    def step(self):
        """
        Execute one optimization iteration.
        Must update positions and call self.evaluate()
        for each function evaluation.
        """
        pass

    # ===================================================
    # Common Utility Methods
    # ===================================================

    def initialize_population(self):
        """
        Create a random initial population within bounds.

        Returns
        -------
        np.ndarray
            Shape (population_size, dimension).
        """

        return self.rng.uniform(
            self.lb,
            self.ub,
            (self.population_size, self.dimension)
        )

    def evaluate(self, position):
        """
        Evaluate a single candidate and increment FE counter.

        Parameters
        ----------
        position : np.ndarray

        Returns
        -------
        float
            Fitness value, or np.inf if budget exhausted.
        """

        if self.fe_count >= self.max_fe:
            return np.inf

        self.fe_count += 1

        return self.problem.evaluate(position)

    def evaluate_population(self, population):
        """
        Evaluate an entire population.

        Parameters
        ----------
        population : np.ndarray
            Shape (N, dimension).

        Returns
        -------
        np.ndarray
            Fitness values of shape (N,).
        """

        fitness = np.full(len(population), np.inf)

        for i, individual in enumerate(population):

            if self.fe_count >= self.max_fe:
                break

            fitness[i] = self.evaluate(individual)

        return fitness

    def update_best(self, position, fitness):
        """
        Update the global best if fitness is better.

        Parameters
        ----------
        position : np.ndarray
        fitness : float
        """

        if fitness < self.best_score:
            self.best_score = fitness
            self.best_position = position.copy()

    def update_leaders(self, position, fitness):
        """
        Update alpha, beta, delta leaders (cascade).

        Subclass must have alpha, beta, delta attributes.

        Parameters
        ----------
        position : np.ndarray
        fitness : float
        """

        if fitness < self.alpha_score:

            self.delta_score = self.beta_score
            self.delta = self.beta.copy()

            self.beta_score = self.alpha_score
            self.beta = self.alpha.copy()

            self.alpha_score = fitness
            self.alpha = position.copy()

        elif fitness < self.beta_score:

            self.delta_score = self.beta_score
            self.delta = self.beta.copy()

            self.beta_score = fitness
            self.beta = position.copy()

        elif fitness < self.delta_score:

            self.delta_score = fitness
            self.delta = position.copy()

    def clip(self, population):
        """
        Clip solutions within search bounds.

        Parameters
        ----------
        population : np.ndarray

        Returns
        -------
        np.ndarray
        """

        return np.clip(population, self.lb, self.ub)

    def record_progress(self):
        """
        Record the current best score.

        Subsamples to ~1000 points for large runs to
        prevent excessive memory and storage usage.
        """

        # Estimate total iterations: max_fe / population_size
        estimated_iters = max(
            1, self.max_fe // max(1, self.population_size)
        )

        # Record every N-th iteration, targeting ~1000 data points
        record_interval = max(1, estimated_iters // 1000)

        if (
            self.iteration % record_interval == 0
            or self.fe_count >= self.max_fe
        ):
            self.convergence_curve.append(self.best_score)

    def reset(self):
        """Reset all tracking variables for a fresh run."""

        self.fe_count = 0

        self.iteration = 0

        self.best_score = np.inf

        self.best_position = None

        self.convergence_curve = []

        # Re-seed RNG for reproducibility
        self.rng = np.random.RandomState(self.seed)

    # ===================================================
    # Timing
    # ===================================================

    def start_timer(self):

        self._start_time = time.perf_counter()

    def stop_timer(self):

        self.execution_time = (
            time.perf_counter() - self._start_time
        )

    # ===================================================
    # Result
    # ===================================================

    def get_results(self):
        """
        Package results as a dictionary.

        Returns
        -------
        dict
        """

        fe_per_sec = (
            self.fe_count / max(self.execution_time, 1e-9)
        )

        return {

            "optimizer": self.__class__.__name__,

            "best_score": float(self.best_score),

            "best_position": (
                self.best_position.tolist()
                if self.best_position is not None
                else None
            ),

            "function_evaluations": self.fe_count,

            "iterations": self.iteration,

            "execution_time": self.execution_time,

            "fe_per_second": fe_per_sec,

            "convergence_curve": [
                float(v) for v in self.convergence_curve
            ],
        }

    def __repr__(self):

        return (
            f"{self.__class__.__name__}"
            f"(Pop={self.population_size}, "
            f"Dim={self.dimension}, "
            f"MaxFE={self.max_fe})"
        )
