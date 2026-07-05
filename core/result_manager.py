"""
result_manager.py

Handles all result saving for the framework.
Optimizers should never save files directly.
"""

from pathlib import Path
import json
import csv


class ResultManager:

    def __init__(self, root_dir):

        self.root = Path(root_dir)

    # ---------------------------------------------------------
    # Directory Creation
    # ---------------------------------------------------------

    def create_experiment_folder(self, experiment):
        """
        Create the hierarchical directory for an experiment.

        Structure:
            results/{benchmark}/{optimizer}/F{n}/D{dim}/P{pop}/
        """

        path = (
            self.root
            / experiment.benchmark
            / experiment.optimizer
            / f"F{experiment.function}"
            / f"D{experiment.dimension}"
            / f"P{experiment.population_size}"
        )

        path.mkdir(parents=True, exist_ok=True)

        return path

    # ---------------------------------------------------------
    # Save Final Result
    # ---------------------------------------------------------

    def save_result(self, experiment, result):
        """Save a single run result as JSON."""

        folder = self.create_experiment_folder(experiment)

        filename = folder / f"run_{experiment.run}.json"

        # Convert numpy types for JSON serialization
        serializable = self._make_serializable(result)

        with open(filename, "w") as f:

            json.dump(serializable, f, indent=4)

    # ---------------------------------------------------------
    # Save Convergence Curve
    # ---------------------------------------------------------

    def save_convergence(self, experiment, curve):
        """Save convergence curve as CSV."""

        folder = self.create_experiment_folder(experiment)

        filename = folder / f"convergence_run_{experiment.run}.csv"

        with open(filename, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(["Iteration", "Best_Fitness"])

            for i, value in enumerate(curve):

                writer.writerow([i + 1, value])

    # ---------------------------------------------------------
    # Save Summary
    # ---------------------------------------------------------

    def save_summary(self, experiment, summary):
        """Save aggregated summary statistics."""

        folder = self.create_experiment_folder(experiment)

        filename = folder / "summary.json"

        serializable = self._make_serializable(summary)

        with open(filename, "w") as f:

            json.dump(serializable, f, indent=4)

    # ---------------------------------------------------------
    # Load Results
    # ---------------------------------------------------------

    def load_result(self, experiment):
        """Load a single run result from JSON."""

        folder = self.create_experiment_folder(experiment)

        filename = folder / f"run_{experiment.run}.json"

        if not filename.exists():
            return None

        with open(filename, "r") as f:
            return json.load(f)

    def result_exists(self, experiment):
        """Check if a result already exists for this experiment."""

        folder = (
            self.root
            / experiment.benchmark
            / experiment.optimizer
            / f"F{experiment.function}"
            / f"D{experiment.dimension}"
            / f"P{experiment.population_size}"
        )

        filename = folder / f"run_{experiment.run}.json"

        return filename.exists()

    # ---------------------------------------------------------
    # Serialization Helpers
    # ---------------------------------------------------------

    @staticmethod
    def _make_serializable(data):
        """Convert numpy types to Python builtins for JSON."""

        import numpy as np

        if isinstance(data, dict):
            return {
                k: ResultManager._make_serializable(v)
                for k, v in data.items()
            }
        elif isinstance(data, (list, tuple)):
            return [
                ResultManager._make_serializable(v) for v in data
            ]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer,)):
            return int(data)
        elif isinstance(data, (np.floating,)):
            return float(data)
        else:
            return data
