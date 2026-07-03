"""
Global configuration for the GWO Population Analysis Framework.
All experiment settings should be modified only from this file.
"""

from pathlib import Path


# ==============================
# Project Directories
# ==============================

ROOT_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = ROOT_DIR / "results"
LOG_DIR = ROOT_DIR / "logs"
CONFIG_DIR = ROOT_DIR / "configs"


# ==============================
# Benchmark Suites
# ==============================

BENCHMARKS = [
    "CEC2013",
    "CEC2017",
    "CEC2020",
    "CEC2022",
]

# Number of functions per benchmark suite.
# Adding a new benchmark only requires a new entry here
# and a corresponding wrapper — no other code changes.
BENCHMARK_FUNCTIONS = {
    "CEC2013": 28,   # F1 – F28
    "CEC2017": 29,   # F1 – F29
    "CEC2020": 10,   # F1 – F10
    "CEC2022": 12,   # F1 – F12
}


# ==============================
# Experiment Parameters
# ==============================

DIMENSIONS = [
    10,
    30,
    50,
    100,
]

POPULATION_SIZES = [
    5,
    10,
    30,
    50,
    75,
    100,
    150,
    200,
    300,
    500,
    1000,
    1500,
    2000,
]

RUNS = 30

MAX_FE_MULTIPLIER = 10000


# ==============================
# Optimizers
# ==============================

OPTIMIZERS = [
    "GWO",
    "BBGWO",
    "IAGWO",
    "MENGWO",
    "MGWO",
    "RWGWO",
    "OBGWO",
    "modGWO",
    "EBGWO",
    "IGWO_MS",
    "AGWO",
    "IGWO_DLH"
]


# ==============================
# Reproducibility
# ==============================

BASE_SEED = 1


# ==============================
# Save Settings
# ==============================

SAVE_CONVERGENCE = True

SAVE_RAW_RESULTS = True

SAVE_SUMMARY = True

OVERWRITE_EXISTING = False


# ==============================
# Logging
# ==============================

LOG_LEVEL = "INFO"

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
