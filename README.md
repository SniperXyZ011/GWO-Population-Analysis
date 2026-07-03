# GWO Population Analysis Framework

A research-grade, modular, scalable, and highly parallelized optimization framework for conducting comprehensive **population size analysis** of Grey Wolf Optimizer (GWO) and its variants. Designed specifically to execute massive benchmarking campaigns on High-Performance Computing (HPC) environments.

## 🚀 Key Features
- **Massive Parallelization**: Auto-detects system resources and leverages 100% of available CPU cores (via Python `ProcessPoolExecutor`) to churn through hundreds of thousands of experiments.
- **Crash-Resilient Checkpointing**: Uses a WAL-mode SQLite database (`checkpoint.db`) to instantly save progress. If the server reboots or you `Ctrl+C`, the framework resumes exactly where it left off.
- **Detailed Progress Tracking**: Run `python main.py status` at any time from another terminal window to view a real-time, highly detailed dashboard of your campaign's throughput, live CPU/RAM usage, and Estimated Time of Arrival (ETA).
- **12 GWO Variants**: Implementations of the standard Grey Wolf Optimizer, Bare Bones variants, Adaptive versions, and more.
- **State-of-the-art Benchmarks**: Integrates natively with `opfunu` for CEC2013, CEC2017, CEC2020, and CEC2022.

---

## 📦 Algorithms Implemented

| # | Algorithm | Class Name | Description |
|---|-----------|-----------|-------------|
| 1 | Original GWO | `GWO` | Standard Grey Wolf Optimizer |
| 2 | BBGWO | `BBGWO` | Bare Bones GWO |
| 3 | IAGWO | `IAGWO` | Improved Adaptive Grey Wolf Optimizer |
| 4 | MENGWO | `MENGWO` | Mutation + Evolution + NPSR GWO |
| 5 | MGWO | `MGWO` | Modified GWO (Lévy flight) |
| 6 | RWGWO | `RWGWO` | Random Walk GWO |
| 7 | OBGWO | `OBGWO` | Opposition-Based GWO |
| 8 | modGWO | `modGWO` | Modified GWO (weighted leaders) |
| 9 | EBGWO | `EBGWO` | Enhanced Bare Bones GWO |
| 10 | IGWO-MS | `IGWO_MS` | Improved GWO Multi-Strategy |
| 11 | AGWO | `AGWO` | Adaptive GWO |
| 12 | IGWO_DLH | `IGWO_DLH`| Improved GWO with Distance-based Local Hunting| 

---

## ⚙️ Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone <your-repo>
cd GWO-Population-Analysis

# 2. Install dependencies (numpy, opfunu, psutil, pyyaml)
pip install -r requirements.txt
```

---

## 🚦 How to Run the Framework

The framework operates declaratively using YAML configuration files.

### 1. Configure your Campaign
Edit the `configs/experiment.yaml` file to define the grid of experiments you want to execute.
```yaml
benchmarks:
  - CEC2020
  - CEC2017
  
optimizers:
  - GWO
  - BBGWO

dimensions: [10, 30, 50, 100]
population_sizes: [10, 30, 50, 100, 500]
runs: 30

execution:
  workers: 0   # 0 means auto-detect and use maximum available CPUs
```

### 2. Start the Campaign
Simply point `main.py` to your YAML config:
```bash
python main.py run --config configs/experiment.yaml
```
*Note: Depending on your grid, this can queue up millions of tasks. The framework will gracefully handle the queue without memory leaks.*

### 3. Monitor Progress in Real-Time
Open a **second terminal window** while the campaign is running and type:
```bash
python main.py status
```
This will generate a highly detailed, live dashboard showing:
- Your exact parallel speedup
- True Function Evaluations (FE) per second
- Live RAM/CPU resource limits
- Estimated Wall-Clock hours remaining (ETA)

---

## 🐛 Troubleshooting & Debugging

### 1. Stopping and Resuming
**Q: I need to reboot my server. Will I lose my progress?**
**A:** No! Press `Ctrl+C` to gracefully terminate the worker pool. When your server comes back online, run the exact same `python main.py run` command. The framework will instantly read `checkpoint.db`, skip all completed experiments, and resume exactly on the ones that were interrupted.

### 2. "ValueError: F12022 problem supports maximum 20 variables!"
**Cause:** The mathematical definitions for the **CEC2022** benchmark were explicitly designed *only* for dimensions 10 and 20. The `opfunu` library enforces this rule.
**Fix:** If you want to benchmark dimensions like `30`, `50`, or `100`, you must use **CEC2013**, **CEC2017**, or **CEC2020** which natively support higher dimensions.

### 3. "RuntimeWarning: invalid value encountered in divide" (opfunu)
**Cause:** Certain hybrid composition functions in **CEC2017** partition variables into chunks. For `D=10`, a chunk can evaluate to exactly `1` variable. The older `opfunu` library divides by `(ndim - 1)`, resulting in a division by zero and `NaN` results.
**Fix:** This framework requires a patched version of `opfunu`'s `operator.py`. If you see this error, ensure you have correctly updated `opfunu` or manually patched `max(1, ndim - 1)` into the `elliptic_func` inside your virtual environment's site-packages.

### 4. Running in Debug Mode (Sequential)
If an optimizer is acting strangely or crashing, you can force the framework to run in sequential mode (1 worker) to see the exact traceback printed to the terminal:
```bash
python main.py run --config configs/experiment.yaml --workers 1
```

---

## 🛠️ Adding Custom Components

### Adding a New Optimizer
1. Create a new python file: `optimizers/my_gwo.py`
2. Inherit from `BaseOptimizer` and override `initialize()` and `step()`
3. Import your file into `optimizers/__init__.py`
4. Use the `@optimizer_registry.register` decorator at the top of your class.

### Adding a New Benchmark
1. Create a folder: `benchmarks/new_benchmark/benchmark.py`
2. Inherit from `BaseProblem` and decorate your class with `@benchmark_registry.register`
3. Add the function count to `configs/config.py` in the `BENCHMARK_FUNCTIONS` dict.
