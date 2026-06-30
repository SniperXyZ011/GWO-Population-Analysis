# GWO Population Analysis Framework

A research-grade, modular, scalable, and extensible optimization framework for conducting comprehensive **population size analysis** of Grey Wolf Optimizer (GWO) and its variants.

## Algorithms

| # | Algorithm | Class Name | Description |
|---|-----------|-----------|-------------|
| 1 | Original GWO | `GWO` | Standard Grey Wolf Optimizer |
| 2 | BBGWO | `BBGWO` | Bare Bones GWO |
| 3 | REGWO | `REGWO` | Refraction-Enhanced GWO |
| 4 | MENGWO | `MENGWO` | Mutation + Evolution + NPSR GWO |
| 5 | MGWO | `MGWO` | Modified GWO (Lévy flight) |
| 6 | RWGWO | `RWGWO` | Random Walk GWO |
| 7 | OBGWO | `OBGWO` | Opposition-Based GWO |
| 8 | modGWO | `modGWO` | Modified GWO (weighted leaders) |
| 9 | EBGWO | `EBGWO` | Enhanced Bare Bones GWO |
| 10 | IGWO-MS | `IGWO_MS` | Improved GWO Multi-Strategy |
| 11 | AGWO | `AGWO` | Adaptive GWO |

## Benchmarks

- **CEC2017** (30 functions)
- **CEC2020** (10 functions)
- **CEC2022** (12 functions)

## Quick Start

```bash
pip install -r requirements.txt
python main.py --benchmark CEC2020 --optimizer GWO --dimension 10 --population 10 --runs 5
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Adding a New Optimizer

1. Create `optimizers/new_optimizer.py`
2. Inherit from `BaseOptimizer`
3. Decorate with `@optimizer_registry.register`
4. Implement `initialize()` and `step()`
5. Add import in `optimizers/__init__.py`

## Adding a New Benchmark

1. Create `benchmarks/new_benchmark/benchmark.py`
2. Inherit from `BaseProblem`
3. Decorate with `@benchmark_registry.register`
4. Add function count to `BENCHMARK_FUNCTIONS` in `configs/config.py`

No other framework component needs modification.

## Design Patterns

- **Template Method**: `BaseOptimizer.optimize()` — single loop
- **Factory**: `OptimizerFactory`, `BenchmarkFactory`
- **Registry**: Auto-registration via decorators
- **Immutable Data**: Frozen `Experiment` dataclass
