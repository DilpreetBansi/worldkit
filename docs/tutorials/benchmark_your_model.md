# Benchmark your model

This tutorial shows how to evaluate a world model using WorldKit-Bench — a standardized benchmark suite covering navigation, manipulation, control, and games.

## Prerequisites

```bash
pip install worldkit[envs]
```

## Quick benchmark

Run a fast 5-task evaluation:

```bash
worldkit bench quick --model my_model.wk
```

Or in Python:

```python
from worldkit import WorldModel
from worldkit.bench import BenchmarkSuite, BenchmarkRunner

model = WorldModel.load("my_model.wk")
suite = BenchmarkSuite.quick()
runner = BenchmarkRunner(suite, model)
results = runner.run(episodes_per_task=10, verbose=True)

print(f"Average success rate: {results.avg_success_rate():.2%}")
print(f"Average prediction MSE: {results.avg_prediction_mse():.4f}")
```

**Expected output:**

```
Running WorldKit-Bench (quick) — 5 tasks
  CartPole .............. success=0.80  pred_mse=0.0123  time=52ms
  Pendulum .............. success=0.65  pred_mse=0.0234  time=48ms
  PushT ................. success=0.70  pred_mse=0.0189  time=145ms
  TwoRoom ............... success=0.55  pred_mse=0.0312  time=89ms
  Pong .................. skipped (ALE not installed)

Average success rate: 0.675
Average prediction MSE: 0.0215
Total time: 12.3s
```

## Full benchmark

Run the complete suite (all tasks, 50 episodes each):

```bash
worldkit bench run \
    --model my_model.wk \
    --suite full \
    --episodes 50 \
    --output results.json
```

```python
suite = BenchmarkSuite.full()
runner = BenchmarkRunner(suite, model)
results = runner.run(episodes_per_task=50, seed=42)
```

## Category benchmark

Run tasks in a specific category only:

```python
# Only control tasks
suite = BenchmarkSuite.category("control")
runner = BenchmarkRunner(suite, model)
results = runner.run(episodes_per_task=50)
```

Available categories: `navigation`, `manipulation`, `control`, `games`.

## Understanding results

### `BenchmarkResults`

```python
results.suite_name         # "full", "quick", or category name
results.num_tasks          # Total tasks
results.num_skipped()      # Skipped (missing env dependencies)
results.avg_success_rate() # Mean success rate (0-1)
results.avg_prediction_mse()  # Mean latent prediction error
results.avg_planning_time_ms() # Mean planning time
results.total_time_s       # Total benchmark wall-clock time
```

### `TaskResult`

Each task produces a `TaskResult`:

```python
for task_result in results.results:
    print(f"{task_result.task_name}:")
    print(f"  Category: {task_result.category}")
    print(f"  Success rate: {task_result.success_rate:.2%}")
    print(f"  Prediction MSE: {task_result.prediction_mse:.4f}")
    print(f"  Planning time: {task_result.planning_time_ms:.0f}ms")
    print(f"  Plausibility AUROC: {task_result.plausibility_auroc:.3f}")
    if task_result.skipped:
        print(f"  Skipped: {task_result.skip_reason}")
```

### Metrics explained

| Metric | Description | Good value |
|--------|-------------|------------|
| `success_rate` | Fraction of episodes where the plan reached the goal | > 0.7 |
| `prediction_mse` | MSE between predicted and actual next latents | < 0.05 |
| `planning_time_ms` | Wall-clock time for CEM planning | < 200ms |
| `plausibility_auroc` | AUROC for anomaly detection | > 0.8 |

## Generate a report

### JSON output

```python
results.to_json("results.json")

# Load back
from worldkit.bench import BenchmarkResults
loaded = BenchmarkResults.from_json("results.json")
```

### HTML report

```python
results.to_html("benchmark_report.html")
```

Or via CLI:

```bash
worldkit bench report --results results.json --output report.html
```

The HTML report includes:
- Summary table with all metrics
- Per-category breakdowns
- Model configuration details

## Compare models

Benchmark multiple models and compare:

```python
from worldkit import WorldModel
from worldkit.bench import BenchmarkSuite, BenchmarkRunner

suite = BenchmarkSuite.quick()
models = {
    "nano": WorldModel.load("nano_model.wk"),
    "base": WorldModel.load("base_model.wk"),
    "large": WorldModel.load("large_model.wk"),
}

for name, model in models.items():
    runner = BenchmarkRunner(suite, model)
    results = runner.run(episodes_per_task=10, verbose=False)
    print(f"{name}: success={results.avg_success_rate():.2%}, "
          f"mse={results.avg_prediction_mse():.4f}, "
          f"params={model.num_params:,}")
```

## Available tasks

### Navigation

| Task | Gym ID | Action dim | Description |
|------|--------|-----------|-------------|
| TwoRoom | — | 2 | Navigate between two rooms |
| GridWorld | MiniGrid-Empty-8x8-v0 | 3 | Grid navigation |

### Manipulation

| Task | Gym ID | Action dim | Description |
|------|--------|-----------|-------------|
| PushT | — | 2 | Push T-block to target position |

### Control

| Task | Gym ID | Action dim | Description |
|------|--------|-----------|-------------|
| CartPole | CartPole-v1 | 1 | Balance pendulum on cart |
| Pendulum | Pendulum-v1 | 1 | Swing-up and balance |
| MountainCar | MountainCarContinuous-v0 | 1 | Drive car up hill |
| Reacher | — | 2 | 2-joint arm reaching |
| Acrobot | Acrobot-v1 | 1 | Swing double pendulum |

### Games

| Task | Gym ID | Action dim | Description |
|------|--------|-----------|-------------|
| Pong | ALE/Pong-v5 | 1 | Atari Pong |
| Breakout | ALE/Breakout-v5 | 1 | Atari Breakout |

Tasks with missing dependencies (e.g., ALE for Atari games) are skipped automatically.

## Next steps

- [Benchmarks reference](../benchmarks.md) — full WorldKit-Bench documentation
- [Contribute a model](contribute_model.md) — share your results on the Hub
- [Model configurations guide](../guides/model_configs.md) — choose the right config
