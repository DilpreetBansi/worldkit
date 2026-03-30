# WorldKit-Bench

WorldKit-Bench is a standardized benchmark suite for evaluating world models across navigation, manipulation, control, and game environments.

## Overview

WorldKit-Bench evaluates models on four metrics:
- **Success rate** — can the model plan to reach goals?
- **Prediction MSE** — how accurately does it predict next states?
- **Planning time** — how fast is the CEM planner?
- **Plausibility AUROC** — how well does it detect anomalies?

## Quick start

### CLI

```bash
# Quick benchmark (5 tasks, 10 episodes each)
worldkit bench quick --model my_model.wk

# Full benchmark (all tasks, 50 episodes each)
worldkit bench run --model my_model.wk --suite full --episodes 50

# Save results
worldkit bench run --model my_model.wk --output results.json

# Generate HTML report
worldkit bench report --results results.json --output report.html
```

### Python

```python
from worldkit import WorldModel
from worldkit.bench import BenchmarkSuite, BenchmarkRunner

model = WorldModel.load("my_model.wk")

# Quick benchmark
suite = BenchmarkSuite.quick()
runner = BenchmarkRunner(suite, model)
results = runner.run(episodes_per_task=10)
print(results.summary())

# Full benchmark
suite = BenchmarkSuite.full()
runner = BenchmarkRunner(suite, model)
results = runner.run(episodes_per_task=50, seed=42, verbose=True)
```

## Benchmark suites

### Full suite

All tasks across all categories. Use `BenchmarkSuite.full()`.

### Quick suite

5 representative tasks (one per category) with 10 episodes. Use `BenchmarkSuite.quick()`.

### Category suite

Tasks in a specific category. Use `BenchmarkSuite.category("control")`.

Available categories: `navigation`, `manipulation`, `control`, `games`.

## Tasks

### Navigation

| Task | Environment | Action dim | Action type |
|------|-------------|-----------|-------------|
| TwoRoom | worldkit/two-room | 2 | continuous |
| GridWorld | MiniGrid-Empty-8x8-v0 | 3 | discrete |

### Manipulation

| Task | Environment | Action dim | Action type |
|------|-------------|-----------|-------------|
| PushT | worldkit/pusht | 2 | continuous |

### Control

| Task | Environment | Action dim | Action type |
|------|-------------|-----------|-------------|
| CartPole | CartPole-v1 | 1 | discrete |
| Pendulum | Pendulum-v1 | 1 | continuous |
| MountainCar | MountainCarContinuous-v0 | 1 | continuous |
| Reacher | worldkit/reacher | 2 | continuous |
| Acrobot | Acrobot-v1 | 1 | discrete |

### Games

| Task | Environment | Action dim | Action type |
|------|-------------|-----------|-------------|
| Pong | ALE/Pong-v5 | 1 | discrete |
| Breakout | ALE/Breakout-v5 | 1 | discrete |

Tasks with missing dependencies are skipped automatically (e.g., Atari games require `pip install gymnasium[atari]`).

## Metrics

### Success rate

Fraction of episodes where the planned action sequence reaches within a threshold distance of the goal in latent space. Range: 0.0 to 1.0.

### Prediction MSE

Mean squared error between predicted and actual next latent states, averaged over all episodes and timesteps. Lower is better.

### Planning time

Wall-clock time for CEM planning in milliseconds. Measured per-plan and averaged.

### Plausibility AUROC

Area Under the ROC Curve for anomaly detection. The model scores normal sequences (from the evaluation data) and random/shuffled sequences (synthetic anomalies). Higher AUROC = better anomaly detector.

## Results API

### `BenchmarkResults`

```python
results.suite_name           # str — suite name
results.num_tasks            # int — total tasks
results.num_skipped()        # int — tasks skipped due to missing deps
results.avg_success_rate()   # float — mean success rate
results.avg_prediction_mse() # float — mean prediction MSE
results.avg_planning_time_ms() # float — mean planning time
results.total_time_s         # float — total benchmark time

# Per-task results
for task in results.results:
    print(f"{task.task_name}: success={task.success_rate:.2f}")

# Export
results.to_json("results.json")
results.to_html("report.html")
summary = results.summary()  # dict
```

### `TaskResult`

```python
task.task_name            # str
task.category             # str
task.success_rate         # float (0-1)
task.planning_time_ms     # float
task.prediction_mse       # float
task.plausibility_auroc   # float
task.episodes             # int
task.skipped              # bool
task.skip_reason          # str
task.metadata             # dict
```

## Custom benchmark tasks

Extend the benchmark with your own tasks:

```python
from worldkit.bench import BenchmarkTask, TaskResult, BenchmarkSuite, BenchmarkRunner

class MyTask(BenchmarkTask):
    @property
    def name(self) -> str:
        return "MyTask"

    @property
    def category(self) -> str:
        return "custom"

    def setup(self) -> None:
        # Check dependencies, raise ImportError if missing
        pass

    def evaluate(self, model, episodes=50, seed=42) -> TaskResult:
        # Run your evaluation logic
        return TaskResult(
            task_name=self.name,
            category=self.category,
            success_rate=0.75,
            prediction_mse=0.012,
            planning_time_ms=120.0,
            episodes=episodes,
        )

# Create a suite with your task
suite = BenchmarkSuite("custom", [MyTask()])
runner = BenchmarkRunner(suite, model)
results = runner.run()
```

## Comparing models

```python
models = {
    "nano": WorldModel.load("nano.wk"),
    "base": WorldModel.load("base.wk"),
}

suite = BenchmarkSuite.quick()

for name, model in models.items():
    runner = BenchmarkRunner(suite, model)
    results = runner.run(episodes_per_task=10, verbose=False)
    print(f"{name}: success={results.avg_success_rate():.2%}, "
          f"mse={results.avg_prediction_mse():.4f}, "
          f"time={results.avg_planning_time_ms():.0f}ms")
```

## Leaderboard

```python
from worldkit.bench import format_leaderboard_entry, save_leaderboard_entry

entry = format_leaderboard_entry(model, results)
save_leaderboard_entry(entry, "leaderboard.json")
```

## Related

- [Benchmark your model tutorial](tutorials/benchmark_your_model.md) — step-by-step walkthrough
- [Model Zoo](model_zoo.md) — pre-trained models to benchmark against
- [CLI reference](cli.md) — `worldkit bench` commands
