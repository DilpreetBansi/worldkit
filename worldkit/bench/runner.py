"""BenchmarkRunner — executes a suite against a model and collects results."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .report import generate_html_report
from .suite import BenchmarkSuite
from .task import TaskResult


@dataclass
class BenchmarkResults:
    """Collected results from running a benchmark suite.

    Attributes:
        suite_name: Name of the suite that was run.
        results: List of TaskResult objects, one per task.
        total_time_s: Total wall-clock time for the full run.
        model_name: Name/config of the model evaluated.
        model_params: Number of model parameters.
    """

    suite_name: str
    results: list[TaskResult] = field(default_factory=list)
    total_time_s: float = 0.0
    model_name: str = ""
    model_params: int = 0

    @property
    def num_tasks(self) -> int:
        """Number of tasks that were evaluated (not skipped)."""
        return sum(1 for r in self.results if not r.skipped)

    @property
    def num_skipped(self) -> int:
        """Number of tasks that were skipped."""
        return sum(1 for r in self.results if r.skipped)

    @property
    def avg_success_rate(self) -> float:
        """Average success rate across non-skipped tasks."""
        active = [r.success_rate for r in self.results if not r.skipped]
        return float(np.mean(active)) if active else 0.0

    @property
    def avg_prediction_mse(self) -> float:
        """Average prediction MSE across non-skipped tasks."""
        active = [r.prediction_mse for r in self.results if not r.skipped]
        return float(np.nanmean(active)) if active else 0.0

    @property
    def avg_planning_time_ms(self) -> float:
        """Average planning time across non-skipped tasks."""
        active = [r.planning_time_ms for r in self.results if not r.skipped]
        return float(np.mean(active)) if active else 0.0

    def summary(self) -> dict:
        """Return an aggregate summary of the benchmark run."""
        return {
            "suite": self.suite_name,
            "model_name": self.model_name,
            "model_params": self.model_params,
            "total_tasks": len(self.results),
            "evaluated": self.num_tasks,
            "skipped": self.num_skipped,
            "avg_success_rate": round(self.avg_success_rate, 4),
            "avg_prediction_mse": round(self.avg_prediction_mse, 6),
            "avg_planning_time_ms": round(self.avg_planning_time_ms, 2),
            "total_time_s": round(self.total_time_s, 2),
        }

    def to_dict(self) -> dict:
        """Serialize full results to a dictionary."""
        return {
            "summary": self.summary(),
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, path: str | Path) -> None:
        """Write results to a JSON file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_html(self, path: str | Path) -> None:
        """Write results to an HTML report.

        Args:
            path: Output file path.
        """
        html = generate_html_report(self)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(html)

    @classmethod
    def from_json(cls, path: str | Path) -> BenchmarkResults:
        """Load results from a JSON file.

        Args:
            path: Path to the JSON results file.

        Returns:
            Reconstructed BenchmarkResults.
        """
        with open(path) as f:
            data = json.load(f)

        summary = data["summary"]
        results = [
            TaskResult(
                task_name=r["task_name"],
                category=r["category"],
                success_rate=r.get("success_rate", 0.0),
                planning_time_ms=r.get("planning_time_ms", 0.0),
                prediction_mse=r.get("prediction_mse", 0.0),
                plausibility_auroc=r.get("plausibility_auroc", 0.0),
                episodes=r.get("episodes", 0),
                skipped=r.get("skipped", False),
                skip_reason=r.get("skip_reason", ""),
                metadata=r.get("metadata", {}),
            )
            for r in data["results"]
        ]

        return cls(
            suite_name=summary.get("suite", "unknown"),
            results=results,
            total_time_s=summary.get("total_time_s", 0.0),
            model_name=summary.get("model_name", ""),
            model_params=summary.get("model_params", 0),
        )


class BenchmarkRunner:
    """Executes a BenchmarkSuite against a WorldModel.

    Usage:
        suite = BenchmarkSuite.full()
        runner = BenchmarkRunner(suite, model)
        results = runner.run(episodes_per_task=50, seed=42)
        results.to_json("results.json")
        results.to_html("report.html")
    """

    def __init__(self, suite: BenchmarkSuite, model):
        """Initialize the runner.

        Args:
            suite: BenchmarkSuite to evaluate.
            model: WorldModel instance.
        """
        self._suite = suite
        self._model = model

    def run(
        self,
        episodes_per_task: int = 50,
        seed: int = 42,
        verbose: bool = True,
    ) -> BenchmarkResults:
        """Run all tasks in the suite against the model.

        Args:
            episodes_per_task: Number of evaluation episodes per task.
            seed: Random seed for reproducibility.
            verbose: If True, print progress to stdout.

        Returns:
            BenchmarkResults with all task results.
        """
        results = BenchmarkResults(
            suite_name=self._suite.name,
            model_name=self._model.config.name,
            model_params=self._model.num_params,
        )

        start = time.perf_counter()

        for i, task in enumerate(self._suite):
            task_label = f"[{i + 1}/{len(self._suite)}] {task.name}"

            if verbose:
                print(f"WorldKit-Bench | Running {task_label}...")

            # Setup — skip if env is missing
            try:
                task.setup()
            except ImportError as e:
                if verbose:
                    print(f"WorldKit-Bench | Skipped {task.name}: {e}")
                results.results.append(
                    TaskResult(
                        task_name=task.name,
                        category=task.category,
                        skipped=True,
                        skip_reason=str(e),
                    )
                )
                continue

            # Evaluate
            try:
                task_result = task.evaluate(
                    self._model,
                    episodes=episodes_per_task,
                    seed=seed,
                )
                results.results.append(task_result)
                if verbose:
                    print(
                        f"WorldKit-Bench | {task.name}: "
                        f"success={task_result.success_rate:.2%} "
                        f"mse={task_result.prediction_mse:.4f} "
                        f"plan={task_result.planning_time_ms:.1f}ms"
                    )
            except Exception as e:
                if verbose:
                    print(f"WorldKit-Bench | Error on {task.name}: {e}")
                results.results.append(
                    TaskResult(
                        task_name=task.name,
                        category=task.category,
                        skipped=True,
                        skip_reason=f"Error: {e}",
                    )
                )

        results.total_time_s = time.perf_counter() - start

        if verbose:
            s = results.summary()
            print(
                f"\nWorldKit-Bench | Done: {s['evaluated']}/{s['total_tasks']} tasks "
                f"| Avg success: {s['avg_success_rate']:.2%} "
                f"| Time: {s['total_time_s']:.1f}s"
            )

        return results
