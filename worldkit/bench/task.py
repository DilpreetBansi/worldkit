"""Benchmark task base class and result dataclass."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TaskResult:
    """Result from evaluating a single benchmark task.

    Attributes:
        task_name: Name of the benchmark task.
        category: Task category (navigation, manipulation, control, games).
        success_rate: Fraction of episodes where the goal was reached (0.0-1.0).
        planning_time_ms: Average planning time in milliseconds.
        prediction_mse: Mean squared error of latent predictions.
        plausibility_auroc: AUROC for plausibility anomaly detection.
        episodes: Number of episodes evaluated.
        skipped: Whether the task was skipped (missing env).
        skip_reason: Reason the task was skipped, if applicable.
        metadata: Additional task-specific metrics.
    """

    task_name: str
    category: str
    success_rate: float = 0.0
    planning_time_ms: float = 0.0
    prediction_mse: float = 0.0
    plausibility_auroc: float = 0.0
    episodes: int = 0
    skipped: bool = False
    skip_reason: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "task_name": self.task_name,
            "category": self.category,
            "success_rate": self.success_rate,
            "planning_time_ms": self.planning_time_ms,
            "prediction_mse": self.prediction_mse,
            "plausibility_auroc": self.plausibility_auroc,
            "episodes": self.episodes,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "metadata": self.metadata,
        }


class BenchmarkTask(ABC):
    """Abstract base class for benchmark evaluation tasks.

    Subclasses must implement:
        - name: str property
        - category: str property
        - setup(): prepare environment and data
        - evaluate(model): run evaluation and return TaskResult
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this benchmark task."""
        ...

    @property
    @abstractmethod
    def category(self) -> str:
        """Category: navigation, manipulation, control, or games."""
        ...

    @abstractmethod
    def setup(self) -> None:
        """Load environment, data, or any resources needed for evaluation.

        Should raise ImportError if required dependencies are missing.
        """
        ...

    @abstractmethod
    def evaluate(self, model, episodes: int = 50, seed: int = 42) -> TaskResult:
        """Run evaluation against a WorldModel.

        Args:
            model: A WorldModel instance.
            episodes: Number of evaluation episodes.
            seed: Random seed for reproducibility.

        Returns:
            TaskResult with metrics.
        """
        ...

    def _generate_random_observations(
        self,
        n_frames: int,
        image_size: int = 96,
        seed: int = 42,
    ) -> list[np.ndarray]:
        """Generate random observation frames for testing.

        Args:
            n_frames: Number of frames to generate.
            image_size: Height and width of each frame.
            seed: Random seed.

        Returns:
            List of (H, W, 3) float32 arrays in [0, 1].
        """
        rng = np.random.RandomState(seed)
        return [
            rng.rand(image_size, image_size, 3).astype(np.float32)
            for _ in range(n_frames)
        ]

    def _generate_random_actions(
        self,
        n_steps: int,
        action_dim: int = 2,
        seed: int = 42,
        discrete: bool = False,
    ) -> list[np.ndarray]:
        """Generate random actions for testing.

        Args:
            n_steps: Number of action steps.
            action_dim: Dimensionality of continuous actions.
            seed: Random seed.
            discrete: If True, generate integer actions instead.

        Returns:
            List of action arrays.
        """
        rng = np.random.RandomState(seed)
        if discrete:
            return [np.array(rng.randint(0, action_dim)) for _ in range(n_steps)]
        return [
            rng.uniform(-1, 1, size=action_dim).astype(np.float32)
            for _ in range(n_steps)
        ]

    def _time_planning(self, model, current: np.ndarray, goal: np.ndarray) -> float:
        """Time a single planning call and return elapsed ms.

        Args:
            model: WorldModel instance.
            current: Current observation.
            goal: Goal observation.

        Returns:
            Planning time in milliseconds.
        """
        start = time.perf_counter()
        try:
            model.plan(current, goal, max_steps=20, n_candidates=64, n_iterations=3)
        except Exception:
            pass
        elapsed = (time.perf_counter() - start) * 1000.0
        return elapsed
