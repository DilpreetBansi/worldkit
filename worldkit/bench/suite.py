"""BenchmarkSuite — manages collections of evaluation tasks."""

from __future__ import annotations

from .task import BenchmarkTask
from .tasks import ALL_TASKS


class BenchmarkSuite:
    """A collection of benchmark tasks to evaluate a world model against.

    Usage:
        suite = BenchmarkSuite.full()           # all tasks
        suite = BenchmarkSuite.category("control")  # one category
        suite = BenchmarkSuite.quick()           # 5 fast tasks for dev
    """

    def __init__(self, tasks: list[BenchmarkTask], name: str = "custom"):
        """Initialize a benchmark suite.

        Args:
            tasks: List of BenchmarkTask instances.
            name: Human-readable name for this suite.
        """
        self._tasks = tasks
        self._name = name

    @property
    def name(self) -> str:
        """Suite name."""
        return self._name

    @property
    def tasks(self) -> list[BenchmarkTask]:
        """List of tasks in this suite."""
        return list(self._tasks)

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self):
        return iter(self._tasks)

    @classmethod
    def full(cls) -> BenchmarkSuite:
        """Create a suite with all available benchmark tasks."""
        tasks = [task_cls() for task_cls in ALL_TASKS]
        return cls(tasks, name="full")

    @classmethod
    def category(cls, cat: str) -> BenchmarkSuite:
        """Create a suite with tasks from a single category.

        Args:
            cat: Category name (navigation, manipulation, control, games).

        Returns:
            BenchmarkSuite containing only tasks in the given category.

        Raises:
            ValueError: If no tasks match the category.
        """
        tasks = [task_cls() for task_cls in ALL_TASKS if task_cls().category == cat]
        if not tasks:
            valid = sorted({task_cls().category for task_cls in ALL_TASKS})
            raise ValueError(
                f"No tasks found for category '{cat}'. "
                f"Valid categories: {valid}"
            )
        return cls(tasks, name=cat)

    @classmethod
    def quick(cls) -> BenchmarkSuite:
        """Create a small suite of 5 fast tasks for development.

        Includes one task from each category plus an extra control task.
        """
        from .tasks.control import CartPoleTask, PendulumTask
        from .tasks.games import PongTask
        from .tasks.manipulation import PushTTask
        from .tasks.navigation import TwoRoomTask

        tasks = [
            TwoRoomTask(),
            PushTTask(),
            CartPoleTask(),
            PendulumTask(),
            PongTask(),
        ]
        return cls(tasks, name="quick")

    def categories(self) -> list[str]:
        """Return sorted list of unique categories in this suite."""
        return sorted({t.category for t in self._tasks})
