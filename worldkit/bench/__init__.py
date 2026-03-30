"""WorldKit-Bench — the canonical benchmark for world models.

Usage:
    from worldkit.bench import BenchmarkSuite, BenchmarkRunner

    suite = BenchmarkSuite.full()
    runner = BenchmarkRunner(suite, model)
    results = runner.run(episodes_per_task=50)
    results.to_json("results.json")
    results.to_html("report.html")
"""

from .leaderboard import format_leaderboard_entry, save_leaderboard_entry
from .runner import BenchmarkResults, BenchmarkRunner
from .suite import BenchmarkSuite
from .task import BenchmarkTask, TaskResult

__all__ = [
    "BenchmarkResults",
    "BenchmarkRunner",
    "BenchmarkSuite",
    "BenchmarkTask",
    "TaskResult",
    "format_leaderboard_entry",
    "save_leaderboard_entry",
]
