"""Leaderboard submission formatting for WorldKit-Bench."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runner import BenchmarkResults


def format_leaderboard_entry(results: BenchmarkResults) -> dict:
    """Format benchmark results as a leaderboard submission entry.

    Args:
        results: BenchmarkResults from a completed benchmark run.

    Returns:
        Dictionary ready for JSON submission to the leaderboard.
    """
    summary = results.summary()

    # Per-category scores
    category_scores: dict[str, dict] = {}
    for r in results.results:
        if r.skipped:
            continue
        cat = r.category
        if cat not in category_scores:
            category_scores[cat] = {
                "success_rates": [],
                "prediction_mses": [],
                "planning_times_ms": [],
            }
        category_scores[cat]["success_rates"].append(r.success_rate)
        category_scores[cat]["prediction_mses"].append(r.prediction_mse)
        category_scores[cat]["planning_times_ms"].append(r.planning_time_ms)

    category_summary = {}
    for cat, scores in sorted(category_scores.items()):
        n = len(scores["success_rates"])
        category_summary[cat] = {
            "tasks_evaluated": n,
            "avg_success_rate": round(
                sum(scores["success_rates"]) / n, 4
            ),
            "avg_prediction_mse": round(
                sum(scores["prediction_mses"]) / n, 6
            ),
            "avg_planning_time_ms": round(
                sum(scores["planning_times_ms"]) / n, 2
            ),
        }

    return {
        "worldkit_bench_version": "0.1.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": summary["model_name"],
            "params": summary["model_params"],
        },
        "suite": summary["suite"],
        "overall": {
            "tasks_evaluated": summary["evaluated"],
            "tasks_skipped": summary["skipped"],
            "avg_success_rate": summary["avg_success_rate"],
            "avg_prediction_mse": summary["avg_prediction_mse"],
            "avg_planning_time_ms": summary["avg_planning_time_ms"],
            "total_time_s": summary["total_time_s"],
        },
        "categories": category_summary,
        "per_task": [r.to_dict() for r in results.results if not r.skipped],
    }


def save_leaderboard_entry(
    results: BenchmarkResults,
    path: str | Path,
) -> None:
    """Save a leaderboard submission to a JSON file.

    Args:
        results: BenchmarkResults from a completed benchmark run.
        path: Output file path for the submission JSON.
    """
    entry = format_leaderboard_entry(results)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entry, f, indent=2)
