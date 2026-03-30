"""Tests for WorldKit-Bench benchmark framework."""

import json

import pytest

from worldkit import WorldModel
from worldkit.bench import (
    BenchmarkResults,
    BenchmarkRunner,
    BenchmarkSuite,
    TaskResult,
)
from worldkit.bench.leaderboard import format_leaderboard_entry
from worldkit.core.config import get_config
from worldkit.core.jepa import JEPA


@pytest.fixture
def model():
    """Create a nano test model."""
    config = get_config("nano", action_dim=2)
    jepa = JEPA.from_config(config)
    return WorldModel(jepa, config, device="cpu")


# ─── Suite creation ──────────────────────────────────


def test_suite_full():
    suite = BenchmarkSuite.full()
    assert len(suite) == 11
    assert suite.name == "full"
    assert "navigation" in suite.categories()
    assert "control" in suite.categories()


def test_suite_category():
    suite = BenchmarkSuite.category("control")
    assert suite.name == "control"
    assert all(t.category == "control" for t in suite)
    assert len(suite) == 4


def test_suite_category_invalid():
    with pytest.raises(ValueError, match="No tasks found"):
        BenchmarkSuite.category("nonexistent")


def test_suite_quick():
    suite = BenchmarkSuite.quick()
    assert len(suite) == 5
    assert suite.name == "quick"


def test_suite_iteration():
    suite = BenchmarkSuite.quick()
    names = [t.name for t in suite]
    assert "two_room" in names
    assert "cartpole" in names


# ─── Runner ──────────────────────────────────────────


def test_runner_with_nano_model(model):
    suite = BenchmarkSuite.quick()
    runner = BenchmarkRunner(suite, model)
    results = runner.run(episodes_per_task=2, seed=42, verbose=False)

    assert isinstance(results, BenchmarkResults)
    assert results.suite_name == "quick"
    assert results.model_name == "nano"
    assert results.num_tasks > 0
    assert results.total_time_s > 0

    for r in results.results:
        if not r.skipped:
            assert 0.0 <= r.success_rate <= 1.0
            assert r.planning_time_ms >= 0
            assert r.episodes == 2


def test_runner_summary(model):
    suite = BenchmarkSuite.quick()
    runner = BenchmarkRunner(suite, model)
    results = runner.run(episodes_per_task=2, seed=42, verbose=False)
    summary = results.summary()

    assert "suite" in summary
    assert "model_name" in summary
    assert "evaluated" in summary
    assert summary["evaluated"] + summary["skipped"] == summary["total_tasks"]


# ─── Report generation ───────────────────────────────


def test_report_json(model, tmp_path):
    suite = BenchmarkSuite.quick()
    runner = BenchmarkRunner(suite, model)
    results = runner.run(episodes_per_task=2, seed=42, verbose=False)

    json_path = tmp_path / "results.json"
    results.to_json(json_path)
    assert json_path.exists()

    with open(json_path) as f:
        data = json.load(f)
    assert "summary" in data
    assert "results" in data
    assert len(data["results"]) == len(results.results)


def test_report_html(model, tmp_path):
    suite = BenchmarkSuite.quick()
    runner = BenchmarkRunner(suite, model)
    results = runner.run(episodes_per_task=2, seed=42, verbose=False)

    html_path = tmp_path / "report.html"
    results.to_html(html_path)
    assert html_path.exists()

    content = html_path.read_text()
    assert "WorldKit-Bench Report" in content
    assert "nano" in content


def test_results_roundtrip_json(model, tmp_path):
    suite = BenchmarkSuite.quick()
    runner = BenchmarkRunner(suite, model)
    results = runner.run(episodes_per_task=2, seed=42, verbose=False)

    json_path = tmp_path / "results.json"
    results.to_json(json_path)

    loaded = BenchmarkResults.from_json(json_path)
    assert loaded.suite_name == results.suite_name
    assert loaded.model_name == results.model_name
    assert len(loaded.results) == len(results.results)


# ─── Leaderboard ─────────────────────────────────────


def test_leaderboard_entry(model):
    suite = BenchmarkSuite.quick()
    runner = BenchmarkRunner(suite, model)
    results = runner.run(episodes_per_task=2, seed=42, verbose=False)

    entry = format_leaderboard_entry(results)
    assert entry["worldkit_bench_version"] == "0.1.0"
    assert entry["model"]["name"] == "nano"
    assert "overall" in entry
    assert "categories" in entry
    assert "per_task" in entry


# ─── Task result ─────────────────────────────────────


def test_task_result_to_dict():
    r = TaskResult(
        task_name="test_task",
        category="control",
        success_rate=0.75,
        planning_time_ms=12.5,
        prediction_mse=0.01,
        plausibility_auroc=0.9,
        episodes=10,
    )
    d = r.to_dict()
    assert d["task_name"] == "test_task"
    assert d["success_rate"] == 0.75
    assert d["skipped"] is False


# ─── CLI bench command ───────────────────────────────


def test_cli_bench_command():
    from click.testing import CliRunner

    from worldkit.cli.main import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["bench", "--help"])
    assert result.exit_code == 0
    assert "evaluate" in result.output.lower() or "bench" in result.output.lower()


def test_cli_bench_report(model, tmp_path):
    from click.testing import CliRunner

    from worldkit.cli.main import cli

    # First generate results
    suite = BenchmarkSuite.quick()
    bench_runner = BenchmarkRunner(suite, model)
    results = bench_runner.run(episodes_per_task=2, seed=42, verbose=False)
    json_path = tmp_path / "results.json"
    results.to_json(json_path)

    # Then test CLI report generation
    html_path = tmp_path / "report.html"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["bench", "report", "--results", str(json_path), "--output", str(html_path)]
    )
    assert result.exit_code == 0
    assert html_path.exists()
