"""Report generation for WorldKit-Bench results."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runner import BenchmarkResults


def generate_html_report(results: BenchmarkResults) -> str:
    """Generate a self-contained HTML report from benchmark results.

    Args:
        results: BenchmarkResults from a benchmark run.

    Returns:
        HTML string for the complete report.
    """
    summary = results.summary()

    # Build per-category aggregates
    categories: dict[str, list] = {}
    for r in results.results:
        categories.setdefault(r.category, []).append(r)

    category_rows = ""
    for cat in sorted(categories):
        tasks = categories[cat]
        active = [t for t in tasks if not t.skipped]
        if active:
            avg_sr = sum(t.success_rate for t in active) / len(active)
            avg_mse = sum(t.prediction_mse for t in active) / len(active)
            avg_plan = sum(t.planning_time_ms for t in active) / len(active)
        else:
            avg_sr = avg_mse = avg_plan = 0.0
        category_rows += (
            f"<tr>"
            f"<td>{cat}</td>"
            f"<td>{len(active)}/{len(tasks)}</td>"
            f"<td>{avg_sr:.2%}</td>"
            f"<td>{avg_mse:.4f}</td>"
            f"<td>{avg_plan:.1f}</td>"
            f"</tr>\n"
        )

    # Build per-task rows
    task_rows = ""
    for r in results.results:
        if r.skipped:
            task_rows += (
                f"<tr class='skipped'>"
                f"<td>{r.task_name}</td>"
                f"<td>{r.category}</td>"
                f"<td colspan='4'>Skipped: {r.skip_reason}</td>"
                f"</tr>\n"
            )
        else:
            task_rows += (
                f"<tr>"
                f"<td>{r.task_name}</td>"
                f"<td>{r.category}</td>"
                f"<td>{r.success_rate:.2%}</td>"
                f"<td>{r.prediction_mse:.4f}</td>"
                f"<td>{r.planning_time_ms:.1f}</td>"
                f"<td>{r.plausibility_auroc:.3f}</td>"
                f"</tr>\n"
            )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WorldKit-Bench Report</title>
<style>
  :root {{
    --bg: #0f1117;
    --card: #1a1d27;
    --border: #2a2d3a;
    --text: #e4e4e7;
    --muted: #9ca3af;
    --accent: #6366f1;
    --accent-light: #818cf8;
    --green: #22c55e;
    --red: #ef4444;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem;
  }}
  h1 {{
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, var(--accent), var(--accent-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }}
  .subtitle {{ color: var(--muted); margin-bottom: 2rem; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }}
  .stat-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem;
  }}
  .stat-card .label {{ color: var(--muted); font-size: 0.85rem; }}
  .stat-card .value {{
    font-size: 1.5rem;
    font-weight: 700;
    margin-top: 0.3rem;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 2rem;
    background: var(--card);
    border-radius: 8px;
    overflow: hidden;
  }}
  th {{
    background: var(--border);
    text-align: left;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
  }}
  td {{ padding: 0.6rem 1rem; border-bottom: 1px solid var(--border); }}
  tr:last-child td {{ border-bottom: none; }}
  .skipped td {{ color: var(--muted); font-style: italic; }}
  h2 {{
    font-size: 1.2rem;
    margin-bottom: 1rem;
    color: var(--accent-light);
  }}
  footer {{
    text-align: center;
    color: var(--muted);
    font-size: 0.8rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>
<h1>WorldKit-Bench Report</h1>
<p class="subtitle">
  Suite: <strong>{summary['suite']}</strong> |
  Model: <strong>{summary['model_name']}</strong>
  ({summary['model_params']:,} params)
</p>

<div class="grid">
  <div class="stat-card">
    <div class="label">Tasks Evaluated</div>
    <div class="value">{summary['evaluated']}/{summary['total_tasks']}</div>
  </div>
  <div class="stat-card">
    <div class="label">Avg Success Rate</div>
    <div class="value">{summary['avg_success_rate']:.2%}</div>
  </div>
  <div class="stat-card">
    <div class="label">Avg Prediction MSE</div>
    <div class="value">{summary['avg_prediction_mse']:.4f}</div>
  </div>
  <div class="stat-card">
    <div class="label">Avg Planning Time</div>
    <div class="value">{summary['avg_planning_time_ms']:.1f}ms</div>
  </div>
  <div class="stat-card">
    <div class="label">Total Time</div>
    <div class="value">{summary['total_time_s']:.1f}s</div>
  </div>
</div>

<h2>Results by Category</h2>
<table>
<thead>
  <tr><th>Category</th><th>Tasks</th><th>Success Rate</th><th>Pred MSE</th>
  <th>Plan Time (ms)</th></tr>
</thead>
<tbody>
{category_rows}
</tbody>
</table>

<h2>Results by Task</h2>
<table>
<thead>
  <tr><th>Task</th><th>Category</th><th>Success Rate</th><th>Pred MSE</th>
  <th>Plan Time (ms)</th><th>Plausibility</th></tr>
</thead>
<tbody>
{task_rows}
</tbody>
</table>

<footer>
  Generated by WorldKit-Bench v0.1.0
</footer>
</body>
</html>"""
    return html
