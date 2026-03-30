"""Model comparison tools for WorldKit.

Compare multiple WorldModel instances on the same dataset, computing
prediction error, plausibility, encoding speed, and latent variance.
Generates a self-contained HTML report with a comparison table.
"""

from __future__ import annotations

import html
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from worldkit.core.model import WorldModel


@dataclass
class ComparisonResult:
    """Result from comparing multiple world models.

    Attributes:
        model_names: List of model names (keys from the input dict).
        metrics: Nested dict — outer key is model name, inner dict maps
            metric name to value.
        best_model: Name of the model with lowest prediction error.
    """

    model_names: list[str]
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    best_model: str = ""


class ModelComparator:
    """Compare multiple world models on the same evaluation dataset.

    Computes per-model metrics including prediction error, plausibility,
    encoding speed, and latent variance, then optionally generates an
    HTML report.

    Args:
        models: Dict mapping model names to WorldModel instances.

    Usage:
        comparator = ModelComparator({"base": model_a, "large": model_b})
        result = comparator.compare("eval_data.h5", episodes=50)
        comparator.report(result, save_to="comparison.html")
    """

    def __init__(self, models: dict[str, WorldModel]) -> None:
        if len(models) < 2:
            raise ValueError(
                "ModelComparator requires at least 2 models to compare."
            )
        self._models = models

    def compare(
        self,
        data_path: str | Path,
        episodes: int = 50,
    ) -> ComparisonResult:
        """Run comparison across all models on the given dataset.

        Args:
            data_path: Path to HDF5 file with pixel observations and actions.
            episodes: Maximum number of episodes to evaluate.

        Returns:
            ComparisonResult with per-model metrics.
        """
        pixels, actions, n_eps, ep_len = self._load_episodes(
            data_path, episodes
        )

        model_names = list(self._models.keys())
        metrics: dict[str, dict[str, float]] = {}

        for name, model in self._models.items():
            pred_error = self._compute_prediction_error(
                model, pixels, actions, n_eps, ep_len
            )
            plaus_score = self._compute_plausibility(
                model, pixels, n_eps, ep_len
            )
            enc_time = self._compute_encoding_time(model, pixels)
            lat_var = self._compute_latent_variance(model, pixels)

            metrics[name] = {
                "prediction_error": pred_error,
                "plausibility_score": plaus_score,
                "encoding_time_ms": enc_time,
                "latent_variance": lat_var,
                "num_params": float(model.num_params),
                "latent_dim": float(model.latent_dim),
            }

        # Best model = lowest prediction error
        best = min(
            model_names,
            key=lambda n: metrics[n]["prediction_error"],
        )

        return ComparisonResult(
            model_names=model_names,
            metrics=metrics,
            best_model=best,
        )

    def report(
        self,
        result: ComparisonResult,
        save_to: str | Path = "comparison.html",
    ) -> Path:
        """Generate an HTML comparison report.

        Args:
            result: A ComparisonResult from ``compare()``.
            save_to: Output path for the HTML file.

        Returns:
            Path to the saved HTML file.
        """
        save_to = Path(save_to)

        # Build table rows
        metric_keys = [
            "prediction_error",
            "plausibility_score",
            "encoding_time_ms",
            "latent_variance",
            "num_params",
            "latent_dim",
        ]
        metric_labels = {
            "prediction_error": "Prediction Error (MSE)",
            "plausibility_score": "Plausibility Score",
            "encoding_time_ms": "Encoding Time (ms)",
            "latent_variance": "Latent Variance",
            "num_params": "Parameters",
            "latent_dim": "Latent Dim",
        }

        header_cells = "".join(
            f"<th>{html.escape(n)}</th>" for n in result.model_names
        )

        rows = ""
        for key in metric_keys:
            label = metric_labels.get(key, key)
            cells = ""
            for name in result.model_names:
                val = result.metrics[name][key]
                if key == "num_params":
                    formatted = f"{int(val):,}"
                elif key == "latent_dim":
                    formatted = str(int(val))
                else:
                    formatted = f"{val:.4f}"

                highlight = ""
                if name == result.best_model:
                    highlight = ' class="best"'
                cells += f"<td{highlight}>{formatted}</td>"
            rows += f"<tr><td class='metric'>{html.escape(label)}</td>"
            rows += f"{cells}</tr>\n"

        report_html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WorldKit Model Comparison</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 Roboto, sans-serif;
    max-width: 900px;
    margin: 40px auto;
    padding: 0 20px;
    color: #333;
  }}
  h1 {{ color: #1a1a2e; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 24px 0;
  }}
  th, td {{
    padding: 10px 14px;
    text-align: right;
    border-bottom: 1px solid #e0e0e0;
  }}
  th {{
    background: #f5f5f5;
    font-weight: 600;
  }}
  td.metric {{
    text-align: left;
    font-weight: 500;
  }}
  .best {{
    background: #e8f5e9;
    font-weight: 600;
  }}
  .winner {{
    margin-top: 20px;
    padding: 12px 16px;
    background: #e8f5e9;
    border-left: 4px solid #4caf50;
    border-radius: 4px;
  }}
  footer {{
    margin-top: 40px;
    color: #999;
    font-size: 0.85em;
  }}
</style>
</head>
<body>
<h1>WorldKit Model Comparison</h1>
<table>
  <thead>
    <tr><th>Metric</th>{header_cells}</tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
<div class="winner">
  Best model (lowest prediction error):
  <strong>{html.escape(result.best_model)}</strong>
</div>
<footer>Generated by WorldKit</footer>
</body>
</html>
"""
        save_to.write_text(report_html)
        return save_to

    # ─── Internal helpers ───────────────────────────────

    @staticmethod
    def _load_episodes(
        data_path: str | Path,
        max_episodes: int,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Load pixel and action data from an HDF5 file.

        Returns:
            Tuple of (pixels, actions, n_episodes, episode_length) where
            pixels is (N_eps, T, H, W, C) and actions is (N_eps, T, D).
        """
        import h5py

        data_path = Path(data_path)
        with h5py.File(data_path, "r") as f:
            pixels_key = None
            for key in ("pixels", "observations", "obs", "images"):
                if key in f:
                    pixels_key = key
                    break
            if pixels_key is None:
                found = list(f.keys())
                raise KeyError(
                    f"No pixel data found in HDF5 file. "
                    f"Expected one of: 'pixels', 'observations', "
                    f"'obs', 'images'. Found keys: {found}"
                )

            actions_key = None
            for key in ("actions", "action"):
                if key in f:
                    actions_key = key
                    break
            if actions_key is None:
                found = list(f.keys())
                raise KeyError(
                    f"No action data found in HDF5 file. "
                    f"Expected one of: 'actions', 'action'. "
                    f"Found keys: {found}"
                )

            pixels = np.array(f[pixels_key])
            actions = np.array(f[actions_key])

        # Ensure 5D: (N_eps, T, H, W, C)
        if pixels.ndim == 4:
            pixels = pixels[np.newaxis]
            actions = actions[np.newaxis]

        # Cap episodes
        n_eps = min(pixels.shape[0], max_episodes)
        pixels = pixels[:n_eps]
        actions = actions[:n_eps]
        ep_len = pixels.shape[1]

        return pixels, actions, n_eps, ep_len

    @staticmethod
    def _compute_prediction_error(
        model: WorldModel,
        pixels: np.ndarray,
        actions: np.ndarray,
        n_eps: int,
        ep_len: int,
    ) -> float:
        """Compute average prediction MSE over episodes.

        For each episode, encodes the first frame, predicts future latents
        using the action sequence, encodes actual future frames, and
        computes MSE between predicted and actual latent trajectories.
        """
        errors: list[float] = []
        model._model.eval()

        with torch.no_grad():
            for ep in range(n_eps):
                first_frame = pixels[ep, 0]
                ep_actions = [actions[ep, t] for t in range(ep_len)]

                result = model.predict(first_frame, ep_actions)
                pred_traj = result.latent_trajectory  # (T, D)

                # Encode actual frames for comparison
                actual_latents = []
                for t in range(min(len(ep_actions), pred_traj.shape[0])):
                    z = model.encode(pixels[ep, t])
                    actual_latents.append(z)
                if not actual_latents:
                    continue
                actual_traj = torch.stack(actual_latents)  # (T, D)

                n_steps = min(pred_traj.shape[0], actual_traj.shape[0])
                mse = (
                    (pred_traj[:n_steps] - actual_traj[:n_steps]) ** 2
                ).mean().item()
                errors.append(mse)

        return float(np.mean(errors)) if errors else 0.0

    @staticmethod
    def _compute_plausibility(
        model: WorldModel,
        pixels: np.ndarray,
        n_eps: int,
        ep_len: int,
    ) -> float:
        """Compute average plausibility score over episodes."""
        scores: list[float] = []
        for ep in range(n_eps):
            frames = [pixels[ep, t] for t in range(min(ep_len, 10))]
            score = model.plausibility(frames)
            scores.append(score)
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def _compute_encoding_time(
        model: WorldModel,
        pixels: np.ndarray,
    ) -> float:
        """Compute average encoding time in milliseconds per frame."""
        # Sample up to 50 frames
        flat = pixels.reshape(-1, *pixels.shape[2:])
        n_sample = min(50, len(flat))
        indices = np.linspace(0, len(flat) - 1, n_sample, dtype=int)

        start = time.perf_counter()
        for idx in indices:
            model.encode(flat[idx])
        elapsed = time.perf_counter() - start

        return (elapsed / n_sample) * 1000  # ms per frame

    @staticmethod
    def _compute_latent_variance(
        model: WorldModel,
        pixels: np.ndarray,
    ) -> float:
        """Compute average variance of latent vectors across frames.

        Higher variance indicates richer, less collapsed representations.
        """
        flat = pixels.reshape(-1, *pixels.shape[2:])
        n_sample = min(100, len(flat))
        indices = np.linspace(0, len(flat) - 1, n_sample, dtype=int)

        latents: list[np.ndarray] = []
        model._model.eval()

        with torch.no_grad():
            for idx in indices:
                z = model.encode(flat[idx])
                latents.append(z.numpy())

        latent_array = np.stack(latents)  # (N, latent_dim)
        return float(latent_array.var(axis=0).mean())
