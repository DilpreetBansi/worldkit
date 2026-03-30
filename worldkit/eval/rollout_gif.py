"""Rollout GIF generator for WorldKit models.

Generates animated GIFs showing a model's predicted latent trajectory
as a 2D point moving over time. Useful for visualizing how a world
model predicts future states given an observation and action sequence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from worldkit.core.model import WorldModel


class RolloutGIFGenerator:
    """Generate animated GIFs of latent-space rollout trajectories.

    Encodes an observation, predicts future latent states from an action
    sequence, projects the trajectory to 2D via PCA, and renders an
    animated GIF with a moving dot tracing the predicted path.

    Args:
        model: A WorldModel instance used for prediction.

    Usage:
        gen = RolloutGIFGenerator(model)
        path = gen.generate(obs, actions, save_to="rollout.gif")
    """

    def __init__(self, model: WorldModel) -> None:
        self._model = model

    def generate(
        self,
        observation: np.ndarray,
        actions: list,
        save_to: str | Path = "rollout.gif",
        fps: int = 10,
        frame_size: int = 256,
    ) -> Path:
        """Generate a rollout GIF from an observation and action sequence.

        Args:
            observation: Initial observation as numpy array (H, W, C).
            actions: List of action arrays, e.g. ``[np.array([0.1, 0.2])]``.
            save_to: Output path for the GIF file.
            fps: Frames per second for the GIF.
            frame_size: Width and height of each GIF frame in pixels.

        Returns:
            Path to the saved GIF file.
        """
        from PIL import Image

        save_to = Path(save_to)

        # Predict latent trajectory
        result = self._model.predict(observation, actions)
        trajectory = result.latent_trajectory.numpy()  # (T, latent_dim)

        # Project to 2D
        points_2d = self._project_to_2d(trajectory)  # (T, 2)

        # Render frames
        frames: list[Image.Image] = []
        for i in range(len(points_2d)):
            frame = self._render_frame(points_2d, i, frame_size)
            frames.append(frame)

        # Save as animated GIF
        duration_ms = int(1000 / fps)
        frames[0].save(
            str(save_to),
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=duration_ms,
        )

        return save_to

    # ─── Internal helpers ───────────────────────────────

    @staticmethod
    def _project_to_2d(trajectory: np.ndarray) -> np.ndarray:
        """Project a latent trajectory to 2D using PCA via SVD.

        Args:
            trajectory: Array of shape (T, latent_dim).

        Returns:
            Array of shape (T, 2).
        """
        # Mean-center
        mean = trajectory.mean(axis=0)
        centered = trajectory - mean

        # Handle degenerate case (all points identical)
        if np.allclose(centered, 0):
            return np.zeros((len(trajectory), 2))

        # SVD to find top-2 principal components
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ vt[:2].T  # (T, 2)

    @staticmethod
    def _render_frame(
        points_2d: np.ndarray,
        current_idx: int,
        frame_size: int,
    ):
        """Render a single GIF frame showing the trajectory and current position.

        Args:
            points_2d: All trajectory points, shape (T, 2).
            current_idx: Index of the current point to highlight.
            frame_size: Width/height of the output image.

        Returns:
            PIL Image.
        """
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", (frame_size, frame_size), "white")
        draw = ImageDraw.Draw(img)

        margin = 30
        usable = frame_size - 2 * margin
        t_total = len(points_2d)

        # Normalize points to fit in the frame
        pts = points_2d.copy()
        p_min = pts.min(axis=0)
        p_max = pts.max(axis=0)
        p_range = p_max - p_min
        p_range[p_range == 0] = 1.0  # avoid division by zero
        pts = (pts - p_min) / p_range  # normalize to [0, 1]
        pts = pts * usable + margin  # scale to frame coords

        # Draw full trajectory line in light gray
        if t_total > 1:
            line_coords = [(float(pts[i, 0]), float(pts[i, 1]))
                           for i in range(t_total)]
            draw.line(line_coords, fill=(200, 200, 200), width=1)

        # Draw small dots at each trajectory point
        for i in range(t_total):
            x, y = float(pts[i, 0]), float(pts[i, 1])
            r = 2
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(180, 180, 180))

        # Draw current position as a larger colored dot
        # Color gradient: blue (start) → red (end)
        if t_total > 1:
            frac = current_idx / (t_total - 1)
        else:
            frac = 0.0
        red = int(255 * frac)
        blue = int(255 * (1 - frac))
        color = (red, 50, blue)

        cx, cy = float(pts[current_idx, 0]), float(pts[current_idx, 1])
        r = 6
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

        # Add frame counter
        label = f"t={current_idx + 1}/{t_total}"
        try:
            font = ImageFont.truetype("arial", 12)
        except (OSError, IOError):
            font = ImageFont.load_default()
        draw.text((5, 5), label, fill=(80, 80, 80), font=font)

        return img
