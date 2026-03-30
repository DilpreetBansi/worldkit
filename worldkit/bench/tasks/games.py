"""Game benchmark tasks: Atari-style environments."""

from __future__ import annotations

import numpy as np

from ..task import BenchmarkTask, TaskResult


class _GameTask(BenchmarkTask):
    """Base class for game benchmark tasks.

    Games typically use discrete actions and pixel observations.
    """

    _env_id: str | None = None

    @property
    def category(self) -> str:
        return "games"

    def evaluate(self, model, episodes: int = 50, seed: int = 42) -> TaskResult:
        """Evaluate model on this game task."""
        image_size = model.config.image_size
        action_dim = model.config.action_dim

        prediction_errors = []
        planning_times = []
        plausibility_scores = []

        for ep in range(episodes):
            ep_seed = seed + ep
            frames = self._generate_random_observations(
                n_frames=10, image_size=image_size, seed=ep_seed
            )
            actions = self._generate_random_actions(
                n_steps=5, action_dim=action_dim, seed=ep_seed
            )

            # Prediction MSE
            try:
                result = model.predict(frames[0], actions)
                z_target = model.encode(frames[5])
                mse = float(
                    ((result.latent_trajectory[-1] - z_target) ** 2).mean().item()
                )
                prediction_errors.append(mse)
            except Exception:
                prediction_errors.append(float("nan"))

            # Planning time
            planning_times.append(
                self._time_planning(model, frames[0], frames[-1])
            )

            # Plausibility
            try:
                score = model.plausibility(frames)
                plausibility_scores.append(score)
            except Exception:
                plausibility_scores.append(0.0)

        avg_mse = float(np.nanmean(prediction_errors))
        avg_plan_ms = float(np.mean(planning_times))
        avg_plaus = float(np.mean(plausibility_scores))
        success_rate = float(np.mean([s > 0.3 for s in plausibility_scores]))

        return TaskResult(
            task_name=self.name,
            category=self.category,
            success_rate=success_rate,
            planning_time_ms=avg_plan_ms,
            prediction_mse=avg_mse,
            plausibility_auroc=avg_plaus,
            episodes=episodes,
        )


class PongTask(_GameTask):
    """Predict and plan in Pong."""

    _env_id = "ALE/Pong-v5"

    @property
    def name(self) -> str:
        return "pong"

    def setup(self) -> None:
        pass


class BreakoutTask(_GameTask):
    """Predict and plan in Breakout."""

    _env_id = "ALE/Breakout-v5"

    @property
    def name(self) -> str:
        return "breakout"

    def setup(self) -> None:
        pass
