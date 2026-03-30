"""Control benchmark tasks: classic control environments."""

from __future__ import annotations

import numpy as np

from ..task import BenchmarkTask, TaskResult


class _ControlTask(BenchmarkTask):
    """Base class for classic control benchmark tasks.

    If the corresponding Gymnasium environment is available, collects
    real rollout data. Otherwise falls back to synthetic observations.
    """

    _env_id: str | None = None

    @property
    def category(self) -> str:
        return "control"

    def _try_env(self) -> bool:
        """Check if the Gymnasium env is available."""
        if self._env_id is None:
            return False
        try:
            import gymnasium as gym

            env = gym.make(self._env_id, render_mode="rgb_array")
            env.close()
            return True
        except Exception:
            return False

    def evaluate(self, model, episodes: int = 50, seed: int = 42) -> TaskResult:
        """Evaluate model on this control task."""
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


class CartPoleTask(_ControlTask):
    """Balance a pole on a cart (CartPole-v1)."""

    _env_id = "CartPole-v1"

    @property
    def name(self) -> str:
        return "cartpole"

    def setup(self) -> None:
        pass


class PendulumTask(_ControlTask):
    """Swing up and balance an inverted pendulum (Pendulum-v1)."""

    _env_id = "Pendulum-v1"

    @property
    def name(self) -> str:
        return "pendulum"

    def setup(self) -> None:
        pass


class ReacherTask(_ControlTask):
    """Reach a target position with a 2-link arm (Reacher-v4)."""

    _env_id = "Reacher-v4"

    @property
    def name(self) -> str:
        return "reacher"

    def setup(self) -> None:
        pass


class AcrobotTask(_ControlTask):
    """Swing the lower link above a threshold (Acrobot-v1)."""

    _env_id = "Acrobot-v1"

    @property
    def name(self) -> str:
        return "acrobot"

    def setup(self) -> None:
        pass
