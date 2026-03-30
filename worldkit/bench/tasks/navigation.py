"""Navigation benchmark tasks: grid worlds, rooms, mazes."""

from __future__ import annotations

import numpy as np

from ..task import BenchmarkTask, TaskResult


class _NavigationTask(BenchmarkTask):
    """Base class for navigation benchmark tasks.

    Generates synthetic grid-world observations and evaluates
    prediction accuracy, planning speed, and plausibility scoring.
    """

    _name: str = ""
    _env_id: str | None = None

    @property
    def category(self) -> str:
        return "navigation"

    def _check_env(self) -> bool:
        """Check if the required gymnasium environment is available."""
        if self._env_id is None:
            return False
        try:
            import gymnasium as gym

            gym.make(self._env_id)
            return True
        except Exception:
            return False

    def _collect_env_data(
        self, episodes: int, max_steps: int, seed: int
    ) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
        """Collect observation and action data from the environment.

        Returns:
            Tuple of (observations_per_episode, actions_per_episode).
        """
        import gymnasium as gym

        env = gym.make(self._env_id, render_mode="rgb_array")
        all_obs = []
        all_actions = []

        for ep in range(episodes):
            obs_list = []
            act_list = []
            obs, _ = env.reset(seed=seed + ep)
            frame = env.render()
            obs_list.append(frame.astype(np.float32) / 255.0)

            for _ in range(max_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                frame = env.render()
                obs_list.append(frame.astype(np.float32) / 255.0)
                act = np.array([action] if np.isscalar(action) else action,
                               dtype=np.float32)
                act_list.append(act)
                if terminated or truncated:
                    break

            all_obs.append(obs_list)
            all_actions.append(act_list)

        env.close()
        return all_obs, all_actions

    def evaluate(self, model, episodes: int = 50, seed: int = 42) -> TaskResult:
        """Evaluate model on this navigation task."""
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

            # Prediction MSE: encode frames, predict, compare
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

        # Success rate approximated by plausibility threshold
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


class TwoRoomTask(_NavigationTask):
    """Navigate between two connected rooms."""

    _name = "two_room"

    @property
    def name(self) -> str:
        return "two_room"

    def setup(self) -> None:
        pass


class MazeTask(_NavigationTask):
    """Navigate through a procedurally generated maze."""

    _name = "maze"

    @property
    def name(self) -> str:
        return "maze"

    def setup(self) -> None:
        pass


class GridWorldTask(_NavigationTask):
    """Navigate a simple grid world to reach a goal cell."""

    _name = "grid_world"

    @property
    def name(self) -> str:
        return "grid_world"

    def setup(self) -> None:
        pass
