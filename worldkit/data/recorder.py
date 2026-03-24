from __future__ import annotations

"""Record environment interactions to HDF5 format for training."""

from pathlib import Path
from typing import Callable

import h5py
import numpy as np
from tqdm import tqdm


class Recorder:
    """Record Gymnasium environment interactions to HDF5.

    Args:
        env: Gymnasium environment (must have render_mode="rgb_array").
        output: Output HDF5 file path.
    """

    def __init__(self, env, output: str | Path):
        self.env = env
        self.output = Path(output)

    def record(
        self,
        episodes: int = 100,
        policy: str | Callable = "random",
        max_steps_per_episode: int = 500,
    ) -> Path:
        """Record episodes.

        Args:
            episodes: Number of episodes to record.
            policy: "random" for random actions, or a callable(obs) -> action.
            max_steps_per_episode: Max steps per episode.

        Returns:
            Path to saved HDF5 file.
        """
        all_pixels = []
        all_actions = []
        max_len = 0

        for ep in tqdm(range(episodes), desc="Recording episodes"):
            obs, info = self.env.reset()
            frames = []
            actions = []

            frame = self.env.render()
            if frame is not None:
                frames.append(frame)

            for step in range(max_steps_per_episode):
                if policy == "random":
                    action = self.env.action_space.sample()
                else:
                    action = policy(obs)

                obs, reward, terminated, truncated, info = self.env.step(action)
                frame = self.env.render()

                if frame is not None:
                    frames.append(frame)
                actions.append(np.array(action, dtype=np.float32).flatten())

                if terminated or truncated:
                    break

            min_len = min(len(frames), len(actions))
            frames = frames[:min_len]
            actions = actions[:min_len]

            if len(frames) > 0:
                all_pixels.append(np.stack(frames))
                all_actions.append(np.stack(actions))
                max_len = max(max_len, len(frames))

        padded_pixels = []
        padded_actions = []
        for pixels, actions in zip(all_pixels, all_actions):
            T = pixels.shape[0]
            if T < max_len:
                pad_pixels = np.zeros(
                    (max_len - T, *pixels.shape[1:]), dtype=pixels.dtype
                )
                pixels = np.concatenate([pixels, pad_pixels])
                pad_actions = np.zeros(
                    (max_len - T, *actions.shape[1:]), dtype=actions.dtype
                )
                actions = np.concatenate([actions, pad_actions])
            padded_pixels.append(pixels)
            padded_actions.append(actions)

        self.output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.output, "w") as f:
            f.create_dataset(
                "pixels", data=np.stack(padded_pixels), compression="gzip"
            )
            f.create_dataset(
                "actions", data=np.stack(padded_actions), compression="gzip"
            )

        print(f"WorldKit | Recorded {len(all_pixels)} episodes -> {self.output}")
        return self.output
