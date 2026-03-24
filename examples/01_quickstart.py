"""WorldKit Quickstart — Train, predict, and plan in 5 lines."""

from worldkit import WorldModel
from worldkit.core.config import get_config
from worldkit.core.jepa import JEPA
import numpy as np

config = get_config("nano", action_dim=2)
jepa = JEPA.from_config(config)
model = WorldModel(jepa, config, device="cpu")

print(f"Model: {config.name} | Params: {model.num_params:,} | Latent dim: {model.latent_dim}")

obs = np.random.rand(96, 96, 3).astype(np.float32)
z = model.encode(obs)
print(f"Encoded observation to latent: {z.shape}")

actions = [np.array([0.1, 0.2])] * 10
result = model.predict(obs, actions)
print(f"Predicted {result.steps} steps, trajectory shape: {result.latent_trajectory.shape}")

goal = np.random.rand(96, 96, 3).astype(np.float32)
plan = model.plan(obs, goal, max_steps=10, n_candidates=50, n_iterations=2)
print(f"Planned {len(plan.actions)} actions in {plan.planning_time_ms:.1f}ms")

frames = [np.random.rand(96, 96, 3).astype(np.float32) for _ in range(10)]
score = model.plausibility(frames)
print(f"Plausibility score: {score:.3f}")
