"""Goal-conditioned planning with a world model."""

from worldkit import WorldModel
from worldkit.core.config import get_config
from worldkit.core.jepa import JEPA
import numpy as np

config = get_config("nano", action_dim=2)
jepa = JEPA.from_config(config)
model = WorldModel(jepa, config, device="cpu")

current_obs = np.random.rand(96, 96, 3).astype(np.float32)
goal_obs = np.random.rand(96, 96, 3).astype(np.float32)

plan = model.plan(
    current_state=current_obs,
    goal_state=goal_obs,
    max_steps=20,
    n_candidates=100,
    n_elite=10,
    n_iterations=3,
)

print(f"Planning completed in {plan.planning_time_ms:.1f}ms")
print(f"Number of actions: {len(plan.actions)}")
print(f"Expected cost: {plan.expected_cost:.4f}")
print(f"Success probability: {plan.success_probability:.3f}")
print(f"First action: {plan.actions[0]}")
