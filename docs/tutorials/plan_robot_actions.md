# Plan robot actions

This tutorial shows how to use WorldKit's CEM planner and hierarchical planner for goal-conditioned action planning.

## Prerequisites

```bash
pip install worldkit[envs]
```

A trained model (see [Train your first model](train_first_model.md)) or a pre-trained one:

```python
from worldkit import WorldModel

model = WorldModel.from_hub("DilpreetBansi/pusht")
```

## Basic planning

Given a current observation and a goal observation, `plan()` finds an action sequence that transitions from current to goal in latent space.

```python
import numpy as np

# Load observations (96x96 RGB images)
current = np.array(...)  # current frame
goal = np.array(...)     # target frame

plan = model.plan(current, goal, max_steps=50)

print(f"Found {len(plan.actions)} actions")
print(f"Expected cost: {plan.expected_cost:.4f}")
print(f"Success probability: {plan.success_probability:.2%}")
print(f"Planning time: {plan.planning_time_ms:.0f}ms")
```

## Execute a plan

Apply the planned actions to an environment:

```python
import gymnasium as gym

env = gym.make("YourEnv-v1", render_mode="rgb_array")
obs, _ = env.reset()

plan = model.plan(obs, goal_obs, max_steps=50)

for action in plan.actions:
    obs, reward, done, truncated, info = env.step(action)
    if done:
        print("Goal reached!")
        break
```

## Model Predictive Control (MPC)

For better real-world performance, use MPC — plan, execute a few steps, then re-plan from the new state:

```python
obs, _ = env.reset()
goal = get_goal_image()

replan_every = 5  # re-plan every 5 steps
max_total_steps = 200

for step in range(max_total_steps):
    if step % replan_every == 0:
        plan = model.plan(obs, goal, max_steps=30)

    action_idx = step % replan_every
    if action_idx < len(plan.actions):
        action = plan.actions[action_idx]
    else:
        break

    obs, reward, done, _, info = env.step(action)
    if done:
        print(f"Goal reached at step {step}")
        break
```

MPC corrects for prediction drift — each re-plan uses the actual current state, not the predicted one.

## Tuning CEM parameters

```python
# Default parameters
plan = model.plan(current, goal, max_steps=50)

# Higher quality search (slower)
plan = model.plan(
    current, goal,
    max_steps=50,
    n_candidates=500,   # more candidate sequences (default: 200)
    n_elite=50,          # more elite sequences kept (default: 20)
    n_iterations=10,     # more refinement rounds (default: 5)
)

# Constrained action space
plan = model.plan(
    current, goal,
    max_steps=50,
    action_space={"low": -0.5, "high": 0.5},  # tighter bounds
)
```

### Parameter effects

| Parameter | Increase | Decrease |
|-----------|----------|----------|
| `n_candidates` | Better exploration, slower | Faster, may miss good solutions |
| `n_elite` | Smoother convergence | Faster convergence, may be noisy |
| `n_iterations` | More refined solution | Faster, less optimized |
| `max_steps` | Longer planning horizon | Shorter, more accurate per-step |

## Hierarchical planning

For long-horizon tasks where a single CEM pass struggles, hierarchical planning decomposes the problem:

```python
result = model.hierarchical_plan(
    current, goal,
    max_subgoals=5,           # 5 intermediate waypoints
    steps_per_subgoal=30,     # 30 actions per segment
)

print(f"Total actions: {len(result.actions)}")
print(f"Subgoals: {len(result.subgoals)}")
print(f"Total time: {result.total_planning_time_ms:.0f}ms")

# Inspect individual segments
for i, seg in enumerate(result.segment_plans):
    print(f"  Segment {i}: {len(seg.actions)} actions, cost={seg.expected_cost:.4f}")
```

### How hierarchical planning works

1. Encode current and goal into latent space
2. Interpolate `max_subgoals` intermediate latent waypoints
3. For each consecutive pair of waypoints, run CEM planning
4. Concatenate all action sequences

This is useful when:
- The goal is very different from the current state
- The task requires multiple distinct phases (e.g., pick up, move, place)
- Single CEM planning consistently fails

## Visualize the plan

Generate a GIF showing the predicted trajectory:

```python
path = model.rollout_gif(
    current,
    plan.actions,
    save_to="plan_rollout.gif",
    fps=10,
)
print(f"Saved rollout to {path}")
```

## Compare planning strategies

```python
# Compare different max_steps values
for horizon in [20, 50, 100]:
    plan = model.plan(current, goal, max_steps=horizon)
    print(f"Horizon {horizon}: cost={plan.expected_cost:.4f}, "
          f"time={plan.planning_time_ms:.0f}ms")
```

## Next steps

- [Planning API reference](../api/planning.md) — detailed CEM and hierarchical planning docs
- [Anomaly detection](anomaly_detection.md) — use plausibility scoring
- [Export and deploy](export_deploy.md) — deploy your planning pipeline
