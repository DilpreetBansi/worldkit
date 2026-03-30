# Planning

WorldKit plans action sequences by searching in latent space using the Cross-Entropy Method (CEM). No rendering or physics engine is needed — the planner "imagines" outcomes using the learned world model.

## `PlanResult`

```python
from worldkit import PlanResult
```

```python
@dataclass
class PlanResult:
    actions: list                           # List of np.ndarray action vectors
    expected_cost: float                    # MSE distance to goal in latent space
    latent_trajectory: torch.Tensor | None  # Predicted latent path
    success_probability: float              # 1 - expected_cost, clipped to [0, 1]
    planning_time_ms: float                 # Wall-clock planning time
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `actions` | `list[np.ndarray]` | Optimized action sequence to reach the goal. |
| `expected_cost` | `float` | MSE distance between the predicted final latent and the goal latent. Lower is better. |
| `latent_trajectory` | `torch.Tensor \| None` | Predicted latent states along the planned path. |
| `success_probability` | `float` | `1 - expected_cost`, clipped to `[0, 1]`. |
| `planning_time_ms` | `float` | Wall-clock time for planning in milliseconds. |

## `HierarchicalPlanResult`

```python
@dataclass
class HierarchicalPlanResult:
    actions: list[np.ndarray]               # Full concatenated action sequence
    subgoals: list[torch.Tensor]            # Latent vectors for each subgoal
    segment_plans: list[PlanResult]         # Individual plan per segment
    total_planning_time_ms: float           # Total wall-clock time
```

## How CEM planning works

The Cross-Entropy Method optimizes action sequences by iteratively sampling and refining:

1. **Sample** N random action sequences (candidates)
2. **Evaluate** each by rolling out the world model in latent space
3. **Score** by MSE distance between the final predicted latent and the goal latent
4. **Select** the top-K (elite) sequences with lowest cost
5. **Refit** the sampling distribution (mean + std) to the elite set
6. **Repeat** for M iterations
7. **Return** the best action sequence found

```
Iteration 1:  [200 random sequences] → evaluate → keep 20 best
Iteration 2:  [200 sequences ~ elite distribution] → evaluate → keep 20 best
...
Iteration 5:  [200 sequences ~ refined distribution] → return best
```

## Basic planning

```python
import numpy as np
from worldkit import WorldModel

model = WorldModel.load("pusht_model.wk")

current = np.array(...)  # (96, 96, 3) current observation
goal = np.array(...)     # (96, 96, 3) goal observation

plan = model.plan(current, goal, max_steps=50)

print(f"Actions: {len(plan.actions)}")
print(f"Cost: {plan.expected_cost:.4f}")
print(f"Success: {plan.success_probability:.2f}")
print(f"Time: {plan.planning_time_ms:.0f}ms")
```

## Tuning CEM parameters

```python
# More candidates = better search, slower
plan = model.plan(
    current, goal,
    max_steps=50,
    n_candidates=500,   # default: 200
    n_elite=50,          # default: 20
    n_iterations=10,     # default: 5
)

# Constrain action bounds
plan = model.plan(
    current, goal,
    action_space={"low": -0.5, "high": 0.5},
)
```

## Model Predictive Control (MPC)

Execute only the first few actions from each plan, then re-plan from the new state. This corrects for prediction drift:

```python
obs = env.reset()
goal = get_goal_image()

done = False
while not done:
    plan = model.plan(obs, goal, max_steps=30)

    # Execute first 5 actions, then re-plan
    for action in plan.actions[:5]:
        obs, reward, done, _, info = env.step(action)
        if done:
            break
```

## Hierarchical planning

For long-horizon tasks, hierarchical planning decomposes the problem into subgoals:

1. Encode current and goal observations into latent vectors
2. Linearly interpolate K subgoal latents between them
3. Plan between each consecutive pair of subgoals using CEM
4. Concatenate all action sequences

```python
result = model.hierarchical_plan(
    current, goal,
    max_subgoals=5,           # 5 intermediate waypoints
    steps_per_subgoal=30,     # 30 actions per segment
    n_candidates=200,
    n_elite=20,
    n_iterations=5,
)

print(f"Total actions: {len(result.actions)}")
print(f"Subgoals: {len(result.subgoals)}")
print(f"Segment costs: {[s.expected_cost for s in result.segment_plans]}")
print(f"Total time: {result.total_planning_time_ms:.0f}ms")
```

This is useful when the goal is far from the current state and a single CEM pass would struggle to find a good path.

## Planning speed

CEM plans in latent space (not pixel space), making it fast:

| Config | Typical planning time | Actions/second |
|--------|----------------------|----------------|
| nano | ~50ms | ~1000 |
| base | ~150ms | ~330 |
| large | ~500ms | ~100 |

Planning time scales with `n_candidates * n_iterations * max_steps`.

## Related

- [WorldModel.plan()](worldmodel.md#plan) — method reference
- [WorldModel.hierarchical_plan()](worldmodel.md#hierarchical_plan) — method reference
- [Prediction](prediction.md) — how latent rollouts work
- [Plan robot actions tutorial](../tutorials/plan_robot_actions.md) — full walkthrough
