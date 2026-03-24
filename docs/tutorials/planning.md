# Planning Guide

## How Planning Works

WorldKit uses Cross-Entropy Method (CEM) to find optimal action sequences entirely in latent space.

## Basic Planning

```python
plan = model.plan(
    current_state=current_obs,
    goal_state=goal_obs,
    max_steps=50,
)
```

## MPC (Model Predictive Control)

```python
done = False
while not done:
    plan = model.plan(obs, goal, max_steps=30)
    for action in plan.actions[:5]:
        obs, reward, done, _, info = env.step(action)
        if done:
            break
```
