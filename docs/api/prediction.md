# Prediction

WorldKit predicts future states entirely in latent space. Given a current observation and a sequence of actions, the predictor rolls out future latent states autoregressively.

## `PredictionResult`

```python
from worldkit import PredictionResult
```

```python
@dataclass
class PredictionResult:
    latent_trajectory: torch.Tensor  # (T, latent_dim)
    confidence: float                 # Prediction confidence
    steps: int                        # Number of steps predicted
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `latent_trajectory` | `torch.Tensor` | Predicted latent states, shape `(T, latent_dim)`. Each row is the predicted latent state at that timestep. |
| `confidence` | `float` | Confidence score for the prediction. |
| `steps` | `int` | Number of timesteps predicted. |

## How prediction works

1. The current observation is encoded into a latent vector `z` via the ViT encoder.
2. Each action is encoded via the action encoder.
3. The predictor (autoregressive transformer with AdaLN-Zero conditioning) takes `z` and the action embedding, and predicts the next latent state `z'`.
4. For multi-step predictions, the predicted `z'` becomes the input for the next step.

```
obs → [Encoder] → z₀ ─┐
                       ├── [Predictor(z₀, a₀)] → z₁
                       │                          │
                       │   [Predictor(z₁, a₁)] → z₂
                       │                          │
                       │   [Predictor(z₂, a₂)] → z₃
                       └── latent_trajectory = [z₁, z₂, z₃]
```

## Basic prediction

```python
import numpy as np
from worldkit import WorldModel

model = WorldModel.load("my_model.wk")

obs = np.random.rand(96, 96, 3).astype(np.float32)
actions = [np.array([0.1, -0.2])] * 10

result = model.predict(obs, actions)
print(result.latent_trajectory.shape)  # torch.Size([10, 192])
print(result.confidence)               # 0.8
print(result.steps)                    # 10
```

## Repeating a single action

Pass `steps` to repeat one action multiple times:

```python
# Predict 20 steps of the same action
result = model.predict(
    obs,
    actions=[np.array([0.5, 0.0])],
    steps=20,
)
print(result.steps)  # 20
```

## Using predictions

Latent trajectories are useful for:

**Comparing action sequences** — predict outcomes of different actions and pick the best:

```python
results = []
for action_seq in candidate_actions:
    result = model.predict(obs, action_seq)
    results.append(result)

# Pick the trajectory closest to a goal
goal_z = model.encode(goal_obs)
best = min(results, key=lambda r: (r.latent_trajectory[-1] - goal_z).norm())
```

**Monitoring drift** — track how predictions diverge over time:

```python
result = model.predict(obs, actions)
for t in range(result.steps):
    z_t = result.latent_trajectory[t]
    # Compare to actual latent at time t
```

**Visualization** — generate rollout GIFs of predicted trajectories:

```python
model.rollout_gif(obs, actions, save_to="rollout.gif")
```

## Related

- [WorldModel.predict()](worldmodel.md#predict) — method reference
- [Planning](planning.md) — CEM uses prediction internally to evaluate action candidates
- [Plausibility](plausibility.md) — prediction errors drive plausibility scoring
