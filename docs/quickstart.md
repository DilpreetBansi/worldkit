# Quickstart

Train, predict, and plan with a world model in 5 minutes.

## 1. Install

```bash
pip install worldkit[envs]
```

## 2. Create a model

```python
from worldkit import WorldModel
from worldkit.core.config import get_config
from worldkit.core.jepa import JEPA

config = get_config("nano", action_dim=2)
jepa = JEPA.from_config(config)
model = WorldModel(jepa, config, device="cpu")

print(f"Parameters: {model.num_params:,}")
print(f"Latent dim: {model.latent_dim}")
# Parameters: 3,500,000
# Latent dim: 128
```

## 3. Encode an observation

WorldModel accepts numpy arrays as input and handles normalization internally.

```python
import numpy as np

# Any (H, W, C) image — uint8 or float32
obs = np.random.rand(96, 96, 3).astype(np.float32)
z = model.encode(obs)
print(f"Latent vector: {z.shape}")
# Latent vector: torch.Size([128])
```

## 4. Predict future states

Given a current observation and a sequence of actions, predict how the latent state evolves.

```python
actions = [np.array([0.1, 0.2])] * 10
result = model.predict(obs, actions)

print(f"Trajectory: {result.latent_trajectory.shape}")
print(f"Confidence: {result.confidence}")
print(f"Steps: {result.steps}")
# Trajectory: torch.Size([10, 128])
# Confidence: 0.8
# Steps: 10
```

`result` is a [`PredictionResult`](api/prediction.md) with the predicted latent trajectory.

## 5. Plan to reach a goal

Use CEM (Cross-Entropy Method) to find actions that reach a goal state.

```python
current = np.random.rand(96, 96, 3).astype(np.float32)
goal = np.random.rand(96, 96, 3).astype(np.float32)

plan = model.plan(current, goal, max_steps=20)
print(f"Actions: {len(plan.actions)}")
print(f"Success probability: {plan.success_probability:.2f}")
print(f"Planning time: {plan.planning_time_ms:.0f}ms")
# Actions: 20
# Success probability: 0.45
# Planning time: 150ms
```

`plan` is a [`PlanResult`](api/planning.md) with the optimized action sequence.

## 6. Score plausibility

Detect whether a video sequence is physically plausible.

```python
frames = [np.random.rand(96, 96, 3).astype(np.float32) for _ in range(20)]
score = model.plausibility(frames)
print(f"Plausibility: {score:.3f}")
# Plausibility: 0.234
# (Low score — random frames are not physically plausible)
```

## 7. Train from data

Train a model on your own HDF5 data:

```python
model = WorldModel.train(
    data="my_data.h5",
    config="base",
    epochs=100,
    device="auto",
)
model.save("my_model.wk")
```

Or use the CLI:

```bash
worldkit train --data my_data.h5 --config base --epochs 100 --output my_model.wk
```

## 8. Load a pre-trained model

```python
# From file
model = WorldModel.load("my_model.wk")

# From Hugging Face Hub
model = WorldModel.from_hub("DilpreetBansi/pusht")
```

## 9. Export for deployment

```python
# ONNX
model.export(format="onnx", output="./deploy/")

# TorchScript
model.export(format="torchscript", output="./deploy/")
```

## 10. Serve as REST API

```bash
worldkit serve --model my_model.wk --port 8000
```

Then query:

```bash
curl -X POST http://localhost:8000/encode \
  -F "observation=@frame.png"
```

## Next steps

- [Train your first model](tutorials/train_first_model.md) — full tutorial with Push-T
- [API Reference](api/worldmodel.md) — every method documented
- [Model configurations](guides/model_configs.md) — choosing nano vs base vs large vs xl
- [Data preparation](guides/data_preparation.md) — preparing HDF5 training data
