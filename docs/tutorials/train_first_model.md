# Train your first world model

This tutorial walks through training a world model on Push-T data — from recording data to evaluating the trained model.

## Prerequisites

```bash
pip install worldkit[envs,train]
```

## Step 1: Prepare training data

WorldKit trains on HDF5 files with pixel observations and actions. You can record data from any Gymnasium environment.

### Record from a Gymnasium environment

```python
import gymnasium as gym
from worldkit.data import Recorder

env = gym.make("CartPole-v1", render_mode="rgb_array")
recorder = Recorder(env, output="cartpole_data.h5")
recorder.record(episodes=200, max_steps_per_episode=500)
print("Data saved to cartpole_data.h5")
```

### Or convert existing videos

```python
from worldkit.data import Converter

converter = Converter()
converter.from_video(
    input_dir="./my_videos/",
    output="video_data.h5",
    fps=10,
)
```

### Verify your data

```python
import h5py

with h5py.File("cartpole_data.h5", "r") as f:
    print("Keys:", list(f.keys()))
    for key in f.keys():
        print(f"  {key}: {f[key].shape}")
# Keys: ['pixels', 'actions']
#   pixels: (200, 500, 96, 96, 3)
#   actions: (200, 500, 1)
```

## Step 2: Train the model

### Python API

```python
from worldkit import WorldModel

model = WorldModel.train(
    data="cartpole_data.h5",
    config="base",          # nano, base, large, xl
    epochs=100,
    batch_size=64,
    lr=1e-4,
    lambda_reg=1.0,         # SIGReg weight (the one hyperparameter)
    device="auto",          # auto-selects CUDA > MPS > CPU
    seed=42,
)
```

**Expected output:**

```
Config: base (13M params, latent_dim=192)
Device: mps
Epoch   1/100  loss=2.4312  pred=1.8901  sigreg=0.5411
Epoch  10/100  loss=0.4523  pred=0.3012  sigreg=0.1511
Epoch  50/100  loss=0.0891  pred=0.0534  sigreg=0.0357
Epoch 100/100  loss=0.0234  pred=0.0145  sigreg=0.0089
Training complete in 62.3s
```

### CLI

```bash
worldkit train \
    --data cartpole_data.h5 \
    --config base \
    --epochs 100 \
    --output cartpole_model.wk
```

## Step 3: Save the model

```python
model.save("cartpole_model.wk")
```

The `.wk` file is a self-contained ZIP archive with the model weights, config, and metadata. See the [.wk format spec](../guides/wk_format.md) for details.

## Step 4: Test the model

### Encode an observation

```python
import numpy as np

model = WorldModel.load("cartpole_model.wk")

obs = np.random.rand(96, 96, 3).astype(np.float32)
z = model.encode(obs)
print(f"Latent: {z.shape}")  # torch.Size([192])
```

### Predict future states

```python
actions = [np.array([1.0])] * 10  # 10 "push right" actions
result = model.predict(obs, actions)
print(f"Trajectory: {result.latent_trajectory.shape}")  # (10, 192)
```

### Plan to a goal

```python
current = np.random.rand(96, 96, 3).astype(np.float32)
goal = np.random.rand(96, 96, 3).astype(np.float32)

plan = model.plan(current, goal, max_steps=30)
print(f"Actions: {len(plan.actions)}")
print(f"Cost: {plan.expected_cost:.4f}")
```

### Score plausibility

```python
frames = [np.random.rand(96, 96, 3).astype(np.float32) for _ in range(20)]
score = model.plausibility(frames)
print(f"Plausibility: {score:.3f}")
```

## Step 5: Evaluate with probing

Linear probing reveals what physical properties the latent space has learned.

```python
result = model.probe(
    data="cartpole_data.h5",
    properties=["cart_position", "pole_angle"],
    labels="cartpole_labels.csv",
)
print(result.summary)
```

## Tips

### Choosing a config

| Config | When to use |
|--------|------------|
| `nano` | Quick prototyping, edge deployment, unit tests |
| `base` | Default choice for most environments |
| `large` | Complex 3D environments, rich visual detail |
| `xl` | Multi-object scenes, high-dimensional action spaces |

See [Model configurations guide](../guides/model_configs.md) for more detail.

### Tuning lambda_reg

`lambda_reg` is the only hyperparameter you need to tune. It controls the balance between prediction accuracy and latent space structure:

- **Too low** (< 0.1): risk of representation collapse
- **Default** (1.0): works well for most tasks
- **Too high** (> 10): overly regularized, may hurt prediction quality

### Multi-environment training

Train on data from multiple environments simultaneously:

```python
model = WorldModel.train(
    data=["pusht.h5", "cartpole.h5"],
    config="base",
    action_dim=4,  # max across environments
    epochs=100,
)
```

Actions from environments with smaller action spaces are zero-padded automatically.

## Next steps

- [Plan robot actions](plan_robot_actions.md) — use your trained model for planning
- [Export and deploy](export_deploy.md) — deploy to production
- [Contribute a model](contribute_model.md) — share your model on the Hub
