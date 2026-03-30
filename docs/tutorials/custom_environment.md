# Custom environments

This tutorial shows how to bring your own environment to WorldKit — record data, train a model, and register the environment.

## Prerequisites

```bash
pip install worldkit[envs]
```

## Option 1: Gymnasium environment

If your environment follows the Gymnasium API, use the `Recorder` directly:

```python
import gymnasium as gym
from worldkit.data import Recorder

env = gym.make("YourEnv-v1", render_mode="rgb_array")
recorder = Recorder(env, output="your_env_data.h5")
recorder.record(episodes=200, max_steps_per_episode=500)
```

Requirements:
- `render_mode="rgb_array"` — the recorder captures pixel observations from `env.render()`
- Standard Gymnasium API (`reset()`, `step()`, `action_space`)

### With a custom policy

Record using a trained policy instead of random actions:

```python
def my_policy(obs):
    # Your policy logic here
    return env.action_space.sample()

recorder.record(episodes=200, policy=my_policy)
```

## Option 2: Non-Gymnasium environment

For environments that don't follow the Gymnasium API, create an HDF5 file manually:

```python
import h5py
import numpy as np

episodes = 100
max_steps = 200
image_size = 96
action_dim = 3

# Collect your data
all_pixels = []
all_actions = []

for ep in range(episodes):
    env = YourCustomEnv()
    pixels = []
    actions = []

    obs = env.reset()
    for step in range(max_steps):
        frame = env.render()  # (H, W, 3) RGB image
        frame = resize_to(frame, image_size)  # resize to 96x96
        action = get_action(obs)

        pixels.append(frame)
        actions.append(action)

        obs = env.step(action)

    all_pixels.append(np.stack(pixels))
    all_actions.append(np.stack(actions))

# Save to HDF5
with h5py.File("custom_data.h5", "w") as f:
    f.create_dataset("pixels", data=np.stack(all_pixels))    # (N, T, H, W, C)
    f.create_dataset("actions", data=np.stack(all_actions))  # (N, T, action_dim)
```

### Data requirements

| Field | Shape | Type | Notes |
|-------|-------|------|-------|
| `pixels` | `(N, T, H, W, C)` | uint8 or float32 | RGB images. Any resolution (resized internally to `image_size`). |
| `actions` | `(N, T, action_dim)` | float32 | Continuous actions. For discrete, use one-hot or integer encoding. |

Key names: WorldKit auto-detects `pixels`, `observations`, `obs`, or `images` for frames, and `actions` or `action` for actions.

## Option 3: From video files

If you have videos (no action labels), use the `Converter`:

```python
from worldkit.data import Converter

converter = Converter()
converter.from_video(
    input_dir="./my_videos/",
    output="video_data.h5",
    fps=10,
)
```

With action labels:

```python
converter.from_video(
    input_dir="./my_videos/",
    output="video_data.h5",
    fps=10,
    action_labels="actions.csv",
)
```

Action CSV format:
```csv
frame_idx,action_0,action_1,action_2
0,0.1,0.2,0.3
1,0.15,0.18,0.25
...
```

## Train on your data

```python
from worldkit import WorldModel

model = WorldModel.train(
    data="custom_data.h5",
    config="base",
    action_dim=3,   # match your action space
    epochs=100,
    device="auto",
)
model.save("custom_model.wk")
```

## Register your environment

Register your environment so it appears in `worldkit env list`:

```python
from worldkit.envs import register

register(
    env_id="worldkit/my-env",
    display_name="My Custom Environment",
    category="manipulation",           # navigation, manipulation, control, games, simulation
    gym_id="MyEnv-v1",                 # optional Gymnasium ID
    action_dim=3,
    action_type="continuous",          # continuous or discrete
    action_low=-1.0,
    action_high=1.0,
    observation_shape=(96, 96, 3),
    description="A custom manipulation environment with 3-DOF actions",
    dataset_url="https://example.com/my_data.h5",  # optional
)
```

Verify:

```bash
worldkit env list
worldkit env info worldkit/my-env
```

## Verify the model works

```python
import numpy as np

model = WorldModel.load("custom_model.wk")

# Encode
obs = np.random.rand(96, 96, 3).astype(np.float32)
z = model.encode(obs)
print(f"Latent: {z.shape}")

# Predict
actions = [np.random.rand(3).astype(np.float32)] * 10
result = model.predict(obs, actions)
print(f"Trajectory: {result.latent_trajectory.shape}")

# Plan
goal = np.random.rand(96, 96, 3).astype(np.float32)
plan = model.plan(obs, goal, max_steps=30)
print(f"Plan cost: {plan.expected_cost:.4f}")

# Plausibility
frames = [np.random.rand(96, 96, 3).astype(np.float32) for _ in range(20)]
score = model.plausibility(frames)
print(f"Plausibility: {score:.3f}")
```

## Tips

- **Image size**: WorldKit resizes inputs to the config's `image_size` (default 96). Use higher resolution data for better results — downsampling is handled internally.
- **Episode length**: Longer episodes provide more training signal. Aim for at least 100 steps per episode.
- **Number of episodes**: More data is better. Start with 100-200 episodes, increase if needed.
- **Action normalization**: Normalize actions to `[-1, 1]` for best results. Set `action_space={"low": -1.0, "high": 1.0}` in `plan()`.

## Next steps

- [Data preparation guide](../guides/data_preparation.md) — detailed data formatting
- [Train your first model](train_first_model.md) — training walkthrough
- [Contribute a model](contribute_model.md) — share your model on the Hub
