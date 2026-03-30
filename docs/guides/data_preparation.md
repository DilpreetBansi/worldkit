# Data preparation

WorldKit trains on HDF5 files containing pixel observations and actions. This guide covers every way to create and prepare training data.

## HDF5 format

WorldKit expects this structure:

```
data.h5
├── pixels: (N, T, H, W, C)     # uint8 [0,255] or float32 [0,1]
└── actions: (N, T, action_dim)  # float32
```

- `N` = number of episodes
- `T` = timesteps per episode
- `H, W` = image height, width (any size — resized internally to `image_size`)
- `C` = channels (3 for RGB)

### Accepted key names

WorldKit auto-detects these key names:

| What | Accepted keys |
|------|--------------|
| Pixel observations | `pixels`, `observations`, `obs`, `images` |
| Actions | `actions`, `action` |

If your keys don't match, you'll get a helpful error listing the actual keys found.

### Pixel format

Both of these work:
- `(N, T, H, W, C)` — channels last (more common)
- `(N, T, C, H, W)` — channels first

Both of these ranges work:
- `uint8` with values in `[0, 255]`
- `float32` with values in `[0, 1]`

WorldKit normalizes automatically.

### Action format

- **Continuous actions**: `float32`, shape `(N, T, action_dim)`
- **Discrete actions**: `int64`, shape `(N, T)` — integers representing action indices

For best results with the CEM planner, normalize continuous actions to `[-1, 1]`.

## Method 1: Record from Gymnasium

The `Recorder` captures pixel observations and actions from any Gymnasium environment.

```python
import gymnasium as gym
from worldkit.data import Recorder

env = gym.make("CartPole-v1", render_mode="rgb_array")
recorder = Recorder(env, output="cartpole_data.h5")

# Random policy
recorder.record(episodes=200, max_steps_per_episode=500)

# Custom policy
def expert_policy(obs):
    return env.action_space.sample()

recorder.record(episodes=200, policy=expert_policy)
```

Requirements:
- The environment must support `render_mode="rgb_array"`
- The environment must follow the Gymnasium API

## Method 2: Convert from video

The `Converter` reads video files and creates HDF5 datasets.

```python
from worldkit.data import Converter

converter = Converter()
converter.from_video(
    input_dir="./videos/",     # directory of MP4/AVI/MOV files
    output="video_data.h5",
    fps=10,                     # target FPS (subsamples if needed)
    max_frames=500,             # max frames per video (optional)
)
```

Supported formats: MP4, AVI, MOV.

### Adding action labels

If you have action labels for your videos, provide them as a CSV:

```python
converter.from_video(
    input_dir="./videos/",
    output="video_data.h5",
    fps=10,
    action_labels="actions.csv",
)
```

CSV format:

```csv
frame_idx,action_0,action_1
0,0.1,0.2
1,0.15,0.18
2,0.12,0.22
```

Without action labels, WorldKit creates zero-valued actions (the model still learns visual dynamics but cannot use action-conditioned prediction or planning).

## Method 3: Create HDF5 manually

For non-Gymnasium environments or custom data pipelines:

```python
import h5py
import numpy as np

episodes = 200
max_steps = 300
action_dim = 3

all_pixels = []
all_actions = []

for ep in range(episodes):
    pixels = []
    actions = []

    obs = my_env.reset()
    for step in range(max_steps):
        frame = my_env.render()  # (H, W, 3) RGB
        action = get_action(obs)

        pixels.append(frame)
        actions.append(action)

        obs, reward, done, info = my_env.step(action)
        if done:
            break

    # Pad short episodes to max_steps
    while len(pixels) < max_steps:
        pixels.append(pixels[-1])
        actions.append(np.zeros(action_dim))

    all_pixels.append(np.stack(pixels))
    all_actions.append(np.stack(actions))

with h5py.File("custom_data.h5", "w") as f:
    f.create_dataset("pixels", data=np.stack(all_pixels).astype(np.uint8))
    f.create_dataset("actions", data=np.stack(all_actions).astype(np.float32))
```

## Method 4: CLI recording and conversion

```bash
# Record from a Gymnasium environment
worldkit record --env CartPole-v1 --episodes 200 --output cartpole.h5 --max-steps 500

# Convert videos
worldkit convert --input ./videos/ --output video_data.h5 --fps 10
```

## Verify your data

Always verify before training:

```python
import h5py

with h5py.File("my_data.h5", "r") as f:
    print("Keys:", list(f.keys()))
    for key in f.keys():
        ds = f[key]
        print(f"  {key}: shape={ds.shape}, dtype={ds.dtype}")
        if ds.dtype == np.uint8:
            print(f"    range: [{ds[:].min()}, {ds[:].max()}]")
```

**Expected output:**

```
Keys: ['pixels', 'actions']
  pixels: shape=(200, 300, 96, 96, 3), dtype=uint8
    range: [0, 255]
  actions: shape=(200, 300, 3), dtype=float32
```

Check for:
- Correct number of episodes and timesteps
- Correct image dimensions
- Non-zero action values (if using action-conditioned features)
- Consistent shapes across episodes

## Multi-environment datasets

Train on data from multiple environments:

```python
from worldkit import WorldModel

model = WorldModel.train(
    data=["pusht.h5", "cartpole.h5", "pendulum.h5"],
    config="base",
    action_dim=4,  # max action dim across all envs
    epochs=100,
)
```

WorldKit automatically:
- Detects the maximum action dimension
- Zero-pads actions from environments with smaller action spaces
- Interleaves batches proportionally

## Data size guidelines

| Metric | Minimum | Recommended |
|--------|---------|-------------|
| Episodes | 50 | 200+ |
| Steps per episode | 50 | 200+ |
| Total frames | 5,000 | 50,000+ |
| Image resolution | 64x64 | 96x96 |

More data generally produces better models. If results are poor, increasing data is often more effective than changing the model config.

## Common issues

### "No pixel data found in HDF5 file"

Your HDF5 keys don't match any of the expected names. Rename to one of: `pixels`, `observations`, `obs`, `images`.

### Variable-length episodes

All episodes in the HDF5 file must have the same number of timesteps. Pad shorter episodes by repeating the final frame with zero actions.

### Large files

HDF5 supports chunked storage and compression:

```python
with h5py.File("data.h5", "w") as f:
    f.create_dataset(
        "pixels",
        data=pixel_data,
        chunks=(1, 100, 96, 96, 3),
        compression="gzip",
        compression_opts=4,
    )
```

## Related

- [Data API reference](../api/data.md) — HDF5Dataset, Recorder, Converter
- [Custom environments](../tutorials/custom_environment.md) — recording from custom envs
- [Train your first model](../tutorials/train_first_model.md) — training walkthrough
