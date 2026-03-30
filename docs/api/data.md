# Data

WorldKit trains on HDF5 files containing pixel observations and actions. The `worldkit.data` module provides tools for loading, recording, and converting data.

```python
from worldkit.data import HDF5Dataset, Recorder, Converter, MultiEnvironmentDataset
```

## HDF5 data format

WorldKit expects HDF5 files with this structure:

```
data.h5
├── pixels (or observations/obs/images): (N, T, H, W, C)  # uint8 or float32
└── actions (or action):                  (N, T, action_dim)
```

- `N` = number of episodes
- `T` = timesteps per episode
- `H, W, C` = height, width, channels (RGB)
- Pixel key names: `pixels`, `observations`, `obs`, or `images` (auto-detected)
- Action key names: `actions` or `action` (auto-detected)
- Pixel format: `(H, W, C)` or `(C, H, W)` — both are handled automatically
- Pixel range: `[0, 255]` uint8 or `[0, 1]` float32 — normalized internally

## `HDF5Dataset`

PyTorch `Dataset` for loading training data.

```python
class HDF5Dataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        sequence_length: int = 16,
        transform=None,
    )
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| Path` | required | Path to HDF5 file. |
| `sequence_length` | `int` | `16` | Number of timesteps per training sample. |
| `transform` | callable | `None` | Optional transform applied to pixel tensors. |

### Returns from `__getitem__`

A tuple `(pixels, actions)`:
- `pixels`: `torch.Tensor` of shape `(T, C, H, W)`, float32, normalized to `[0, 1]`
- `actions`: `torch.Tensor` of shape `(T, action_dim)`, float32

### Example

```python
from worldkit.data import HDF5Dataset
from torch.utils.data import DataLoader

dataset = HDF5Dataset("pusht_data.h5", sequence_length=16)
print(f"Samples: {len(dataset)}")

loader = DataLoader(dataset, batch_size=64, shuffle=True)
for pixels, actions in loader:
    print(f"Pixels: {pixels.shape}")   # (64, 16, 3, 96, 96)
    print(f"Actions: {actions.shape}") # (64, 16, 2)
    break
```

## `MultiEnvironmentDataset`

Dataset that interleaves samples from multiple HDF5 files, handling different action dimensions by zero-padding.

```python
class MultiEnvironmentDataset(Dataset):
    def __init__(
        self,
        paths: list[str | Path],
        sequence_length: int = 16,
        transform=None,
    )
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `max_action_dim` | `int` | Maximum action dimension across all datasets. |

### Example

```python
from worldkit.data import MultiEnvironmentDataset

dataset = MultiEnvironmentDataset(
    paths=["pusht.h5", "cartpole.h5", "pendulum.h5"],
    sequence_length=16,
)
print(f"Total samples: {len(dataset)}")
print(f"Max action dim: {dataset.max_action_dim}")

# Actions from smaller action spaces are zero-padded
pixels, actions = dataset[0]
print(f"Actions shape: {actions.shape}")  # (16, max_action_dim)
```

## `Recorder`

Record Gymnasium environment interactions to HDF5.

```python
class Recorder:
    def __init__(self, env, output: str | Path)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `env` | Gymnasium env | Must have `render_mode="rgb_array"`. |
| `output` | `str \| Path` | Output HDF5 file path. |

### `record()`

```python
def record(
    self,
    episodes: int = 100,
    policy: str | Callable = "random",
    max_steps_per_episode: int = 500,
) -> Path
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `episodes` | `int` | `100` | Number of episodes to record. |
| `policy` | `str \| Callable` | `"random"` | `"random"` for random actions, or a callable `policy(obs) -> action`. |
| `max_steps_per_episode` | `int` | `500` | Maximum steps per episode. |

**Returns:** Path to the saved HDF5 file.

### Example

```python
import gymnasium as gym
from worldkit.data import Recorder

# Record random episodes
env = gym.make("CartPole-v1", render_mode="rgb_array")
recorder = Recorder(env, output="cartpole_data.h5")
recorder.record(episodes=200, max_steps_per_episode=500)

# Record with a custom policy
def my_policy(obs):
    return env.action_space.sample()  # replace with your policy

recorder.record(episodes=100, policy=my_policy)
```

## `Converter`

Convert video files or other formats to WorldKit HDF5.

```python
class Converter:
    def from_video(
        self,
        input_dir: str | Path,
        output: str | Path,
        fps: int = 10,
        action_labels: str | Path | None = None,
        max_frames: int | None = None,
    ) -> Path
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dir` | `str \| Path` | required | Directory containing video files (MP4, AVI, MOV). |
| `output` | `str \| Path` | required | Output HDF5 file path. |
| `fps` | `int` | `10` | Target FPS. Videos are subsampled to this rate. |
| `action_labels` | `str \| Path \| None` | `None` | Optional CSV mapping frames to actions. |
| `max_frames` | `int \| None` | `None` | Maximum frames per video. |

**Returns:** Path to the HDF5 file.

### Example

```python
from worldkit.data import Converter

converter = Converter()

# Convert a directory of videos
converter.from_video(
    input_dir="./raw_videos/",
    output="training_data.h5",
    fps=10,
)

# With action labels
converter.from_video(
    input_dir="./raw_videos/",
    output="training_data.h5",
    fps=10,
    action_labels="actions.csv",
)
```

### Action label CSV format

```csv
frame_idx,action_0,action_1
0,0.1,0.2
1,0.15,0.18
2,0.12,0.22
...
```

## CLI commands

```bash
# Record from a Gymnasium environment
worldkit record --env CartPole-v1 --episodes 200 --output cartpole.h5

# Convert videos to HDF5
worldkit convert --input ./videos/ --output data.h5 --fps 10
```

## Related

- [Data preparation guide](../guides/data_preparation.md) — detailed guide on preparing training data
- [Train your first model](../tutorials/train_first_model.md) — uses HDF5 data for training
- [Custom environments](../tutorials/custom_environment.md) — recording from custom envs
