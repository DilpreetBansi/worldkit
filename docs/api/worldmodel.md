# WorldModel

`worldkit.WorldModel` is the main class for training, inference, and deployment. It wraps the internal JEPA architecture and exposes a numpy-in / numpy-out API.

```python
from worldkit import WorldModel
```

## Construction

### `WorldModel.train()`

Train a world model from HDF5 data.

```python
@classmethod
def train(
    cls,
    data: str | Path | list[str | Path],
    config: str | ModelConfig = "base",
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-4,
    lambda_reg: float = 1.0,
    action_dim: int = 2,
    device: str = "auto",
    log_to: str | None = None,
    checkpoint_dir: str = "./checkpoints",
    seed: int = 42,
    **kwargs,
) -> WorldModel
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `str \| Path \| list` | required | Path to HDF5 file(s). When a list is passed, action dimensions are zero-padded and batches interleaved proportionally. |
| `config` | `str \| ModelConfig` | `"base"` | Config name (`"nano"`, `"base"`, `"large"`, `"xl"`) or a `ModelConfig` instance. |
| `epochs` | `int` | `100` | Number of training epochs. |
| `batch_size` | `int` | `64` | Training batch size. |
| `lr` | `float` | `1e-4` | Learning rate for AdamW optimizer. |
| `lambda_reg` | `float` | `1.0` | SIGReg regularization weight. Higher values enforce stronger Gaussian structure on the latent space. |
| `action_dim` | `int` | `2` | Action dimension (only used when `config` is a string). |
| `device` | `str` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"`, or `"mps"`. |
| `log_to` | `str \| None` | `None` | Logging destination (e.g., WandB project name). |
| `checkpoint_dir` | `str` | `"./checkpoints"` | Directory for saving training checkpoints. |
| `seed` | `int` | `42` | Random seed for reproducibility. |

**Returns:** Trained `WorldModel` instance.

**Example:**

```python
model = WorldModel.train(
    data="pusht_data.h5",
    config="base",
    epochs=100,
    lambda_reg=1.0,
    device="auto",
)
model.save("pusht_model.wk")
```

**Multi-environment training:**

```python
model = WorldModel.train(
    data=["pusht.h5", "cartpole.h5"],
    config="base",
    action_dim=4,  # max action dim across datasets
    epochs=100,
)
```

### `WorldModel.load()`

Load a model from a `.wk` file.

```python
@classmethod
def load(cls, path: str | Path, device: str = "auto") -> WorldModel
```

Supports both the current ZIP-based format (v2) and legacy `torch.save` format (v1). Legacy files trigger a deprecation warning.

**Example:**

```python
model = WorldModel.load("my_model.wk", device="cpu")
```

### `WorldModel.from_hub()`

Download and load a pre-trained model from Hugging Face Hub.

```python
@classmethod
def from_hub(cls, model_id: str, device: str = "auto") -> WorldModel
```

**Example:**

```python
model = WorldModel.from_hub("DilpreetBansi/pusht")
```

### `WorldModel.auto_config()`

Recommend a model configuration based on data and constraints.

```python
@classmethod
def auto_config(
    cls,
    data: str | Path,
    max_training_time: str = "2h",
    target_device: str | None = None,
    trial_epochs: int = 5,
    device: str = "cpu",
) -> tuple[ModelConfig, str]
```

Samples data, runs quick training trials with each config, and picks the best one that fits within the time and device constraints.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `str \| Path` | required | Path to HDF5 training data. |
| `max_training_time` | `str` | `"2h"` | Time budget, e.g., `"30m"`, `"2h"`. |
| `target_device` | `str \| None` | `None` | Deployment target, e.g., `"jetson"`, `"browser"`. |
| `trial_epochs` | `int` | `5` | Number of epochs per trial run. |
| `device` | `str` | `"cpu"` | Device for running trials. |

**Returns:** Tuple of `(ModelConfig, explanation_string)`.

**Example:**

```python
config, explanation = WorldModel.auto_config(
    data="my_data.h5",
    max_training_time="30m",
    target_device="jetson",
)
print(explanation)
# "Recommended 'nano' config: fits 30m budget, 3.5M params suitable for Jetson"
```

## Properties

### `config`

```python
@property
def config(self) -> ModelConfig
```

The `ModelConfig` for this model.

### `latent_dim`

```python
@property
def latent_dim(self) -> int
```

Dimensionality of the latent space (e.g., 128 for nano, 192 for base).

### `device`

```python
@property
def device(self) -> str
```

Device string: `"cpu"`, `"cuda"`, or `"mps"`.

### `num_params`

```python
@property
def num_params(self) -> int
```

Number of trainable parameters.

**Example:**

```python
print(f"Config: {model.config.name}")
print(f"Latent dim: {model.latent_dim}")
print(f"Device: {model.device}")
print(f"Parameters: {model.num_params:,}")
# Config: base
# Latent dim: 192
# Device: cpu
# Parameters: 13,000,000
```

## Inference

### `encode()`

Encode a raw observation into a latent vector.

```python
def encode(self, observation: np.ndarray | torch.Tensor) -> torch.Tensor
```

Handles pixel format detection automatically:
- Shape: `(H, W, C)` or `(C, H, W)`
- Range: `[0, 255]` uint8, `[0, 1]` float32, or `[-1, 1]` float32

**Returns:** Latent tensor of shape `(latent_dim,)`.

**Example:**

```python
import numpy as np

obs = np.random.rand(96, 96, 3).astype(np.float32)
z = model.encode(obs)
print(z.shape)
# torch.Size([192])
```

### `predict()`

Predict future latent states from a current observation and action sequence.

```python
@torch.no_grad()
def predict(
    self,
    observation: np.ndarray | torch.Tensor,
    actions: list,
    steps: int | None = None,
    return_latents: bool = False,
) -> PredictionResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `observation` | `np.ndarray` | required | Current observation image. |
| `actions` | `list` | required | List of action arrays. |
| `steps` | `int \| None` | `None` | If set with a single action, repeats that action `steps` times. |

**Returns:** [`PredictionResult`](prediction.md) with `latent_trajectory`, `confidence`, `steps`.

**Example:**

```python
actions = [np.array([0.1, 0.2])] * 10
result = model.predict(obs, actions)
print(result.latent_trajectory.shape)  # (10, 192)
print(result.confidence)               # 0.8
```

### `plan()`

Find an optimal action sequence to reach a goal state using CEM.

```python
@torch.no_grad()
def plan(
    self,
    current_state: np.ndarray,
    goal_state: np.ndarray,
    max_steps: int = 50,
    n_candidates: int = 200,
    n_elite: int = 20,
    n_iterations: int = 5,
    action_space: dict | None = None,
) -> PlanResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `current_state` | `np.ndarray` | required | Current observation `(H, W, C)`. |
| `goal_state` | `np.ndarray` | required | Goal observation `(H, W, C)`. |
| `max_steps` | `int` | `50` | Planning horizon (max actions). |
| `n_candidates` | `int` | `200` | CEM candidate action sequences per iteration. |
| `n_elite` | `int` | `20` | Number of elite sequences kept per iteration. |
| `n_iterations` | `int` | `5` | CEM refinement iterations. |
| `action_space` | `dict \| None` | `None` | Action bounds: `{"low": -1.0, "high": 1.0}`. |

**Returns:** [`PlanResult`](planning.md) with `actions`, `expected_cost`, `success_probability`, `planning_time_ms`.

**Example:**

```python
plan = model.plan(
    current_frame, goal_frame,
    max_steps=50,
    n_candidates=500,
    n_elite=50,
)
for action in plan.actions:
    obs, reward, done, _, info = env.step(action)
```

See [Planning](planning.md) for more details on CEM and hierarchical planning.

### `hierarchical_plan()`

Plan long-horizon action sequences by decomposing into subgoals.

```python
@torch.no_grad()
def hierarchical_plan(
    self,
    current_state: np.ndarray,
    goal_state: np.ndarray,
    max_subgoals: int = 5,
    steps_per_subgoal: int = 50,
    n_candidates: int = 200,
    n_elite: int = 20,
    n_iterations: int = 5,
    action_space: dict | None = None,
) -> HierarchicalPlanResult
```

Interpolates subgoals in latent space between current and goal states, then uses CEM to plan between consecutive subgoals.

**Returns:** `HierarchicalPlanResult` with `actions`, `subgoals`, `segment_plans`, `total_planning_time_ms`.

**Example:**

```python
result = model.hierarchical_plan(
    current_frame, goal_frame,
    max_subgoals=5,
    steps_per_subgoal=30,
)
print(f"Total actions: {len(result.actions)}")
print(f"Subgoals: {len(result.subgoals)}")
print(f"Time: {result.total_planning_time_ms:.0f}ms")
```

### `plausibility()`

Score how physically plausible a sequence of observations is.

```python
@torch.no_grad()
def plausibility(
    self,
    frames: list[np.ndarray],
    actions: list | None = None,
) -> float
```

Computes consecutive-frame prediction errors and applies exponential decay.

**Returns:** `float` in `[0.0, 1.0]`. 1.0 = fully expected, 0.0 = physically impossible.

**Example:**

```python
score = model.plausibility(video_frames)
if score < 0.3:
    print("Anomaly detected!")
```

## Evaluation

### `probe()`

Train linear probes to measure what the latent space encodes.

```python
def probe(
    self,
    data: str | Path,
    properties: list[str],
    labels: str | Path,
    alpha: float = 1.0,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> ProbeResult
```

Encodes observations through the frozen encoder, fits Ridge regression from latent vectors to target properties.

**Returns:** [`ProbeResult`](probing.md) with `property_scores` (R²), `mse_scores`, `probes`, `summary`.

**Example:**

```python
result = model.probe(
    data="pusht.h5",
    properties=["x_position", "y_position", "angle"],
    labels="pusht_labels.csv",
)
print(result.summary)
# x_position: R²=0.92, MSE=0.0034
# y_position: R²=0.89, MSE=0.0051
# angle: R²=0.78, MSE=0.0123
```

### `visualize_latent_space()`

Plot a dimensionality-reduced view of the latent space.

```python
def visualize_latent_space(
    self,
    data: str | Path,
    method: str = "pca",
    color_by: str | None = None,
    save_to: str | Path | None = None,
) -> matplotlib.figure.Figure
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `str \| Path` | required | Path to HDF5 file with observations. |
| `method` | `str` | `"pca"` | Reduction method: `"pca"`, `"tsne"`, or `"umap"`. |
| `color_by` | `str \| None` | `None` | Color points by `"episode"` or `"timestep"`. |
| `save_to` | `str \| Path \| None` | `None` | Save figure to path, or display if `None`. |

**Example:**

```python
fig = model.visualize_latent_space(
    data="pusht.h5",
    method="tsne",
    color_by="episode",
    save_to="latent_space.png",
)
```

### `rollout_gif()`

Generate an animated GIF of a latent-space rollout trajectory.

```python
def rollout_gif(
    self,
    observation: np.ndarray,
    actions: list,
    save_to: str | Path = "rollout.gif",
    fps: int = 10,
) -> Path
```

**Example:**

```python
path = model.rollout_gif(obs, actions, save_to="my_rollout.gif", fps=15)
print(f"Saved to {path}")
```

### `WorldModel.compare()`

Compare multiple world models on the same data.

```python
@classmethod
def compare(
    cls,
    models: dict[str, WorldModel],
    data_path: str | Path,
    episodes: int = 50,
    save_to: str | Path = "comparison.html",
) -> ComparisonResult
```

**Example:**

```python
nano = WorldModel.load("nano_model.wk")
base = WorldModel.load("base_model.wk")

result = WorldModel.compare(
    models={"nano": nano, "base": base},
    data_path="eval_data.h5",
    save_to="comparison.html",
)
print(f"Best model: {result.best_model}")
```

## Persistence

### `save()`

Save the model to a `.wk` ZIP archive.

```python
def save(
    self,
    path: str | Path,
    metadata: dict | None = None,
    action_space: dict | None = None,
    model_card: dict | None = None,
) -> None
```

**Example:**

```python
model.save(
    "my_model.wk",
    metadata={"dataset": "pusht", "epochs": 100, "final_train_loss": 0.023},
    action_space={"dim": 2, "type": "continuous", "low": -1.0, "high": 1.0},
)
```

### `export()`

Export the model for deployment.

```python
def export(
    self,
    format: str = "onnx",
    output: str | Path = "./export/",
    optimize: bool = True,
    fp16: bool = True,
    int8: bool = False,
) -> Path
```

**Supported formats:**

| Format | File | Notes |
|--------|------|-------|
| `"onnx"` | `.onnx` | Portable, runs with ONNX Runtime |
| `"torchscript"` | `.pt` | PyTorch-native, no Python dependency |
| `"tensorrt"` | `.engine` | Optimized for NVIDIA GPUs. Requires `tensorrt`. |
| `"coreml"` | `.mlpackage` | Apple devices. Requires `coremltools`. |

**Example:**

```python
onnx_path = model.export(format="onnx", output="./deploy/")
print(f"Exported to {onnx_path}")
```

## Distillation

### `WorldModel.distill()`

Distill a large model into a smaller one.

```python
@classmethod
def distill(
    cls,
    teacher: WorldModel,
    student_config: str | ModelConfig = "nano",
    data: str | Path = "",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: str = "auto",
    seed: int = 42,
) -> WorldModel
```

The student learns to match the teacher's latent predictions via MSE loss.

**Example:**

```python
teacher = WorldModel.load("large_model.wk")
student = WorldModel.distill(
    teacher=teacher,
    student_config="nano",
    data="training_data.h5",
    epochs=50,
)
student.save("distilled_nano.wk")
```

## Online Learning

### `enable_online_learning()`

Enable incremental online learning.

```python
def enable_online_learning(
    self,
    lr: float = 1e-5,
    buffer_size: int = 1000,
    batch_size: int = 16,
    update_every: int = 4,
    ema_decay: float = 0.0,
) -> None
```

### `update()`

Add a transition and optionally perform an online gradient step.

```python
def update(
    self,
    observation: np.ndarray,
    action: np.ndarray,
    next_observation: np.ndarray,
) -> float | None
```

**Returns:** Loss value if a gradient step was performed, else `None`.

**Example:**

```python
model.enable_online_learning(lr=1e-5, buffer_size=5000)

for step in range(10000):
    action = model.plan(obs, goal).actions[0]
    next_obs, reward, done, _, info = env.step(action)
    loss = model.update(obs, action, next_obs)
    if loss is not None:
        print(f"Step {step}: loss={loss:.4f}")
    obs = next_obs
```
