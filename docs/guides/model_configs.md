# Choosing the right model configuration

WorldKit ships four configs: `nano`, `base`, `large`, and `xl`. They share the same API — the only difference is capacity and speed.

## Config comparison

| Config | Params | Latent dim | Encoder | Predictor | Train time* |
|--------|--------|------------|---------|-----------|-------------|
| `nano` | ~3.5M | 128 | ViT-Tiny (192) | 2 layers, 4 heads | ~30s |
| `base` | ~13M | 192 | ViT-Small (384) | 3 layers, 4 heads | ~60s |
| `large` | ~54M | 384 | ViT-Base (768) | 4 layers, 8 heads | ~8 min |
| `xl` | ~102M | 512 | ViT-Large (1024) | 6 layers, 8 heads | ~20 min |

*100 epochs on Apple M4 Pro with MPS. GPU times vary.

## When to use each config

### `nano` — prototyping and edge deployment

Use nano when:
- Prototyping a new environment or data pipeline
- Running unit tests (fast iteration)
- Deploying to edge devices (Jetson, mobile, browser)
- Latency-critical applications (real-time control loops)
- You need the fastest possible training turnaround

```python
model = WorldModel.train(data="data.h5", config="nano", epochs=50)
```

### `base` — the default choice

Use base when:
- Training your first model on a new environment
- Most single-environment tasks (Push-T, CartPole, etc.)
- Moderate compute budget (laptop GPU or CPU)
- Good balance of quality and speed

```python
model = WorldModel.train(data="data.h5", config="base", epochs=100)
```

### `large` — complex environments

Use large when:
- Complex visual environments (3D, rich textures)
- Environments with many distinct objects
- You need higher latent capacity for fine-grained state encoding
- Training time is not a bottleneck

```python
model = WorldModel.train(data="data.h5", config="large", epochs=100)
```

### `xl` — maximum capacity

Use xl when:
- Multi-object scenes with complex interactions
- High-dimensional action spaces (6+ DOF)
- Research experiments requiring maximum model capacity
- You have access to a GPU with sufficient VRAM (16GB+)

```python
model = WorldModel.train(data="data.h5", config="xl", epochs=100)
```

## Auto-configuration

Let WorldKit recommend a config based on your data and constraints:

```python
config, explanation = WorldModel.auto_config(
    data="my_data.h5",
    max_training_time="30m",
    target_device="jetson",
)
print(explanation)
# "Recommended 'nano' config: fits 30m budget, suitable for Jetson deployment"

model = WorldModel.train(data="my_data.h5", config=config, epochs=100)
```

## Custom configurations

Override any field on a preset:

```python
from worldkit.core.config import get_config

# Base config with larger latent space
config = get_config("base", latent_dim=256, action_dim=6)

# Nano config with more predictor depth
config = get_config("nano", pred_depth=4, pred_heads=8)
```

Register a reusable custom config:

```python
from worldkit.core.config import register_config

register_config(
    "my_custom",
    latent_dim=256,
    pred_depth=4,
    pred_heads=8,
    action_dim=6,
)

# Now usable by name
model = WorldModel.train(data="data.h5", config="my_custom")
```

## Memory and VRAM requirements

| Config | CPU RAM | GPU VRAM (training) | GPU VRAM (inference) |
|--------|---------|--------------------|--------------------|
| `nano` | ~1 GB | ~2 GB | ~0.5 GB |
| `base` | ~2 GB | ~4 GB | ~1 GB |
| `large` | ~4 GB | ~8 GB | ~2 GB |
| `xl` | ~8 GB | ~16 GB | ~4 GB |

If you run out of memory, try:
- A smaller config
- Reducing `batch_size` (e.g., 32 instead of 64)
- Using CPU instead of GPU (slower but no VRAM limit)

## Latent dim and capacity

The `latent_dim` is the dimensionality of the compressed state representation:

- **128** (nano): Captures coarse state features. Sufficient for simple environments.
- **192** (base): Good for most single-environment tasks.
- **384** (large): Fine-grained state encoding. Better for complex visual environments.
- **512** (xl): Maximum capacity. Useful for multi-object, multi-agent scenarios.

Higher latent dim = more information retained from the observation, but also more parameters and slower inference.

## Distillation: getting the best of both

Train a large model for quality, then distill to nano for deployment:

```python
# Train a high-quality large model
teacher = WorldModel.train(data="data.h5", config="large", epochs=200)

# Distill to nano for edge deployment
student = WorldModel.distill(
    teacher=teacher,
    student_config="nano",
    data="data.h5",
    epochs=50,
)
student.save("distilled_nano.wk")
```

The distilled nano model retains much of the large model's prediction quality at a fraction of the compute cost.

## Related

- [Config API reference](../api/config.md) — ModelConfig fields and presets
- [Train your first model](../tutorials/train_first_model.md) — training tutorial
- [Export and deploy](../tutorials/export_deploy.md) — deployment options
