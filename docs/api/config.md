# Config

Model configuration in WorldKit uses Python dataclasses — no YAML files.

```python
from worldkit import ModelConfig, get_config
```

## `get_config()`

Get a model configuration by name with optional overrides.

```python
def get_config(name: str = "base", **overrides) -> ModelConfig
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"base"` | Config name: `"nano"`, `"base"`, `"large"`, or `"xl"`. |
| `**overrides` | keyword args | | Override any `ModelConfig` field. |

**Example:**

```python
# Default base config
config = get_config("base")

# Base config with custom action dimension
config = get_config("base", action_dim=6)

# Nano config with larger latent space
config = get_config("nano", latent_dim=256)
```

**Raises:** `ValueError` if the name is not found.

## `register_config()`

Register a custom named configuration.

```python
def register_config(name: str, **overrides) -> ModelConfig
```

**Example:**

```python
from worldkit.core.config import register_config

my_config = register_config(
    "my_custom",
    latent_dim=256,
    pred_depth=4,
    pred_heads=8,
    action_dim=6,
)

# Now usable by name anywhere
model = WorldModel.train(data="data.h5", config="my_custom")
```

## `ModelConfig`

The full configuration dataclass for a WorldKit model.

```python
@dataclass
class ModelConfig:
    # Identity
    name: str = "base"
    backend: str = "lewm"

    # Encoder (Vision Transformer)
    encoder_name: str = "google/vit-base-patch16-224"
    encoder_embed_dim: int = 768
    image_size: int = 96
    patch_size: int = 16

    # Latent space
    latent_dim: int = 192
    proj_hidden_dim: int = 384
    proj_output_dim: int = 192

    # Predictor (Autoregressive Transformer)
    pred_depth: int = 3
    pred_heads: int = 4
    pred_dim_head: int = 64
    pred_mlp_dim: int = 384
    pred_dropout: float = 0.0
    pred_emb_dropout: float = 0.0
    context_length: int = 3

    # Action encoder
    action_dim: int = 2
    action_embed_dim: int = 192

    # SIGReg loss
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024

    # Training
    lambda_reg: float = 1.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    batch_size: int = 64
    num_workers: int = 4
    sequence_length: int = 16
```

### Field reference

#### Encoder fields

| Field | Default | Description |
|-------|---------|-------------|
| `encoder_name` | varies by preset | Hugging Face model ID for ViT backbone |
| `encoder_embed_dim` | varies | ViT embedding dimension |
| `image_size` | `96` | Input image size (square) |
| `patch_size` | `16` | ViT patch size |

#### Latent space fields

| Field | Default | Description |
|-------|---------|-------------|
| `latent_dim` | varies | Dimensionality of the latent space |
| `proj_hidden_dim` | varies | Hidden layer size in the projector MLP |
| `proj_output_dim` | varies | Projector output dimension (matches `latent_dim`) |

#### Predictor fields

| Field | Default | Description |
|-------|---------|-------------|
| `pred_depth` | varies | Number of transformer layers |
| `pred_heads` | varies | Number of attention heads |
| `pred_dim_head` | varies | Dimension per attention head |
| `pred_mlp_dim` | varies | Feed-forward hidden dimension |
| `pred_dropout` | `0.0` | Dropout rate in transformer |
| `pred_emb_dropout` | `0.0` | Embedding dropout rate |
| `context_length` | `3` | Number of past frames the predictor sees |

#### Action fields

| Field | Default | Description |
|-------|---------|-------------|
| `action_dim` | `2` | Action space dimensionality |
| `action_embed_dim` | varies | Action embedding dimension (matches `latent_dim`) |

#### Training fields

| Field | Default | Description |
|-------|---------|-------------|
| `lambda_reg` | `1.0` | SIGReg regularization weight (the ONE hyperparameter) |
| `learning_rate` | `1e-4` | AdamW learning rate |
| `weight_decay` | `0.05` | AdamW weight decay |
| `warmup_epochs` | `10` | Linear warmup epochs |
| `batch_size` | `64` | Default batch size |
| `num_workers` | `4` | DataLoader workers |
| `sequence_length` | `16` | Training sequence length (= `context_length` + prediction_length) |

## Config presets

### `nano`

The smallest config. Fast to train, suitable for prototyping and edge deployment.

| Field | Value |
|-------|-------|
| Parameters | ~3.5M |
| Encoder | ViT-Tiny (`google/vit-tiny-patch16-224`, dim=192) |
| Latent dim | 128 |
| Predictor | 2 layers, 4 heads, dim_head=32 |
| Action embed | 128 |

### `base`

The default config. Good balance of quality and speed.

| Field | Value |
|-------|-------|
| Parameters | ~13M |
| Encoder | ViT-Small (`google/vit-small-patch16-224`, dim=384) |
| Latent dim | 192 |
| Predictor | 3 layers, 4 heads, dim_head=64 |
| Action embed | 192 |

### `large`

For complex environments with rich visual detail.

| Field | Value |
|-------|-------|
| Parameters | ~54M |
| Encoder | ViT-Base (`google/vit-base-patch16-224`, dim=768) |
| Latent dim | 384 |
| Predictor | 4 layers, 8 heads, dim_head=64 |
| Action embed | 384 |

### `xl`

Maximum capacity. For multi-object scenes and high-dimensional action spaces.

| Field | Value |
|-------|-------|
| Parameters | ~102M |
| Encoder | ViT-Large (`google/vit-large-patch16-224`, dim=1024) |
| Latent dim | 512 |
| Predictor | 6 layers, 8 heads, dim_head=64 |
| Action embed | 512 |

## Choosing a config

See [Model Configurations Guide](../guides/model_configs.md) for detailed guidance on selecting the right config for your use case.
