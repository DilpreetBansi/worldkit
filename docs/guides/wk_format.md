# .wk file format

The `.wk` format is WorldKit's model file format. It is a ZIP archive containing everything needed to load and run a world model — weights, config, and metadata.

For the full specification, see [.wk Format Specification](../wk_format_spec.md).

## Format overview

```
model.wk (ZIP archive)
├── config.json            [required]  Model architecture configuration
├── weights.safetensors    [required]  Model weights (safetensors format)
├── metadata.json          [required]  Training and provenance metadata
├── action_space.json      [optional]  Action space definition
└── model_card.yaml        [optional]  Human-readable model card
```

## Design principles

- **Safe**: Uses [safetensors](https://github.com/huggingface/safetensors) instead of pickle. No arbitrary code execution on load.
- **Portable**: JSON + safetensors can be read from Python, Rust, JavaScript, C, and Go.
- **Inspectable**: Metadata and config are readable without loading weights.

## Format versions

| Version | Format | Status |
|---------|--------|--------|
| v1 | `torch.save()` pickle | Deprecated. Still loadable with a warning. |
| v2 | ZIP archive (current) | Stable. |

## Save a model

```python
model.save(
    "my_model.wk",
    metadata={"dataset": "pusht", "epochs": 100, "final_train_loss": 0.023},
    action_space={"dim": 2, "type": "continuous", "low": -1.0, "high": 1.0},
    model_card={"name": "pusht-base", "tags": ["robotics", "manipulation"]},
)
```

## Load a model

```python
model = WorldModel.load("my_model.wk", device="cpu")
```

Both v1 (legacy) and v2 (current) files are supported. Legacy files trigger a deprecation warning.

## Inspect without loading weights

```python
from worldkit.core.format import WKFormat

info = WKFormat.inspect("my_model.wk")
print(f"Config: {info['config'].name}")
print(f"Latent dim: {info['config'].latent_dim}")
print(f"Weights size: {info['weights_size_bytes'] / 1e6:.1f} MB")
print(f"Version: {info['metadata']['worldkit_version']}")
```

Or via CLI:

```bash
worldkit inspect my_model.wk
```

## Validate a file

```python
WKFormat.validate("my_model.wk")  # raises ValueError if invalid
```

```bash
worldkit validate my_model.wk
```

## Low-level API

```python
from worldkit.core.format import WKFormat

# Save manually
WKFormat.save(
    path="model.wk",
    model_state_dict=jepa.state_dict(),
    config=config,
    metadata={"worldkit_version": "0.1.0", "format_version": 2},
)

# Load all components
data = WKFormat.load("model.wk")
config = data["config"]           # ModelConfig
weights = data["model_state_dict"]  # state dict
metadata = data["metadata"]       # dict

# Check format version
is_v2 = WKFormat.is_new_format("model.wk")
```

## Reading .wk files in other languages

Since `.wk` files are standard ZIP archives, they can be read from any language:

1. Unzip the archive
2. Parse `config.json` and `metadata.json` as JSON
3. Load weights with the safetensors library (available for Python, Rust, JS, C, Go)

See the [full specification](../wk_format_spec.md) for JavaScript and Rust examples.

## Related

- [.wk Format Specification](../wk_format_spec.md) — complete spec with all fields
- [WorldModel.save()](../api/worldmodel.md) — save method reference
- [WorldModel.load()](../api/worldmodel.md) — load method reference
- [Export and deploy](../tutorials/export_deploy.md) — other export formats
