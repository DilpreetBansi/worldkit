# `.wk` File Format Specification

**Version:** 2
**Status:** Stable
**Authors:** WorldKit Contributors

## Overview

A `.wk` file is a ZIP archive containing everything needed to load and run a WorldKit world model. The format is designed to be safe (no pickle), portable (JSON + safetensors), and inspectable (metadata readable without loading weights).

## Format History

| Version | Description |
|---------|-------------|
| 1 | Raw `torch.save()` pickle. Deprecated — still loadable for backward compatibility. |
| 2 | ZIP archive with JSON config, safetensors weights, and structured metadata. Current version. |

## Archive Structure

```
model.wk (ZIP, DEFLATE compression)
├── config.json            [REQUIRED]  Model architecture configuration
├── weights.safetensors    [REQUIRED]  Model weights in safetensors format
├── metadata.json          [REQUIRED]  Training and provenance metadata
├── action_space.json      [OPTIONAL]  Action space definition
└── model_card.yaml        [OPTIONAL]  Human-readable model card
```

## Entry Specifications

### config.json (required)

JSON serialization of a `ModelConfig` dataclass. All fields must be present.

```json
{
  "name": "base",
  "encoder_name": "google/vit-small-patch16-224",
  "encoder_embed_dim": 384,
  "image_size": 96,
  "patch_size": 16,
  "latent_dim": 192,
  "proj_hidden_dim": 384,
  "proj_output_dim": 192,
  "pred_depth": 3,
  "pred_heads": 4,
  "pred_dim_head": 64,
  "pred_mlp_dim": 384,
  "pred_dropout": 0.0,
  "pred_emb_dropout": 0.0,
  "context_length": 3,
  "action_dim": 2,
  "action_embed_dim": 192,
  "sigreg_knots": 17,
  "sigreg_num_proj": 1024,
  "lambda_reg": 1.0,
  "learning_rate": 0.0001,
  "weight_decay": 0.05,
  "warmup_epochs": 10,
  "batch_size": 64,
  "num_workers": 4,
  "sequence_length": 16
}
```

### weights.safetensors (required)

Model weights stored in the [safetensors](https://github.com/huggingface/safetensors) format. This is a safe, zero-copy tensor serialization format — no arbitrary code execution risk (unlike pickle).

The state dict corresponds to the `JEPA` module's `state_dict()`.

### metadata.json (required)

Provenance and training information.

```json
{
  "worldkit_version": "0.1.0",
  "format_version": 2,
  "created_at": "2026-03-30T12:00:00+00:00",
  "dataset": "pusht",
  "epochs": 100,
  "final_train_loss": 0.0234,
  "final_val_loss": 0.0312,
  "training_time_seconds": 3600
}
```

**Required fields:** `worldkit_version`, `format_version`, `created_at`.
All other fields are optional and depend on context (e.g., models loaded from hub may not have training info).

### action_space.json (optional)

Defines the action space the model was trained on.

```json
{
  "dim": 2,
  "type": "continuous",
  "low": -1.0,
  "high": 1.0
}
```

For discrete action spaces:

```json
{
  "dim": 4,
  "type": "discrete",
  "num_actions": 4
}
```

### model_card.yaml (optional)

Human-readable model card in YAML format.

```yaml
name: base
description: WorldKit base model trained on PushT
architecture: JEPA + SIGReg
parameters: 15000000
latent_dim: 192
worldkit_version: 0.1.0
benchmarks:
  pusht_success_rate: 0.85
  planning_fps: 120
tags:
  - robotics
  - manipulation
```

## Reading .wk Files in Other Languages

The format is designed for cross-language compatibility:

1. **Unzip** the archive using any ZIP library.
2. **Parse** `config.json` and `metadata.json` as standard JSON.
3. **Parse** `model_card.yaml` as standard YAML.
4. **Load weights** using the safetensors library, which has bindings for Python, Rust, JavaScript, C, Go, and more. See: https://github.com/huggingface/safetensors

### Example: Reading metadata in JavaScript

```javascript
const JSZip = require('jszip');
const fs = require('fs');

const data = fs.readFileSync('model.wk');
const zip = await JSZip.loadAsync(data);
const metadata = JSON.parse(await zip.file('metadata.json').async('string'));
console.log(metadata.worldkit_version);
```

### Example: Reading metadata in Rust

```rust
use std::io::Read;
use zip::ZipArchive;

let file = std::fs::File::open("model.wk")?;
let mut archive = ZipArchive::new(file)?;
let mut metadata_file = archive.by_name("metadata.json")?;
let mut contents = String::new();
metadata_file.read_to_string(&mut contents)?;
let metadata: serde_json::Value = serde_json::from_str(&contents)?;
```

## Detecting Format Version

To distinguish v1 (legacy) from v2 (current):

```python
import zipfile

def detect_version(path):
    if zipfile.is_zipfile(path):
        return 2  # New ZIP-based format
    else:
        return 1  # Legacy torch.save format
```

## Versioning Policy

- The `format_version` field in `metadata.json` tracks the archive structure version.
- Minor additions (new optional entries) do NOT bump the format version.
- Structural changes to required entries or incompatible changes bump the format version.
- WorldKit will always support loading the previous format version with a deprecation warning.

## CLI Tools

```bash
# Validate a .wk file
worldkit validate model.wk

# Inspect metadata without loading weights
worldkit inspect model.wk
```

## Python API

```python
from worldkit.core.format import WKFormat

# Save
WKFormat.save("model.wk", state_dict, config, metadata, action_space, model_card)

# Load everything
data = WKFormat.load("model.wk")
config = data["config"]
weights = data["model_state_dict"]

# Validate
WKFormat.validate("model.wk")  # raises ValueError if invalid

# Inspect (no weight loading)
info = WKFormat.inspect("model.wk")
print(info["metadata"]["worldkit_version"])
```
