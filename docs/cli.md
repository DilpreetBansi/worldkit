# CLI Reference

WorldKit provides a command-line interface for training, serving, exporting, and managing models.

```bash
worldkit --help
```

## `worldkit train`

Train a world model from HDF5 data.

```bash
worldkit train --data <path> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | required | Path to HDF5 training data |
| `--config` | `base` | Config name: `nano`, `base`, `large`, `xl` |
| `--epochs` | `100` | Number of training epochs |
| `--batch-size` | `64` | Training batch size |
| `--lr` | `1e-4` | Learning rate |
| `--lambda-reg` | `1.0` | SIGReg regularization weight |
| `--action-dim` | `2` | Action dimension |
| `--device` | `auto` | Device: `auto`, `cpu`, `cuda`, `mps` |
| `--output` | `./model.wk` | Output model path |
| `--seed` | `42` | Random seed |

**Example:**

```bash
worldkit train --data pusht.h5 --config base --epochs 200 --output pusht_model.wk
```

## `worldkit serve`

Start a FastAPI inference server.

```bash
worldkit serve --model <path> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | required | Path to `.wk` model file |
| `--port` | `8000` | Server port |
| `--host` | `0.0.0.0` | Server host |

**Example:**

```bash
worldkit serve --model pusht_model.wk --port 8000
```

See [REST API Reference](rest_api.md) for endpoint documentation.

## `worldkit export`

Export a model for deployment.

```bash
worldkit export --model <path> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | required | Path to `.wk` model file |
| `--format` | `onnx` | Export format: `onnx`, `torchscript`, `tensorrt`, `coreml`, `ros2` |
| `--output` | `./export/` | Output directory |
| `--fp16/--no-fp16` | `--fp16` | FP16 precision (TensorRT only) |
| `--int8` | off | INT8 quantization (TensorRT only) |
| `--node-name` | `worldkit_node` | ROS2 node name (ROS2 only) |

**Examples:**

```bash
# ONNX
worldkit export --model model.wk --format onnx

# TensorRT with FP16
worldkit export --model model.wk --format tensorrt --fp16

# ROS2 package
worldkit export --model model.wk --format ros2 --node-name my_node
```

## `worldkit info`

Display model information.

```bash
worldkit info --model <path>
```

**Output:**

```
Model: base
Parameters: 13,000,000
Latent dim: 192
Device: cpu
```

## `worldkit inspect`

Inspect a `.wk` file without loading weights.

```bash
worldkit inspect <path>
```

**Output:**

```
Config: base
Latent dim: 192
Image size: 96
Action dim: 2
Weights: 52.3 MB
Format version: 2
Created: 2026-03-30T12:00:00+00:00
```

## `worldkit validate`

Validate a `.wk` file structure.

```bash
worldkit validate <path>
```

Checks that all required entries (`config.json`, `weights.safetensors`, `metadata.json`) are present and valid.

## `worldkit convert`

Convert video files to HDF5 format.

```bash
worldkit convert --input <dir> --output <path> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | required | Directory containing video files |
| `--output` | required | Output HDF5 path |
| `--fps` | `10` | Target frames per second |

**Example:**

```bash
worldkit convert --input ./videos/ --output data.h5 --fps 10
```

## `worldkit record`

Record Gymnasium environment interactions to HDF5.

```bash
worldkit record --env <id> --output <path> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--env` | required | Gymnasium environment ID |
| `--episodes` | `100` | Number of episodes |
| `--output` | required | Output HDF5 path |
| `--max-steps` | `500` | Max steps per episode |

**Example:**

```bash
worldkit record --env CartPole-v1 --episodes 200 --output cartpole.h5
```

## `worldkit probe`

Train linear probes on the latent space.

```bash
worldkit probe --model <path> --data <path> --labels <path> --properties <list>
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | required | Path to `.wk` model file |
| `--data` | required | Path to HDF5 observations |
| `--labels` | required | Path to CSV or HDF5 labels |
| `--properties` | required | Comma-separated property names |
| `--alpha` | `1.0` | Ridge regularization strength |
| `--test-fraction` | `0.2` | Test set fraction |
| `--seed` | `42` | Random seed |

**Example:**

```bash
worldkit probe \
    --model pusht_model.wk \
    --data pusht.h5 \
    --labels pusht_labels.csv \
    --properties x_position,y_position,angle
```

## `worldkit compare`

Compare multiple world models.

```bash
worldkit compare --models <paths> --data <path> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--models` | required | Model paths (use multiple `--models` flags) |
| `--data` | required | Path to evaluation HDF5 data |
| `--episodes` | `50` | Episodes to evaluate |
| `--output` | `comparison.html` | Output HTML report path |

**Example:**

```bash
worldkit compare \
    --models nano_model.wk --models base_model.wk \
    --data eval_data.h5 \
    --output comparison.html
```

## Environment commands

### `worldkit env list`

List registered environments.

```bash
worldkit env list [--category <name>]
```

### `worldkit env info`

Show detailed environment information.

```bash
worldkit env info <env_id>
```

### `worldkit env search`

Search environments by keyword.

```bash
worldkit env search <query>
```

### `worldkit env install`

Install dependencies for an environment.

```bash
worldkit env install <env_id>
```

## Hub commands

### `worldkit hub list`

List available pre-trained models on the Hub.

```bash
worldkit hub list
```

### `worldkit hub download`

Download a model from Hugging Face.

```bash
worldkit hub download <model_id> [--output <dir>]
```

**Example:**

```bash
worldkit hub download DilpreetBansi/pusht --output ./models/
```

## Benchmark commands

### `worldkit bench run`

Run benchmark evaluation.

```bash
worldkit bench run --model <path> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | required | Path to `.wk` model file |
| `--suite` | `full` | Benchmark suite: `full`, `quick`, or a category name |
| `--episodes` | `50` | Episodes per task |
| `--seed` | `42` | Random seed |
| `--output` | none | JSON output path for results |

### `worldkit bench quick`

Quick 5-task benchmark.

```bash
worldkit bench quick --model <path> [--seed 42]
```

### `worldkit bench report`

Generate HTML report from benchmark results.

```bash
worldkit bench report --results <json_path> --output <html_path>
```

**Example:**

```bash
worldkit bench run --model model.wk --output results.json
worldkit bench report --results results.json --output report.html
```
