# Probing

Linear probing measures what the world model's latent space has learned. By training simple linear models (Ridge regression) on frozen latent representations, you can determine whether the model encodes specific physical properties like position, velocity, or orientation.

## `ProbeResult`

```python
from worldkit import ProbeResult
```

```python
@dataclass
class ProbeResult:
    property_scores: dict[str, float]  # R² score per property
    mse_scores: dict[str, float]       # MSE per property
    probes: dict                        # Trained sklearn Ridge models
    summary: str                        # Human-readable summary
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `property_scores` | `dict[str, float]` | R² score per probed property. Higher is better (max 1.0). |
| `mse_scores` | `dict[str, float]` | Mean squared error per property. Lower is better. |
| `probes` | `dict` | Trained sklearn `Ridge` models, keyed by property name. Can be used for downstream prediction. |
| `summary` | `str` | Formatted human-readable summary of all probe results. |

## How it works

1. Observations from an HDF5 file are encoded through the frozen world model encoder
2. For each target property, a Ridge regression model is fit from latent vectors to property values
3. R² and MSE are computed on a held-out test split

A high R² score (close to 1.0) means the latent space linearly encodes that property. A low score means the property is either not represented or requires nonlinear decoding.

## Basic usage

### Via WorldModel

```python
from worldkit import WorldModel

model = WorldModel.load("pusht_model.wk")

result = model.probe(
    data="pusht.h5",
    properties=["x_position", "y_position", "angle"],
    labels="pusht_labels.csv",
)

print(result.summary)
# x_position: R²=0.92, MSE=0.0034
# y_position: R²=0.89, MSE=0.0051
# angle:      R²=0.78, MSE=0.0123
```

### Via LinearProbe class

```python
from worldkit.eval import LinearProbe

probe = LinearProbe(model)
result = probe.fit(
    data_path="pusht.h5",
    properties=["x_position", "y_position"],
    labels_path="pusht_labels.csv",
    alpha=1.0,
    test_fraction=0.2,
    seed=42,
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `str \| Path` | required | Path to HDF5 file containing pixel observations. |
| `properties` | `list[str]` | required | Property names to probe. Must exist as columns in the labels file. |
| `labels` | `str \| Path` | required | Path to CSV or HDF5 file with per-frame property values. |
| `alpha` | `float` | `1.0` | Ridge regularization strength. Higher values reduce overfitting. |
| `test_fraction` | `float` | `0.2` | Fraction of data reserved for evaluation. |
| `seed` | `int` | `42` | Random seed for train/test split. |

## Label file format

**CSV format** — one row per frame, columns for each property:

```csv
x_position,y_position,angle
0.45,0.32,1.57
0.46,0.33,1.58
...
```

**HDF5 format** — datasets named by property:

```
labels.h5
├── x_position: (N,)
├── y_position: (N,)
└── angle: (N,)
```

The number of label rows must match the number of frames in the observation HDF5 file.

## Using trained probes

The `probes` field contains trained sklearn `Ridge` models that can predict property values from latent vectors:

```python
from worldkit.eval import LinearProbe

probe = LinearProbe(model)

# Predict properties for a new observation
obs = np.array(...)  # (96, 96, 3)
predictions = probe.predict(obs, result.probes)
print(predictions)
# {"x_position": 0.47, "y_position": 0.31, "angle": 1.55}
```

## Interpreting results

| R² Score | Interpretation |
|----------|---------------|
| > 0.9 | Excellent — property is linearly encoded in the latent space |
| 0.7 - 0.9 | Good — property is well-represented, some nonlinear structure |
| 0.4 - 0.7 | Moderate — property is partially captured |
| < 0.4 | Poor — property is not linearly accessible from the latent space |

## CLI

```bash
worldkit probe \
    --model pusht_model.wk \
    --data pusht.h5 \
    --labels pusht_labels.csv \
    --properties x_position,y_position,angle
```

## Related

- [WorldModel.probe()](worldmodel.md#probe) — method reference
- [Latent visualization](worldmodel.md#visualize_latent_space) — visualize the latent space structure
- [Architecture guide](../guides/architecture.md) — how the encoder creates latent representations
