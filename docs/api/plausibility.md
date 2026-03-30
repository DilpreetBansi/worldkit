# Plausibility

WorldKit can score whether a sequence of observations is physically plausible. This is useful for anomaly detection, quality assurance, and violation-of-expectation testing.

## How it works

The plausibility scorer:

1. Encodes each frame in the sequence into the latent space
2. For each consecutive pair of frames, predicts the next latent from the current latent
3. Computes the prediction error (MSE between predicted and actual next latent)
4. Applies exponential decay to aggregate errors into a single score

**High prediction error** = the world model did not expect this transition = anomaly.

```
Frame 1 → encode → z₁ → predict → z₁' → compare with z₂ → error₁
Frame 2 → encode → z₂ → predict → z₂' → compare with z₃ → error₂
Frame 3 → encode → z₃ → predict → z₃' → compare with z₄ → error₃
...
Score = exp(-mean(errors))  →  [0.0, 1.0]
```

## Return value

`plausibility()` returns a single `float`:

| Score | Meaning |
|-------|---------|
| 1.0 | Fully expected behavior — transitions match the model's predictions |
| 0.5-0.8 | Minor deviations — unusual but not impossible |
| 0.2-0.5 | Significant anomalies — transitions the model finds unlikely |
| 0.0-0.2 | Physically implausible — the model strongly rejects these transitions |

## Basic usage

```python
import numpy as np
from worldkit import WorldModel

model = WorldModel.load("factory_model.wk")

# Score a video sequence
frames = [np.array(...) for _ in range(100)]  # list of (H, W, C) images
score = model.plausibility(frames)
print(f"Plausibility: {score:.3f}")
```

## Anomaly detection

Set a threshold to flag anomalies:

```python
threshold = 0.3

for video in video_stream:
    frames = extract_frames(video)
    score = model.plausibility(frames)

    if score < threshold:
        print(f"ANOMALY DETECTED: score={score:.3f}")
        alert(video)
```

## Sliding window detection

For real-time monitoring, score overlapping windows:

```python
window_size = 30
stride = 10
frame_buffer = []

for frame in live_camera_feed():
    frame_buffer.append(frame)

    if len(frame_buffer) >= window_size:
        window = frame_buffer[-window_size:]
        score = model.plausibility(window)

        if score < 0.3:
            print(f"Anomaly at frame {len(frame_buffer)}: {score:.3f}")

        # Slide window
        if len(frame_buffer) > window_size + stride:
            frame_buffer = frame_buffer[stride:]
```

## Use cases

- **Manufacturing QA** — detect defects on an assembly line by scoring production footage
- **Robotics safety** — flag unexpected physical interactions
- **Video surveillance** — identify unusual events in security footage
- **Simulation validation** — verify that generated physics look realistic

## Related

- [WorldModel.plausibility()](worldmodel.md#plausibility) — method reference
- [Anomaly detection tutorial](../tutorials/anomaly_detection.md) — full walkthrough
