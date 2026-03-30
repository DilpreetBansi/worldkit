# Anomaly detection

This tutorial shows how to use WorldKit's plausibility scoring for anomaly detection. A trained world model learns what "normal" looks like — anything that violates its expectations is flagged as anomalous.

## Prerequisites

```bash
pip install worldkit
```

A world model trained on "normal" data:

```python
from worldkit import WorldModel

# Train on normal factory footage, normal robot behavior, etc.
model = WorldModel.train(data="normal_data.h5", config="base", epochs=100)
```

Or load a pre-trained model:

```python
model = WorldModel.load("factory_model.wk")
```

## How it works

The plausibility scorer computes how well consecutive frame transitions match the model's expectations:

1. Each frame is encoded into the latent space
2. The predictor predicts what the next frame's latent should be
3. The prediction error (MSE) is measured against the actual next frame
4. Errors are aggregated into a score from 0.0 (impossible) to 1.0 (expected)

Normal sequences produce low prediction errors (high plausibility).
Anomalous sequences produce high prediction errors (low plausibility).

## Basic anomaly detection

```python
import numpy as np

# Score a sequence of frames
frames = load_video_frames("test_video.mp4")  # list of (H, W, C) arrays
score = model.plausibility(frames)

print(f"Plausibility: {score:.3f}")
if score < 0.3:
    print("ANOMALY DETECTED")
else:
    print("Normal behavior")
```

## Setting a threshold

The right threshold depends on your use case. Evaluate on labeled data to find it:

```python
# Score normal videos
normal_scores = []
for video_path in normal_video_paths:
    frames = load_video_frames(video_path)
    score = model.plausibility(frames)
    normal_scores.append(score)

# Score anomalous videos
anomaly_scores = []
for video_path in anomaly_video_paths:
    frames = load_video_frames(video_path)
    score = model.plausibility(frames)
    anomaly_scores.append(score)

print(f"Normal scores:  mean={np.mean(normal_scores):.3f}, "
      f"min={np.min(normal_scores):.3f}")
print(f"Anomaly scores: mean={np.mean(anomaly_scores):.3f}, "
      f"max={np.max(anomaly_scores):.3f}")

# Choose threshold between the distributions
threshold = (np.min(normal_scores) + np.max(anomaly_scores)) / 2
print(f"Recommended threshold: {threshold:.3f}")
```

## Real-time monitoring

For live camera feeds, score sliding windows:

```python
window_size = 30  # frames per window
stride = 10       # slide by 10 frames
threshold = 0.3

frame_buffer = []

for frame in camera_stream():
    frame_buffer.append(frame)

    if len(frame_buffer) >= window_size:
        window = frame_buffer[-window_size:]
        score = model.plausibility(window)

        if score < threshold:
            timestamp = get_timestamp()
            print(f"[{timestamp}] Anomaly detected: score={score:.3f}")
            save_clip(window, f"anomaly_{timestamp}.mp4")

        # Keep buffer manageable
        if len(frame_buffer) > window_size * 2:
            frame_buffer = frame_buffer[-window_size:]
```

## Batch processing

Score multiple video files:

```python
import json
from pathlib import Path

results = {}
video_dir = Path("./surveillance_videos/")

for video_path in sorted(video_dir.glob("*.mp4")):
    frames = load_video_frames(str(video_path))
    score = model.plausibility(frames)
    results[video_path.name] = {
        "score": score,
        "anomaly": score < 0.3,
        "num_frames": len(frames),
    }
    status = "ANOMALY" if score < 0.3 else "OK"
    print(f"{video_path.name}: {score:.3f} [{status}]")

# Save results
with open("anomaly_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

## Training for anomaly detection

For best results, train only on "normal" data. The model learns what normal transitions look like — anything else becomes anomalous by definition.

```python
# 1. Record normal behavior
from worldkit.data import Recorder
import gymnasium as gym

env = gym.make("YourEnv-v1", render_mode="rgb_array")
recorder = Recorder(env, output="normal_data.h5")
recorder.record(episodes=500)

# 2. Train on normal data only
model = WorldModel.train(
    data="normal_data.h5",
    config="base",
    epochs=200,  # more epochs = better anomaly detection
    lambda_reg=1.0,
)
model.save("anomaly_detector.wk")
```

## Use cases

| Domain | Normal data | Anomalies detected |
|--------|------------|-------------------|
| Manufacturing | Normal production line footage | Defective products, equipment malfunctions |
| Robotics | Successful task executions | Collisions, dropped objects, unexpected contact |
| Surveillance | Routine activity | Unusual behavior, unauthorized access |
| Simulation | Physically correct renders | Physics glitches, rendering artifacts |

## Combining with a REST API

Deploy the anomaly detector as a service:

```bash
worldkit serve --model anomaly_detector.wk --port 8000
```

```python
import requests

# Score a video via the API
with open("test_video.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8000/plausibility",
        files={"video": f},
    )

result = response.json()
print(f"Score: {result['plausibility_score']:.3f}")
print(f"Anomaly: {result['anomaly_detected']}")
```

## Next steps

- [Plausibility API reference](../api/plausibility.md) — detailed scoring docs
- [REST API reference](../rest_api.md) — deploy as a service
- [Export and deploy](export_deploy.md) — production deployment
