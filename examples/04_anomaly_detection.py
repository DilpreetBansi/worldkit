"""Anomaly detection using plausibility scoring."""

from worldkit import WorldModel
from worldkit.core.config import get_config
from worldkit.core.jepa import JEPA
import numpy as np

config = get_config("nano", action_dim=2)
jepa = JEPA.from_config(config)
model = WorldModel(jepa, config, device="cpu")

normal_frames = []
base = np.random.rand(96, 96, 3).astype(np.float32) * 0.5
for i in range(10):
    frame = base + np.random.rand(96, 96, 3).astype(np.float32) * 0.1
    normal_frames.append(frame.clip(0, 1))

anomaly_frames = [np.random.rand(96, 96, 3).astype(np.float32) for _ in range(10)]

normal_score = model.plausibility(normal_frames)
anomaly_score = model.plausibility(anomaly_frames)

print(f"Normal sequence plausibility:  {normal_score:.3f}")
print(f"Anomaly sequence plausibility: {anomaly_score:.3f}")
