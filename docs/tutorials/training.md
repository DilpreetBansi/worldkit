# Training Guide

## Data Preparation

WorldKit expects HDF5 files with:
- `pixels` or `observations`: shape (N, T, H, W, C)
- `actions`: shape (N, T, action_dim)

### From Gymnasium

```python
from worldkit.data import Recorder
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="rgb_array")
recorder = Recorder(env, output="data.h5")
recorder.record(episodes=1000)
```

### From Video

```python
from worldkit.data import Converter

converter = Converter()
converter.from_video(input_dir="./videos/", output="data.h5", fps=10)
```

## Training

```python
from worldkit import WorldModel

model = WorldModel.train(
    data="data.h5",
    config="base",
    epochs=100,
    lambda_reg=1.0,
    device="auto",
)
model.save("my_model.wk")
```

## CLI Training

```bash
worldkit train --data data.h5 --config base --epochs 100
```
