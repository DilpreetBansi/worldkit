# Quickstart

## Installation

```bash
pip install worldkit
```

## Create a Model

```python
from worldkit import WorldModel
from worldkit.core.config import get_config
from worldkit.core.jepa import JEPA

config = get_config("nano", action_dim=2)
jepa = JEPA.from_config(config)
model = WorldModel(jepa, config, device="cpu")
```

## Encode, Predict, Plan

```python
import numpy as np

obs = np.random.rand(96, 96, 3).astype(np.float32)
z = model.encode(obs)

actions = [np.array([0.1, 0.2])] * 10
result = model.predict(obs, actions)

goal = np.random.rand(96, 96, 3).astype(np.float32)
plan = model.plan(obs, goal, max_steps=20)
```

## Train from Data

```python
model = WorldModel.train(data="my_data.h5", config="base", epochs=100)
model.save("my_model.wk")
```
