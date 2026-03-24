# Model Zoo

Pre-trained models available on WorldKit Hub (Hugging Face).

## Navigation Models

| Model | Config | Description |
|-------|--------|-------------|
| worldkit/two-room-nav | base | Navigate between rooms |
| worldkit/maze-nav | base | Solve procedural mazes |
| worldkit/grid-world | nano | Multi-goal grid navigation |

## Manipulation Models

| Model | Config | Description |
|-------|--------|-------------|
| worldkit/pusht | base | Push T-block to target |
| worldkit/block-stack | base | Stack colored blocks |
| worldkit/cube-3d | base | 6-DOF pick-and-place |

## Control Models

| Model | Config | Description |
|-------|--------|-------------|
| worldkit/reacher | base | 2-joint arm reaching |
| worldkit/cartpole | nano | Balance pendulum |
| worldkit/pendulum | nano | Swing-up and balance |

## Usage

```python
from worldkit import WorldModel
model = WorldModel.from_hub("worldkit/pusht")
```
