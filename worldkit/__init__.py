"""WorldKit — The open-source world model runtime.

Train physics-aware AI on a laptop. Deploy anywhere.

Usage:
    from worldkit import WorldModel

    model = WorldModel.train(data="my_data.h5")
    plan = model.plan(current_frame, goal_frame)
"""

__version__ = "0.1.0"

from worldkit.core.config import ModelConfig, get_config
from worldkit.core.model import PredictionResult, ProbeResult, WorldModel
from worldkit.core.planner import PlanResult

__all__ = [
    "WorldModel",
    "PredictionResult",
    "PlanResult",
    "ProbeResult",
    "get_config",
    "ModelConfig",
]
