"""WorldKit — The open-source world model SDK.

Train, predict, plan, and deploy world models from a single Python interface.
Built on the JEPA architecture with SIGReg training (Maes et al., 2026).

Usage:
    from worldkit import WorldModel

    model = WorldModel.train(data="my_data.h5", config="base", epochs=100)
    result = model.predict(current_frame, actions)
    plan = model.plan(current_frame, goal_frame)
"""

__version__ = "0.2.0"

from worldkit.core.config import ModelConfig, get_config
from worldkit.core.hierarchical_planner import HierarchicalPlanResult
from worldkit.core.model import PredictionResult, ProbeResult, WorldModel
from worldkit.core.online import OnlineLearner
from worldkit.core.planner import PlanResult

__all__ = [
    "WorldModel",
    "PredictionResult",
    "PlanResult",
    "ProbeResult",
    "HierarchicalPlanResult",
    "OnlineLearner",
    "get_config",
    "ModelConfig",
]
