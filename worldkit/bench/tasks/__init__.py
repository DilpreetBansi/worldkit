"""Benchmark task implementations."""

from __future__ import annotations

from .control import AcrobotTask, CartPoleTask, PendulumTask, ReacherTask
from .games import BreakoutTask, PongTask
from .manipulation import BlockStackTask, PushTTask
from .navigation import GridWorldTask, MazeTask, TwoRoomTask

ALL_TASKS = [
    # Navigation
    TwoRoomTask,
    MazeTask,
    GridWorldTask,
    # Manipulation
    PushTTask,
    BlockStackTask,
    # Control
    CartPoleTask,
    PendulumTask,
    ReacherTask,
    AcrobotTask,
    # Games
    PongTask,
    BreakoutTask,
]

__all__ = [
    "ALL_TASKS",
    "AcrobotTask",
    "BlockStackTask",
    "BreakoutTask",
    "CartPoleTask",
    "GridWorldTask",
    "MazeTask",
    "PendulumTask",
    "PongTask",
    "PushTTask",
    "ReacherTask",
    "TwoRoomTask",
]
