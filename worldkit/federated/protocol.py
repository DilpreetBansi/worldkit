"""Federated learning protocol — message types and weight delta serialization.

Messages are JSON dicts sent over WebSocket with a ``type`` field.
Weight deltas are serialized via safetensors to bytes, then base64 encoded
so they can be embedded in JSON without binary framing.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import torch
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

# ── Message types ────────────────────────────────────────


class MessageType(str, Enum):
    """All message types exchanged between client and server."""

    REGISTER = "register"
    ROUND_START = "round_start"
    UPDATE = "update"
    GLOBAL_WEIGHTS = "global_weights"
    DONE = "done"
    ERROR = "error"


# ── Message dataclasses ─────────────────────────────────


@dataclass
class RegisterMessage:
    """Sent by client on connect."""

    client_id: str
    num_params: int


@dataclass
class RoundStartMessage:
    """Sent by server at the beginning of each round."""

    round_number: int
    global_weights_b64: str | None = None


@dataclass
class UpdateMessage:
    """Sent by client after local training."""

    client_id: str
    round_number: int
    delta_b64: str
    num_samples: int
    train_loss: float


@dataclass
class GlobalWeightsMessage:
    """Sent by server after aggregation."""

    round_number: int
    aggregated_delta_b64: str
    num_clients: int


@dataclass
class DoneMessage:
    """Sent by server when all rounds are complete."""

    reason: str
    final_round: int


@dataclass
class ErrorMessage:
    """Sent on protocol errors."""

    detail: str


# ── Serialization helpers ────────────────────────────────

_MSG_CLS_BY_TYPE: dict[str, type] = {
    MessageType.REGISTER: RegisterMessage,
    MessageType.ROUND_START: RoundStartMessage,
    MessageType.UPDATE: UpdateMessage,
    MessageType.GLOBAL_WEIGHTS: GlobalWeightsMessage,
    MessageType.DONE: DoneMessage,
    MessageType.ERROR: ErrorMessage,
}


def to_json(msg: Any) -> dict:
    """Serialize a message dataclass to a JSON-compatible dict.

    Adds a ``type`` field based on the message class.
    """
    type_map: dict[type, str] = {v: k for k, v in _MSG_CLS_BY_TYPE.items()}
    msg_type = type_map.get(type(msg))
    if msg_type is None:
        raise ValueError(f"Unknown message class: {type(msg).__name__}")
    data = asdict(msg)
    data["type"] = str(msg_type.value) if isinstance(msg_type, MessageType) else str(msg_type)
    return data


def from_json(data: dict) -> Any:
    """Deserialize a JSON dict into the appropriate message dataclass."""
    msg_type = data.get("type")
    if msg_type is None:
        raise ValueError("Message missing 'type' field")
    cls = _MSG_CLS_BY_TYPE.get(msg_type)
    if cls is None:
        raise ValueError(f"Unknown message type: {msg_type}")
    # Filter to only fields the dataclass expects
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


# ── Weight delta encoding ────────────────────────────────


def encode_delta(state_dict: dict[str, torch.Tensor]) -> str:
    """Serialize a state dict to a base64 string via safetensors.

    Args:
        state_dict: Dictionary of parameter name → tensor.

    Returns:
        Base64-encoded string of the safetensors bytes.
    """
    # safetensors requires contiguous tensors
    contiguous = {k: v.contiguous() for k, v in state_dict.items()}
    raw_bytes = safetensors_save(contiguous)
    return base64.b64encode(raw_bytes).decode("ascii")


def decode_delta(b64_string: str) -> dict[str, torch.Tensor]:
    """Deserialize a base64 string back to a state dict.

    Args:
        b64_string: Base64-encoded safetensors bytes.

    Returns:
        Dictionary of parameter name → tensor.
    """
    raw_bytes = base64.b64decode(b64_string)
    return safetensors_load(raw_bytes)


# ── Delta arithmetic ─────────────────────────────────────


def compute_delta(
    initial: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute the weight delta: current - initial.

    Args:
        initial: Snapshot of weights before local training.
        current: Weights after local training.

    Returns:
        Delta dict with the same keys.
    """
    return {k: current[k].cpu() - initial[k].cpu() for k in initial}


def apply_delta(
    state_dict: dict[str, torch.Tensor],
    delta: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply a weight delta to a state dict: state_dict + delta.

    Args:
        state_dict: Base weights.
        delta: Weight delta to add.

    Returns:
        New state dict with updated weights.
    """
    return {k: state_dict[k].cpu() + delta[k].cpu() for k in state_dict}
