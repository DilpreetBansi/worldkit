#!/usr/bin/env python3
"""Generate per-frame property labels for Push-T data.

Reads a Push-T HDF5 dataset (with environment state) and extracts
physical properties (agent position, block angle) as a CSV file
suitable for linear probing with `worldkit probe`.

Usage:
    python scripts/generate_pusht_labels.py \\
        --data pusht_data.h5 \\
        --output pusht_labels.csv

The HDF5 file should contain:
    - pixels / observations / obs / images: (N_episodes, T, H, W, C)
    - state (optional): (N_episodes, T, state_dim) with columns:
        [agent_x, agent_y, block_x, block_y, block_angle, ...]

If no 'state' key exists, the script attempts to reconstruct labels
from 'agent_pos' and 'block_pos' / 'block_angle' keys.

Output CSV columns: frame_idx, agent_x, agent_y, block_angle
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np


def extract_labels(data_path: str | Path) -> list[dict]:
    """Extract per-frame labels from a Push-T HDF5 file.

    Args:
        data_path: Path to the HDF5 dataset.

    Returns:
        List of dicts with keys: frame_idx, agent_x, agent_y, block_angle.
    """
    data_path = Path(data_path)
    rows: list[dict] = []

    with h5py.File(data_path, "r") as f:
        keys = list(f.keys())

        # Strategy 1: 'state' array with known column layout
        if "state" in f:
            state = np.array(f["state"])
            # Flatten episodes: (N_eps, T, D) → (N_eps*T, D)
            if state.ndim == 3:
                state = state.reshape(-1, state.shape[-1])
            for i in range(state.shape[0]):
                rows.append(
                    {
                        "frame_idx": i,
                        "agent_x": float(state[i, 0]),
                        "agent_y": float(state[i, 1]),
                        "block_angle": float(state[i, 4]) if state.shape[1] > 4 else 0.0,
                    }
                )
            return rows

        # Strategy 2: separate keys for agent_pos, block_angle
        agent_pos = None
        block_angle = None

        for key in ("agent_pos", "agent_position"):
            if key in f:
                agent_pos = np.array(f[key])
                break

        for key in ("block_angle", "block_rotation"):
            if key in f:
                block_angle = np.array(f[key])
                break

        if agent_pos is not None:
            # Flatten episodes if needed
            if agent_pos.ndim == 3:
                agent_pos = agent_pos.reshape(-1, agent_pos.shape[-1])
            if block_angle is not None and block_angle.ndim == 2:
                block_angle = block_angle.reshape(-1)

            for i in range(agent_pos.shape[0]):
                rows.append(
                    {
                        "frame_idx": i,
                        "agent_x": float(agent_pos[i, 0]),
                        "agent_y": float(agent_pos[i, 1]),
                        "block_angle": float(block_angle[i]) if block_angle is not None else 0.0,
                    }
                )
            return rows

        # No recognized state keys
        raise KeyError(
            f"Cannot extract labels from HDF5. "
            f"Expected 'state', 'agent_pos', or 'agent_position'. "
            f"Found keys: {keys}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-frame property labels for Push-T data."
    )
    parser.add_argument("--data", required=True, help="Path to Push-T HDF5 file")
    parser.add_argument("--output", default="pusht_labels.csv", help="Output CSV path")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    rows = extract_labels(data_path)
    output_path = Path(args.output)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "agent_x", "agent_y", "block_angle"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} labels to {output_path}")


if __name__ == "__main__":
    main()
