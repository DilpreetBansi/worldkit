"""Convert the LeWM Push-T dataset to WorldKit format.

The LeWM dataset uses a flat layout (all steps concatenated) with
ep_offset/ep_len arrays. This script extracts episodes, resizes
frames to 96x96, and saves in the (N, T, H, W, C) format WorldKit expects.

Usage:
    python scripts/prepare_pusht_real.py [--episodes 200] [--seq-len 16]
"""

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import hdf5plugin  # noqa: F401 — registers Blosc filter
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/pusht_expert_train.h5",
        help="Path to LeWM Push-T HDF5",
    )
    parser.add_argument("--output", default="data/pusht_real.h5", help="Output path")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes to extract")
    parser.add_argument("--seq-len", type=int, default=16, help="Steps per episode window")
    parser.add_argument("--image-size", type=int, default=96, help="Target image size")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Reading {input_path}...")
    with h5py.File(input_path, "r") as f:
        ep_offsets = f["ep_offset"][:]
        ep_lens = f["ep_len"][:]
        total_episodes = len(ep_offsets)
        print(f"  Total episodes: {total_episodes}")
        print(f"  Total steps: {f['pixels'].shape[0]}")
        print(f"  Image size: {f['pixels'].shape[1]}x{f['pixels'].shape[2]}")

        n_episodes = min(args.episodes, total_episodes)
        seq_len = args.seq_len
        img_size = args.image_size

        # Select episodes that are long enough
        valid_indices = [i for i in range(total_episodes) if ep_lens[i] >= seq_len]
        if len(valid_indices) < n_episodes:
            print(f"  Warning: only {len(valid_indices)} episodes >= {seq_len} steps")
            n_episodes = len(valid_indices)

        # Randomly sample episodes for diversity
        rng = np.random.RandomState(42)
        chosen = rng.choice(valid_indices, size=n_episodes, replace=False)
        chosen.sort()

        print(f"  Extracting {n_episodes} episodes, {seq_len} steps each, resizing to {img_size}x{img_size}")

        all_pixels = np.zeros(
            (n_episodes, seq_len, img_size, img_size, 3), dtype=np.uint8
        )
        all_actions = np.zeros((n_episodes, seq_len, 2), dtype=np.float32)

        for out_idx, ep_idx in enumerate(tqdm(chosen, desc="Extracting episodes")):
            offset = int(ep_offsets[ep_idx])
            length = int(ep_lens[ep_idx])

            # Take a random window within the episode
            max_start = length - seq_len
            start = rng.randint(0, max_start + 1)
            global_start = offset + start

            # Read frames
            frames = f["pixels"][global_start : global_start + seq_len]
            actions = f["action"][global_start : global_start + seq_len]

            # Resize frames
            for t in range(seq_len):
                all_pixels[out_idx, t] = cv2.resize(
                    frames[t], (img_size, img_size), interpolation=cv2.INTER_AREA
                )
            all_actions[out_idx] = actions

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {output_path}...")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("pixels", data=all_pixels, compression="gzip", compression_opts=4)
        f.create_dataset("actions", data=all_actions, compression="gzip", compression_opts=4)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  pixels: {all_pixels.shape} ({all_pixels.dtype})")
    print(f"  actions: {all_actions.shape} ({all_actions.dtype})")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Action range: [{all_actions.min():.3f}, {all_actions.max():.3f}]")
    print("Done!")


if __name__ == "__main__":
    main()
