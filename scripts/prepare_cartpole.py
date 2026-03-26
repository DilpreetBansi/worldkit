"""Collect CartPole pixel observations from Gymnasium and save in WorldKit HDF5 format.

Renders CartPole-v1 as RGB frames, resizes to 96x96, and stores episodes
in the standard WorldKit layout:
    pixels:  (N, T, 96, 96, 3)  uint8
    actions: (N, T, 1)          float32

Usage:
    python scripts/prepare_cartpole.py
    python scripts/prepare_cartpole.py --episodes 500 --seq-len 32 --output data/cartpole_big.h5
"""

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


def resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize an RGB frame to (size, size) using LANCZOS interpolation."""
    img = Image.fromarray(frame)
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def collect_episodes(
    n_episodes: int,
    seq_len: int,
    img_size: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect CartPole episodes with pixel observations.

    If an episode terminates before seq_len steps, the environment is reset
    and collection continues so that every episode slot has exactly seq_len
    frames.

    Returns:
        pixels:  (n_episodes, seq_len, img_size, img_size, 3) uint8
        actions: (n_episodes, seq_len, 1) float32
    """
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    all_pixels = np.zeros(
        (n_episodes, seq_len, img_size, img_size, 3), dtype=np.uint8
    )
    all_actions = np.zeros((n_episodes, seq_len, 1), dtype=np.float32)

    rng = np.random.RandomState(seed)
    total_resets = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(rng.randint(0, 2**31)))
        frame = env.render()
        step = 0

        while step < seq_len:
            # Resize and store the current frame
            resized = resize_frame(frame, img_size)
            all_pixels[ep, step] = resized

            # Random policy: choose 0 or 1
            action = int(rng.randint(0, 2))
            all_actions[ep, step, 0] = float(action)

            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()

            step += 1

            # If the episode ended early, reset and keep collecting
            if (terminated or truncated) and step < seq_len:
                obs, info = env.reset(seed=int(rng.randint(0, 2**31)))
                frame = env.render()
                total_resets += 1

        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"  Collected {ep + 1}/{n_episodes} episodes")

    env.close()

    print(f"  Mid-episode resets (early terminations): {total_resets}")
    return all_pixels, all_actions


def main():
    parser = argparse.ArgumentParser(
        description="Collect CartPole pixel data for WorldKit training."
    )
    parser.add_argument(
        "--episodes", type=int, default=200, help="Number of episodes to collect"
    )
    parser.add_argument(
        "--seq-len", type=int, default=16, help="Timesteps per episode"
    )
    parser.add_argument(
        "--image-size", type=int, default=96, help="Output image size (square)"
    )
    parser.add_argument(
        "--output",
        default="data/cartpole_train.h5",
        help="Output HDF5 path (relative to project root)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Resolve output path relative to project root
    project_root = Path(__file__).parent.parent
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("WorldKit | CartPole Data Collection")
    print("=" * 60)
    print(f"  Episodes:   {args.episodes}")
    print(f"  Seq length: {args.seq_len}")
    print(f"  Image size: {args.image_size}x{args.image_size}")
    print(f"  Output:     {output_path}")
    print(f"  Seed:       {args.seed}")
    print()

    # Collect data
    pixels, actions = collect_episodes(
        n_episodes=args.episodes,
        seq_len=args.seq_len,
        img_size=args.image_size,
        seed=args.seed,
    )

    # Save to HDF5
    print(f"\nSaving to {output_path}...")
    with h5py.File(output_path, "w") as f:
        f.create_dataset(
            "pixels", data=pixels, compression="gzip", compression_opts=4
        )
        f.create_dataset(
            "actions", data=actions, compression="gzip", compression_opts=4
        )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    # Print stats
    print()
    print("--- Dataset Statistics ---")
    print(f"  pixels shape:  {pixels.shape}  dtype={pixels.dtype}")
    print(f"  actions shape: {actions.shape}  dtype={actions.dtype}")
    print(f"  File size:     {file_size_mb:.1f} MB")
    print(f"  Pixel range:   [{pixels.min()}, {pixels.max()}]")
    print(f"  Action values: {np.unique(actions)}")
    print(
        f"  Total frames:  {args.episodes * args.seq_len:,} "
        f"({args.episodes} episodes x {args.seq_len} steps)"
    )
    print()
    print("Done! Use this data with:")
    print(f'  WorldModel.train(data="{args.output}", config="base", action_dim=1)')


if __name__ == "__main__":
    main()
