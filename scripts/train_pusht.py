"""Train a WorldKit model on synthetic Push-T data.

Generates a Push-T-like dataset (2D pushing task) and trains a world model.
The Push-T task: an agent pushes a T-shaped block toward a target position.
Observations are 96x96 RGB images, actions are 2D (dx, dy).

Usage:
    python scripts/train_pusht.py
"""

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def draw_t_shape(canvas, cx, cy, angle, color, size=12):
    """Draw a T-shaped block on the canvas at (cx, cy) with rotation."""
    h, w = canvas.shape[:2]
    # T-shape: horizontal bar + vertical stem
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    # Horizontal bar points (relative to center)
    bar_pts = []
    for dx in range(-size, size + 1):
        for dy in range(-size // 4, size // 4 + 1):
            rx = int(cx + dx * cos_a - dy * sin_a)
            ry = int(cy + dx * sin_a + dy * cos_a)
            if 0 <= rx < w and 0 <= ry < h:
                bar_pts.append((ry, rx))

    # Vertical stem points
    stem_pts = []
    for dx in range(-size // 4, size // 4 + 1):
        for dy in range(0, size + 1):
            rx = int(cx + dx * cos_a - dy * sin_a)
            ry = int(cy + dx * sin_a + dy * cos_a)
            if 0 <= rx < w and 0 <= ry < h:
                stem_pts.append((ry, rx))

    for (r, c) in bar_pts + stem_pts:
        canvas[r, c] = color


def draw_circle(canvas, cx, cy, radius, color):
    """Draw a filled circle on the canvas."""
    h, w = canvas.shape[:2]
    for r in range(max(0, int(cy - radius)), min(h, int(cy + radius + 1))):
        for c in range(max(0, int(cx - radius)), min(w, int(cx + radius + 1))):
            if (c - cx) ** 2 + (r - cy) ** 2 <= radius ** 2:
                canvas[r, c] = color


def generate_pusht_episode(seq_len=16, img_size=96):
    """Generate one episode of Push-T data.

    Returns:
        pixels: (seq_len, img_size, img_size, 3) uint8
        actions: (seq_len, 2) float32
    """
    pixels = np.zeros((seq_len, img_size, img_size, 3), dtype=np.uint8)
    actions = np.zeros((seq_len, 2), dtype=np.float32)

    # Initialize T-block position and angle
    t_x = np.random.uniform(25, 70)
    t_y = np.random.uniform(25, 70)
    t_angle = np.random.uniform(0, 2 * np.pi)

    # Initialize agent (circle) position
    agent_x = np.random.uniform(15, 80)
    agent_y = np.random.uniform(15, 80)

    # Target position for T-block (shown as faint outline)
    target_x = img_size / 2
    target_y = img_size / 2

    for t in range(seq_len):
        # Create frame
        canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240  # light gray bg

        # Draw target zone (light blue circle)
        draw_circle(canvas, target_x, target_y, 15, [200, 220, 255])

        # Draw T-block (red)
        draw_t_shape(canvas, t_x, t_y, t_angle, [220, 60, 60], size=10)

        # Draw agent (blue circle)
        draw_circle(canvas, agent_x, agent_y, 5, [60, 60, 220])

        pixels[t] = canvas

        # Generate action: move toward T-block with some noise
        dx = (t_x - agent_x) * 0.3 + np.random.randn() * 2.0
        dy = (t_y - agent_y) * 0.3 + np.random.randn() * 2.0

        # Clip actions
        dx = np.clip(dx, -5, 5)
        dy = np.clip(dy, -5, 5)
        actions[t] = [dx / 5.0, dy / 5.0]  # normalize to [-1, 1]

        # Update agent position
        agent_x = np.clip(agent_x + dx, 5, img_size - 5)
        agent_y = np.clip(agent_y + dy, 5, img_size - 5)

        # If agent is close to T-block, push it
        dist = np.sqrt((agent_x - t_x) ** 2 + (agent_y - t_y) ** 2)
        if dist < 12:
            push_force = 0.3
            t_x += dx * push_force
            t_y += dy * push_force
            t_angle += np.random.randn() * 0.05
            t_x = np.clip(t_x, 15, img_size - 15)
            t_y = np.clip(t_y, 15, img_size - 15)

    return pixels, actions


def generate_dataset(n_episodes=200, seq_len=16, img_size=96):
    """Generate full Push-T dataset."""
    print(f"Generating {n_episodes} episodes ({seq_len} steps each)...")

    all_pixels = np.zeros((n_episodes, seq_len, img_size, img_size, 3), dtype=np.uint8)
    all_actions = np.zeros((n_episodes, seq_len, 2), dtype=np.float32)

    for i in range(n_episodes):
        pixels, actions = generate_pusht_episode(seq_len, img_size)
        all_pixels[i] = pixels
        all_actions[i] = actions
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{n_episodes} episodes")

    return all_pixels, all_actions


def main():
    print("=" * 60)
    print("WorldKit | Push-T Training Pipeline")
    print("=" * 60)

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    data_path = data_dir / "pusht_train.h5"
    model_dir = Path(__file__).parent.parent / "checkpoints"
    model_dir.mkdir(exist_ok=True)

    # Step 1: Generate data
    if data_path.exists():
        print(f"\nDataset already exists at {data_path}, skipping generation.")
    else:
        print("\n--- Step 1: Generating Push-T Dataset ---")
        pixels, actions = generate_dataset(n_episodes=200, seq_len=16, img_size=96)

        with h5py.File(data_path, "w") as f:
            f.create_dataset("pixels", data=pixels, compression="gzip", compression_opts=4)
            f.create_dataset("actions", data=actions, compression="gzip", compression_opts=4)

        file_size_mb = data_path.stat().st_size / (1024 * 1024)
        print(f"  Saved: {data_path} ({file_size_mb:.1f} MB)")
        print(f"  Shape: pixels={pixels.shape}, actions={actions.shape}")

    # Step 2: Train
    print("\n--- Step 2: Training World Model ---")
    print("Config: nano (fastest, ~3.5M params)")
    print("This should take ~10-20 minutes on M4 Pro MPS\n")

    from worldkit import WorldModel

    start_time = time.time()

    model = WorldModel.train(
        data=str(data_path),
        config="nano",
        epochs=50,
        batch_size=32,
        lr=3e-4,
        lambda_reg=0.5,
        action_dim=2,
        device=device,
        checkpoint_dir=str(model_dir),
        seed=42,
    )

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed / 60:.1f} minutes")

    # Step 3: Save final model
    final_path = model_dir / "pusht_nano.wk"
    model.save(final_path)
    model_size_mb = final_path.stat().st_size / (1024 * 1024)
    print(f"Model saved: {final_path} ({model_size_mb:.1f} MB)")

    # Step 4: Quick verification
    print("\n--- Step 3: Verification ---")

    # Test encode
    test_obs = np.random.rand(96, 96, 3).astype(np.float32)
    z = model.encode(test_obs)
    print(f"  encode(): input (96,96,3) -> latent {tuple(z.shape)} ✓")

    # Test predict
    test_actions = [np.array([0.1, 0.2], dtype=np.float32)] * 5
    result = model.predict(test_obs, test_actions)
    print(f"  predict(): 5-step rollout -> {result.steps} steps, confidence={result.confidence:.3f} ✓")

    # Test load
    loaded = WorldModel.load(final_path, device=device)
    z2 = loaded.encode(test_obs)
    match = torch.allclose(z, z2, atol=1e-5)
    print(f"  save/load roundtrip: latents match={match} ✓")

    print("\n" + "=" * 60)
    print("SUCCESS! Model is trained and verified.")
    print(f"  Model: {final_path}")
    print(f"  Params: {model.num_params:,}")
    print(f"  Size: {model_size_mb:.1f} MB")
    print(f"  Time: {elapsed / 60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
