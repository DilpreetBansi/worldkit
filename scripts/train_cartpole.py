"""Train WorldKit world models on CartPole pixel data.

Trains both a 'base' and a 'nano' variant, saves them as .wk files,
and prints final statistics.

Requires: python scripts/prepare_cartpole.py  (to generate the data first)

Usage:
    python scripts/train_cartpole.py
    python scripts/train_cartpole.py --data data/cartpole_train.h5 --epochs 200
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def train_variant(
    name: str,
    config: str,
    data_path: str,
    model_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> tuple:
    """Train a single model variant and return (model, elapsed, save_path)."""
    from worldkit import WorldModel

    print(f"\n{'─' * 60}")
    print(f"Training CartPole [{config}] for {epochs} epochs")
    print(f"{'─' * 60}")

    start = time.time()

    model = WorldModel.train(
        data=data_path,
        config=config,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lambda_reg=0.5,
        action_dim=1,
        device=device,
        checkpoint_dir=str(model_dir / f"cartpole_{config}"),
        seed=42,
    )

    elapsed = time.time() - start

    save_path = model_dir / f"cartpole_{config}.wk"
    model.save(save_path)

    return model, elapsed, save_path


def print_model_stats(label: str, model, elapsed: float, save_path: Path):
    """Print summary statistics for a trained model."""
    size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"\n  [{label}]")
    print(f"    Parameters:  {model.num_params:,}")
    print(f"    Latent dim:  {model.latent_dim}")
    print(f"    Model size:  {size_mb:.1f} MB")
    print(f"    Train time:  {elapsed / 60:.1f} min")


def verify_model(model, device: str):
    """Run basic inference checks on the trained model."""
    # Encode
    test_obs = np.random.rand(96, 96, 3).astype(np.float32)
    z = model.encode(test_obs)
    print(f"    encode():    (96,96,3) -> latent {tuple(z.shape)}")

    # Predict
    test_actions = [np.array([1.0], dtype=np.float32)] * 5
    result = model.predict(test_obs, test_actions)
    print(f"    predict():   {result.steps} steps, confidence={result.confidence:.3f}")

    return z


def main():
    parser = argparse.ArgumentParser(
        description="Train WorldKit models on CartPole pixel data."
    )
    parser.add_argument(
        "--data",
        default="data/cartpole_train.h5",
        help="Path to CartPole HDF5 data (relative to project root)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Training epochs per variant"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_path = project_root / args.data
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Check data exists
    if not data_path.exists():
        print(f"ERROR: Data not found at {data_path}")
        print("Run first:  python scripts/prepare_cartpole.py")
        sys.exit(1)

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("=" * 60)
    print("WorldKit | CartPole Training Pipeline")
    print("=" * 60)
    print(f"  Data:    {data_path}")
    print(f"  Device:  {device}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Batch:   {args.batch_size}")
    print(f"  LR:      {args.lr}")

    # ── Train base variant ──────────────────────────────────
    base_model, base_time, base_path = train_variant(
        name="base",
        config="base",
        data_path=str(data_path),
        model_dir=model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    # ── Train nano variant ──────────────────────────────────
    nano_model, nano_time, nano_path = train_variant(
        name="nano",
        config="nano",
        data_path=str(data_path),
        model_dir=model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    # ── Final stats ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training Complete — Final Statistics")
    print("=" * 60)

    print_model_stats("base", base_model, base_time, base_path)
    print_model_stats("nano", nano_model, nano_time, nano_path)

    # ── Verification ────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Verification")
    print(f"{'─' * 60}")

    print("\n  Base model:")
    z_base = verify_model(base_model, device)

    print("\n  Nano model:")
    z_nano = verify_model(nano_model, device)

    # Save/load roundtrip check
    from worldkit import WorldModel

    loaded = WorldModel.load(base_path, device=device)
    test_obs = np.random.rand(96, 96, 3).astype(np.float32)
    z1 = base_model.encode(test_obs)
    z2 = loaded.encode(test_obs)
    match = torch.allclose(z1, z2, atol=1e-5)
    print(f"\n  Save/load roundtrip (base): match={match}")

    print(f"\n{'=' * 60}")
    print("SUCCESS! Both CartPole models trained and verified.")
    print(f"  Base: {base_path}")
    print(f"  Nano: {nano_path}")
    total_time = base_time + nano_time
    print(f"  Total training time: {total_time / 60:.1f} min")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
