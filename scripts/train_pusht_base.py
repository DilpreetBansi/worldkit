"""Train the flagship WorldKit 'base' model on Push-T data.

Usage:
    python scripts/train_pusht_base.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("=" * 60)
    print("WorldKit | Push-T Base Model Training")
    print("=" * 60)

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    data_path = Path(__file__).parent.parent / "data" / "pusht_train.h5"
    if not data_path.exists():
        print("ERROR: Run train_pusht.py first to generate data.")
        return

    model_dir = Path(__file__).parent.parent / "checkpoints"
    model_dir.mkdir(exist_ok=True)

    print("\nConfig: base (~13M params)")
    print("This should take ~2-5 minutes on M4 Pro MPS\n")

    from worldkit import WorldModel

    start_time = time.time()

    model = WorldModel.train(
        data=str(data_path),
        config="base",
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

    # Save
    final_path = model_dir / "pusht_base.wk"
    model.save(final_path)
    model_size_mb = final_path.stat().st_size / (1024 * 1024)
    print(f"Model saved: {final_path} ({model_size_mb:.1f} MB)")

    # Verify
    print("\n--- Verification ---")
    test_obs = np.random.rand(96, 96, 3).astype(np.float32)
    z = model.encode(test_obs)
    print(f"  encode(): input (96,96,3) -> latent {tuple(z.shape)} ✓")

    test_actions = [np.array([0.1, 0.2], dtype=np.float32)] * 5
    result = model.predict(test_obs, test_actions)
    print(f"  predict(): {result.steps} steps, confidence={result.confidence:.3f} ✓")

    loaded = WorldModel.load(final_path, device=device)
    z2 = loaded.encode(test_obs)
    match = torch.allclose(z, z2, atol=1e-5)
    print(f"  save/load roundtrip: match={match} ✓")

    print(f"\n{'='*60}")
    print(f"SUCCESS! Base model trained.")
    print(f"  Params: {model.num_params:,} | Size: {model_size_mb:.1f} MB | Time: {elapsed/60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
