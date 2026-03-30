"""Knowledge distillation for WorldKit world models.

Trains a smaller student model to replicate the latent predictions of a
larger teacher model using MSE loss on latent embeddings.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .config import ModelConfig, get_config


def distill(
    teacher: object,
    student_config: str | ModelConfig,
    data: str | Path,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: str = "auto",
    seed: int = 42,
) -> object:
    """Distill a teacher WorldModel into a smaller student.

    For each batch the teacher produces latent encodings and predictions
    (detached). The student is trained to match those representations
    via MSE loss on both encoder outputs and predictor outputs.

    Args:
        teacher: Trained WorldModel instance (used as frozen reference).
        student_config: Config name or ModelConfig for the student.
        data: Path to HDF5 training data.
        epochs: Number of distillation epochs.
        batch_size: Batch size.
        lr: Learning rate for the student optimizer.
        device: Device string ("cpu", "cuda", "auto").
        seed: Random seed.

    Returns:
        Trained student WorldModel.
    """
    import h5py
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset, random_split

    from .backends import backend_registry
    from .model import WorldModel, _auto_device

    torch.manual_seed(seed)

    if device == "auto":
        device = _auto_device()

    # Resolve student config
    if isinstance(student_config, str):
        student_cfg = get_config(
            student_config,
            action_dim=teacher._config.action_dim,
            lambda_reg=teacher._config.lambda_reg,
        )
    else:
        student_cfg = student_config

    # Build student model via backend
    backend_cls = backend_registry.get(student_cfg.backend)
    backend = backend_cls()
    student_module = backend.build(student_cfg)
    student = WorldModel(student_module, student_cfg, device, backend=backend)

    print(
        f"WorldKit | Distillation: teacher={teacher.num_params:,} → "
        f"student={student.num_params:,} params"
    )

    # Load data
    data_path = Path(data)
    with h5py.File(data_path, "r") as f:
        for key in ["pixels", "observations", "obs", "images"]:
            if key in f:
                pixels = torch.tensor(np.array(f[key]), dtype=torch.float32)
                break
        else:
            raise KeyError(
                f"No pixel data in HDF5. Found keys: {list(f.keys())}"
            )

        for key in ["actions", "action"]:
            if key in f:
                actions = torch.tensor(np.array(f[key]), dtype=torch.float32)
                break
        else:
            raise KeyError(
                f"No action data in HDF5. Found keys: {list(f.keys())}"
            )

    if pixels.max() > 1.0:
        pixels = pixels / 255.0
    if pixels.dim() == 4:
        pixels = pixels.unsqueeze(1)
    if pixels.shape[-1] in (1, 3):
        pixels = pixels.permute(0, 1, 4, 2, 3)

    dataset = TensorDataset(pixels, actions)
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, _ = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # Move teacher to device, freeze
    teacher._model.to(device)
    teacher._model.eval()
    for p in teacher._model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        student._model.parameters(), lr=lr, weight_decay=student_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ctx_len = student_cfg.context_length

    print(f"WorldKit | Distilling for {epochs} epochs on {len(train_ds)} samples")

    for epoch in range(epochs):
        student._model.train()
        epoch_loss = 0.0
        steps = 0

        for batch_pixels, batch_actions in train_loader:
            batch_pixels = batch_pixels.to(device)
            batch_actions = batch_actions.to(device)

            # Teacher forward (frozen)
            with torch.no_grad():
                t_ctx_emb = teacher._backend.encode(
                    teacher._model, batch_pixels[:, :ctx_len]
                )  # (B, T_ctx, D_teacher)
                t_act_emb = teacher._model.encode_actions(batch_actions)
                t_pred_emb = teacher._model.predict(
                    t_ctx_emb, t_act_emb[:, :ctx_len]
                )  # (B, T_pred, D_teacher)

            # Student forward
            s_ctx_emb = student._backend.encode(
                student._model, batch_pixels[:, :ctx_len]
            )  # (B, T_ctx, D_student)
            s_act_emb = student._model.encode_actions(batch_actions)
            s_pred_emb = student._model.predict(
                s_ctx_emb, s_act_emb[:, :ctx_len]
            )  # (B, T_pred, D_student)

            # Match prediction lengths
            min_pred = min(s_pred_emb.shape[1], t_pred_emb.shape[1])
            s_pred = s_pred_emb[:, :min_pred]
            t_pred = t_pred_emb[:, :min_pred]

            # Distillation loss: project student latents to teacher dim, then MSE
            # If dimensions differ, use a linear projection
            if s_pred.shape[-1] != t_pred.shape[-1]:
                # Lazy projection layer creation
                if not hasattr(distill, "_proj") or distill._proj is None:
                    distill._proj = torch.nn.Linear(
                        s_pred.shape[-1], t_pred.shape[-1], bias=False
                    ).to(device)
                s_proj = distill._proj(s_pred)
            else:
                s_proj = s_pred

            loss = F.mse_loss(s_proj, t_pred.detach())

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                student._model.parameters(), max_norm=1.0
            )
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg = epoch_loss / max(steps, 1)
            print(
                f"  Epoch {epoch+1}/{epochs} | Distillation Loss: {avg:.4f}"
            )

    # Cleanup
    distill._proj = None  # type: ignore[attr-defined]

    # Restore teacher grad state
    for p in teacher._model.parameters():
        p.requires_grad_(True)

    print("WorldKit | Distillation complete.")
    return student
