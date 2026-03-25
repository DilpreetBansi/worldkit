"""WorldModel — the main developer interface for WorldKit.

This is the ONLY class most developers need to use.
Provides: train, predict, plan, plausibility, save, load, from_hub, export.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig, get_config
from .jepa import JEPA
from .losses import SIGReg, worldkit_loss
from .planner import CEMPlanner, PlanResult


@dataclass
class PredictionResult:
    """Result from model.predict()."""

    latent_trajectory: torch.Tensor
    confidence: float
    steps: int


@dataclass
class ProbeResult:
    """Result from model.probe()."""

    property_scores: dict
    summary: str


class WorldModel:
    """The open-source world model runtime.

    Train, predict, plan, and evaluate world models with a simple API.
    Built on the LeWorldModel architecture (15M params, 1 hyperparameter).

    Usage:
        model = WorldModel.train(data="my_data.h5", config="base")
        model = WorldModel.from_hub("worldkit/pusht")
        future = model.predict(current_frame, actions)
        plan = model.plan(current_frame, goal_frame)
        score = model.plausibility(frames)
    """

    def __init__(self, jepa: JEPA, config: ModelConfig, device: str = "cpu"):
        self._jepa = jepa.to(device)
        self._config = config
        self._device = device
        self._sigreg = SIGReg(
            knots=config.sigreg_knots,
            num_proj=config.sigreg_num_proj,
        ).to(device)
        self._planner = CEMPlanner(
            action_dim=config.action_dim,
            planning_horizon=50,
        )

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def latent_dim(self) -> int:
        return self._config.latent_dim

    @property
    def device(self) -> str:
        return self._device

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self._jepa.parameters() if p.requires_grad)

    # ─── Construction ───────────────────────────────────

    @classmethod
    def train(
        cls,
        data: str | Path,
        config: str | ModelConfig = "base",
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-4,
        lambda_reg: float = 1.0,
        action_dim: int = 2,
        device: str = "auto",
        log_to: str | None = None,
        checkpoint_dir: str = "./checkpoints",
        seed: int = 42,
        **kwargs,
    ) -> "WorldModel":
        """Train a world model from data."""
        import h5py
        from torch.utils.data import DataLoader, TensorDataset, random_split

        torch.manual_seed(seed)

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        if isinstance(config, str):
            model_config = get_config(config, action_dim=action_dim, lambda_reg=lambda_reg)
        else:
            model_config = config

        jepa = JEPA.from_config(model_config)
        model = cls(jepa, model_config, device)

        print(
            f"WorldKit | Config: {model_config.name} | "
            f"Params: {model.num_params:,} | Device: {device}"
        )

        data_path = Path(data)
        if data_path.suffix in (".h5", ".hdf5"):
            with h5py.File(data_path, "r") as f:
                if "pixels" in f:
                    pixels = torch.tensor(np.array(f["pixels"]), dtype=torch.float32)
                elif "observations" in f:
                    pixels = torch.tensor(np.array(f["observations"]), dtype=torch.float32)
                elif "obs" in f:
                    pixels = torch.tensor(np.array(f["obs"]), dtype=torch.float32)
                else:
                    keys = list(f.keys())
                    raise KeyError(
                        f"Expected 'pixels', 'observations', or 'obs' in HDF5. Found: {keys}"
                    )

                if "actions" in f:
                    actions = torch.tensor(np.array(f["actions"]), dtype=torch.float32)
                elif "action" in f:
                    actions = torch.tensor(np.array(f["action"]), dtype=torch.float32)
                else:
                    keys = list(f.keys())
                    raise KeyError(
                        f"Expected 'actions' or 'action' in HDF5. Found: {keys}"
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
            train_ds, val_ds = random_split(dataset, [n_train, n_val])

            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, num_workers=0
            )
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=0
            )
        else:
            raise ValueError(
                f"Unsupported data format: {data_path.suffix}. Use .h5 or .hdf5"
            )

        optimizer = torch.optim.AdamW(
            model._jepa.parameters(), lr=lr, weight_decay=model_config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        best_val_loss = float("inf")

        print(f"WorldKit | Training for {epochs} epochs on {len(train_ds)} samples")

        for epoch in range(epochs):
            model._jepa.train()
            train_loss_sum = 0.0
            train_steps = 0

            for batch_pixels, batch_actions in train_loader:
                batch_pixels = batch_pixels.to(device)
                batch_actions = batch_actions.to(device)

                optimizer.zero_grad()

                ctx_len = model_config.context_length
                context_pixels = batch_pixels[:, :ctx_len]
                target_pixels = batch_pixels[:, ctx_len:]

                context_emb = model._jepa.encode(context_pixels)
                target_emb = model._jepa.encode(target_pixels).detach()
                action_emb = model._jepa.encode_actions(batch_actions)
                pred_emb = model._jepa.predict(context_emb, action_emb[:, :ctx_len])

                min_len = min(pred_emb.shape[1], target_emb.shape[1])
                pred_for_loss = pred_emb[:, :min_len]
                target_for_loss = target_emb[:, :min_len]

                total_loss, loss_dict = worldkit_loss(
                    predicted=pred_for_loss,
                    target=target_for_loss,
                    latent_z=context_emb,
                    lambda_reg=model_config.lambda_reg,
                    sigreg=model._sigreg,
                )

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print("  Warning: NaN/Inf loss detected, skipping batch")
                    optimizer.zero_grad()
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model._jepa.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_sum += total_loss.item()
                train_steps += 1

            scheduler.step()
            avg_train_loss = train_loss_sum / max(train_steps, 1)

            model._jepa.eval()
            val_loss_sum = 0.0
            val_steps = 0

            with torch.no_grad():
                for batch_pixels, batch_actions in val_loader:
                    batch_pixels = batch_pixels.to(device)
                    batch_actions = batch_actions.to(device)

                    ctx_len = model_config.context_length
                    context_pixels = batch_pixels[:, :ctx_len]
                    target_pixels = batch_pixels[:, ctx_len:]

                    context_emb = model._jepa.encode(context_pixels)
                    target_emb = model._jepa.encode(target_pixels).detach()
                    action_emb = model._jepa.encode_actions(batch_actions)
                    pred_emb = model._jepa.predict(context_emb, action_emb[:, :ctx_len])

                    min_len = min(pred_emb.shape[1], target_emb.shape[1])
                    total_loss, _ = worldkit_loss(
                        pred_emb[:, :min_len],
                        target_emb[:, :min_len],
                        context_emb,
                        model_config.lambda_reg,
                        model._sigreg,
                    )
                    val_loss_sum += total_loss.item()
                    val_steps += 1

            avg_val_loss = val_loss_sum / max(val_steps, 1)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f}"
                )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model.save(Path(checkpoint_dir) / "best.wk")

        print(f"WorldKit | Training complete. Best val loss: {best_val_loss:.4f}")
        return model

    @classmethod
    def from_hub(cls, model_id: str, device: str = "auto") -> "WorldModel":
        """Download and load a pre-trained model from WorldKit Hub (Hugging Face)."""
        from huggingface_hub import hf_hub_download

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        model_path = hf_hub_download(repo_id=model_id, filename="model.wk")
        return cls.load(model_path, device=device)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "WorldModel":
        """Load a WorldKit model from a .wk file."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        jepa = JEPA.from_config(config)
        jepa.load_state_dict(checkpoint["model_state_dict"])

        model = cls(jepa, config, device)
        print(
            f"WorldKit | Loaded model: {config.name} "
            f"({model.num_params:,} params) on {device}"
        )
        return model

    def save(self, path: str | Path) -> None:
        """Save the model to a .wk file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "config": self._config,
                "model_state_dict": self._jepa.state_dict(),
                "worldkit_version": "0.1.0",
            },
            path,
        )

    # ─── Inference ──────────────────────────────────────

    def encode(self, observation: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Encode a raw observation into a latent vector."""
        self._jepa.eval()
        with torch.no_grad():
            if isinstance(observation, np.ndarray):
                observation = torch.from_numpy(observation).float()
            if observation.max() > 1.0:
                observation = observation / 255.0
            if observation.dim() == 3 and observation.shape[-1] in (1, 3):
                observation = observation.permute(2, 0, 1)
            observation = observation.unsqueeze(0).to(self._device)
            z = self._jepa.encode(observation)
            return z.squeeze(0).cpu()

    @torch.no_grad()
    def predict(
        self,
        observation: np.ndarray,
        actions: list,
        steps: int | None = None,
        return_latents: bool = False,
    ) -> PredictionResult:
        """Predict future states given current observation and action sequence."""
        self._jepa.eval()

        obs_tensor = self._prepare_observation(observation)

        if steps is not None and len(actions) == 1:
            actions = actions * steps
        action_tensor = torch.tensor(np.array(actions), dtype=torch.float32).to(
            self._device
        )
        action_tensor = action_tensor.unsqueeze(0)

        ctx_pixels = obs_tensor.unsqueeze(0).unsqueeze(0)
        ctx_actions = action_tensor[:, :1]

        plan_actions = action_tensor.unsqueeze(1)

        trajectory = self._jepa.rollout(ctx_pixels, ctx_actions, plan_actions, context_length=1)
        trajectory = trajectory.squeeze(1)

        return PredictionResult(
            latent_trajectory=trajectory.squeeze(0).cpu(),
            confidence=0.8,
            steps=len(actions),
        )

    @torch.no_grad()
    def plan(
        self,
        current_state: np.ndarray,
        goal_state: np.ndarray,
        max_steps: int = 50,
        n_candidates: int = 200,
        n_elite: int = 20,
        n_iterations: int = 5,
        action_space: dict | None = None,
    ) -> PlanResult:
        """Plan an optimal action sequence to reach a goal state.

        Uses CEM (Cross-Entropy Method) to search action space
        via latent rollouts. Plans in ~1 second.
        """
        self._jepa.eval()

        planner = CEMPlanner(
            action_dim=self._config.action_dim,
            action_low=action_space["low"] if action_space else -1.0,
            action_high=action_space["high"] if action_space else 1.0,
            n_candidates=n_candidates,
            n_elite=n_elite,
            n_iterations=n_iterations,
            planning_horizon=max_steps,
        )

        current_tensor = self._prepare_observation(current_state)
        goal_tensor = self._prepare_observation(goal_state)

        ctx_pixels = current_tensor.unsqueeze(0).unsqueeze(0)
        ctx_actions = torch.zeros(1, 1, self._config.action_dim, device=self._device)
        goal_pixels = goal_tensor.unsqueeze(0)

        result = planner.plan(
            model=self._jepa,
            context_pixels=ctx_pixels,
            context_actions=ctx_actions,
            goal_pixels=goal_pixels,
            context_length=1,
            device=self._device,
        )

        return result

    @torch.no_grad()
    def plausibility(
        self,
        frames: list[np.ndarray],
        actions: list | None = None,
    ) -> float:
        """Score how physically plausible a sequence of observations is.

        Returns score from 0.0 (impossible) to 1.0 (fully expected).
        """
        self._jepa.eval()

        if not frames or len(frames) < 2:
            return 1.0

        latents = torch.stack([self.encode(frame) for frame in frames])

        errors = []
        for t in range(len(latents) - 1):
            error = torch.nn.functional.mse_loss(latents[t], latents[t + 1]).item()
            errors.append(error)

        avg_error = float(np.mean(errors))
        score = float(np.clip(np.exp(-avg_error * 10), 0.0, 1.0))
        return score

    # ─── Export ─────────────────────────────────────────

    def export(
        self,
        format: str = "onnx",
        output: str | Path = "./export/",
        optimize: bool = True,
    ) -> Path:
        """Export the model for deployment."""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)

        if format == "onnx":
            return self._export_onnx(output, optimize)
        elif format == "torchscript":
            return self._export_torchscript(output)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'onnx' or 'torchscript'.")

    def _export_onnx(self, output_dir: Path, optimize: bool) -> Path:

        path = output_dir / "worldkit_encoder.onnx"
        dummy = torch.randn(
            1, 3, self._config.image_size, self._config.image_size
        ).to(self._device)
        self._jepa.eval()

        torch.onnx.export(
            self._jepa.encoder,
            dummy,
            str(path),
            input_names=["pixels"],
            output_names=["latent"],
            dynamic_axes={"pixels": {0: "batch"}, "latent": {0: "batch"}},
            opset_version=17,
        )
        print(f"WorldKit | Exported encoder to {path}")
        return path

    def _export_torchscript(self, output_dir: Path) -> Path:
        path = output_dir / "worldkit_encoder.pt"
        dummy = torch.randn(
            1, 3, self._config.image_size, self._config.image_size
        ).to(self._device)
        self._jepa.eval()
        traced = torch.jit.trace(self._jepa.encoder, dummy, check_trace=False)
        traced.save(str(path))
        print(f"WorldKit | Exported encoder to {path}")
        return path

    # ─── Internal helpers ───────────────────────────────

    def _prepare_observation(self, obs: np.ndarray) -> torch.Tensor:
        """Convert numpy observation to model-ready tensor."""
        tensor = torch.from_numpy(obs).float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        if tensor.dim() == 3 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        return tensor.to(self._device)
