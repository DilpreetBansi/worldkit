"""WorldModel — the main developer interface for WorldKit.

This is the ONLY class most developers need to use.
Provides: train, predict, plan, plausibility, save, load, from_hub, export.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import ModelConfig, get_config
from .format import WKFormat
from .hierarchical_planner import HierarchicalPlanner, HierarchicalPlanResult
from .planner import CEMPlanner, PlanResult


@dataclass
class PredictionResult:
    """Result from model.predict()."""

    latent_trajectory: torch.Tensor
    confidence: float
    steps: int


@dataclass
class ProbeResult:
    """Result from model.probe().

    Attributes:
        property_scores: R² score per probed property.
        mse_scores: Mean squared error per probed property.
        probes: Trained sklearn Ridge models, keyed by property name.
        summary: Human-readable summary of all probe results.
    """

    property_scores: dict
    mse_scores: dict
    probes: dict
    summary: str


def _auto_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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

    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        device: str = "cpu",
        *,
        backend: object | None = None,
    ):
        from .backends import BaseWorldModelBackend, backend_registry

        if backend is None:
            backend_name = getattr(config, "backend", "lewm")
            backend_cls = backend_registry.get(backend_name)
            backend = backend_cls()

        self._backend: BaseWorldModelBackend = backend
        self._model = model.to(device)
        self._config = config
        self._device = device
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
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    # ─── Construction ───────────────────────────────────

    @classmethod
    def train(
        cls,
        data: str | Path | list[str | Path],
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
        """Train a world model from data.

        Args:
            data: Path to an HDF5 file, or a list of paths for
                multi-environment training.  When multiple paths are given,
                action dimensions are zero-padded to the maximum across
                datasets and batches are interleaved proportionally.
        """
        import h5py
        from torch.utils.data import DataLoader, TensorDataset, random_split

        from .backends import backend_registry

        torch.manual_seed(seed)

        if device == "auto":
            device = _auto_device()

        # ── Multi-environment path ──────────────────────────
        if isinstance(data, list):
            from worldkit.data.multi_dataset import MultiEnvironmentDataset

            multi_ds = MultiEnvironmentDataset(
                [Path(p) for p in data],
                sequence_length=16,
            )
            # Override action_dim to the max across datasets
            action_dim = multi_ds.max_action_dim

            if isinstance(config, str):
                model_config = get_config(
                    config, action_dim=action_dim, lambda_reg=lambda_reg,
                )
            else:
                model_config = config

            backend_cls = backend_registry.get(model_config.backend)
            backend = backend_cls()
            model_module = backend.build(model_config)
            model = cls(model_module, model_config, device, backend=backend)

            print(
                f"WorldKit | Multi-env training: {len(data)} datasets | "
                f"Config: {model_config.name} | "
                f"Params: {model.num_params:,} | Device: {device}"
            )

            n_val = max(1, int(len(multi_ds) * 0.1))
            n_train = len(multi_ds) - n_val
            train_ds, val_ds = random_split(multi_ds, [n_train, n_val])

            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
            )
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
            )
        else:
            # ── Single-file path ────────────────────────────
            if isinstance(config, str):
                model_config = get_config(
                    config, action_dim=action_dim, lambda_reg=lambda_reg,
                )
            else:
                model_config = config

            backend_cls = backend_registry.get(model_config.backend)
            backend = backend_cls()
            model_module = backend.build(model_config)
            model = cls(model_module, model_config, device, backend=backend)

            print(
                f"WorldKit | Config: {model_config.name} | "
                f"Params: {model.num_params:,} | Device: {device}"
            )

            data_path = Path(data)
            if data_path.suffix in (".h5", ".hdf5"):
                with h5py.File(data_path, "r") as f:
                    if "pixels" in f:
                        pixels = torch.tensor(
                            np.array(f["pixels"]), dtype=torch.float32,
                        )
                    elif "observations" in f:
                        pixels = torch.tensor(
                            np.array(f["observations"]), dtype=torch.float32,
                        )
                    elif "obs" in f:
                        pixels = torch.tensor(
                            np.array(f["obs"]), dtype=torch.float32,
                        )
                    else:
                        keys = list(f.keys())
                        raise KeyError(
                            f"Expected 'pixels', 'observations', or 'obs' "
                            f"in HDF5. Found: {keys}"
                        )

                    if "actions" in f:
                        actions = torch.tensor(
                            np.array(f["actions"]), dtype=torch.float32,
                        )
                    elif "action" in f:
                        actions = torch.tensor(
                            np.array(f["action"]), dtype=torch.float32,
                        )
                    else:
                        keys = list(f.keys())
                        raise KeyError(
                            f"Expected 'actions' or 'action' in HDF5. "
                            f"Found: {keys}"
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
                    train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
                )
                val_loader = DataLoader(
                    val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
                )
            else:
                raise ValueError(
                    f"Unsupported data format: {data_path.suffix}. "
                    f"Use .h5 or .hdf5"
                )

        optimizer = torch.optim.AdamW(
            model._model.parameters(), lr=lr, weight_decay=model_config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        best_val_loss = float("inf")

        print(f"WorldKit | Training for {epochs} epochs on {len(train_ds)} samples")

        for epoch in range(epochs):
            model._model.train()
            train_loss_sum = 0.0
            train_steps = 0

            for batch_pixels, batch_actions in train_loader:
                batch_pixels = batch_pixels.to(device)
                batch_actions = batch_actions.to(device)

                optimizer.zero_grad()

                total_loss, loss_dict = model._backend.training_step(
                    model._model, (batch_pixels, batch_actions), model_config
                )

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print("  Warning: NaN/Inf loss detected, skipping batch")
                    optimizer.zero_grad()
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model._model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_sum += total_loss.item()
                train_steps += 1

            scheduler.step()
            avg_train_loss = train_loss_sum / max(train_steps, 1)

            model._model.eval()
            val_loss_sum = 0.0
            val_steps = 0

            with torch.no_grad():
                for batch_pixels, batch_actions in val_loader:
                    batch_pixels = batch_pixels.to(device)
                    batch_actions = batch_actions.to(device)

                    total_loss, _ = model._backend.training_step(
                        model._model, (batch_pixels, batch_actions), model_config
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
            device = _auto_device()

        model_path = hf_hub_download(repo_id=model_id, filename="model.wk")
        return cls.load(model_path, device=device)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "WorldModel":
        """Load a WorldKit model from a .wk file.

        Supports both the new ZIP-based format (v2) and the legacy torch.save
        format (v1). Legacy files trigger a deprecation warning.
        """
        from .backends import backend_registry

        if device == "auto":
            device = _auto_device()

        path = Path(path)

        if WKFormat.is_new_format(path):
            data = WKFormat.load(path)
            config = data["config"]
            state_dict = data["model_state_dict"]
        else:
            warnings.warn(
                "Loading a legacy .wk file (torch.save format). "
                "Re-save with model.save() to upgrade to the new ZIP-based format.",
                DeprecationWarning,
                stacklevel=2,
            )
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            config = checkpoint["config"]
            state_dict = checkpoint["model_state_dict"]
            # Backward compat: old checkpoints may store backend at top level
            backend_name = checkpoint.get("backend", "lewm")
            if not hasattr(config, "backend"):
                object.__setattr__(config, "backend", backend_name)

        # Resolve backend, build model, and load weights
        backend_name = getattr(config, "backend", "lewm")
        backend_cls = backend_registry.get(backend_name)
        backend = backend_cls()
        model_module = backend.build(config)
        model_module.load_state_dict(state_dict)

        model = cls(model_module, config, device, backend=backend)
        print(
            f"WorldKit | Loaded model: {config.name} "
            f"({model.num_params:,} params) on {device}"
        )
        return model

    def save(
        self,
        path: str | Path,
        metadata: dict | None = None,
        action_space: dict | None = None,
        model_card: dict | None = None,
    ) -> None:
        """Save the model to a .wk ZIP archive.

        Args:
            path: Destination file path.
            metadata: Optional training metadata (dataset, epochs, loss, etc.).
            action_space: Optional action space definition overrides.
            model_card: Optional model card overrides.
        """
        WKFormat.save(
            path=path,
            model_state_dict=self._model.state_dict(),
            config=self._config,
            metadata=metadata,
            action_space=action_space,
            model_card=model_card,
        )

    # ─── Inference ──────────────────────────────────────

    def encode(self, observation: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Encode a raw observation into a latent vector."""
        was_training = self._model.training
        self._model.eval()
        try:
            with torch.no_grad():
                if isinstance(observation, np.ndarray):
                    observation = torch.from_numpy(observation).float()
                else:
                    observation = observation.float()
                if observation.max() > 1.0:
                    observation = observation / 255.0
                if observation.dim() == 3 and observation.shape[-1] in (1, 3):
                    observation = observation.permute(2, 0, 1)
                observation = observation.unsqueeze(0).to(self._device)
                z = self._backend.encode(self._model, observation)
                return z.squeeze(0).cpu()
        finally:
            if was_training:
                self._model.train()

    @torch.no_grad()
    def predict(
        self,
        observation: np.ndarray | torch.Tensor,
        actions: list,
        steps: int | None = None,
        return_latents: bool = False,
    ) -> PredictionResult:
        """Predict future states given current observation and action sequence."""
        was_training = self._model.training
        self._model.eval()

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

        trajectory = self._backend.rollout(
            self._model, ctx_pixels, ctx_actions, plan_actions, context_length=1
        )
        trajectory = trajectory.squeeze(1)

        result = PredictionResult(
            latent_trajectory=trajectory.squeeze(0).cpu(),
            confidence=0.8,
            steps=len(actions),
        )
        if was_training:
            self._model.train()
        return result

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
        was_training = self._model.training
        self._model.eval()

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
            model=self._model,
            context_pixels=ctx_pixels,
            context_actions=ctx_actions,
            goal_pixels=goal_pixels,
            context_length=1,
            device=self._device,
        )

        if was_training:
            self._model.train()
        return result

    @torch.no_grad()
    def hierarchical_plan(
        self,
        current_state: np.ndarray,
        goal_state: np.ndarray,
        max_subgoals: int = 5,
        steps_per_subgoal: int = 50,
        n_candidates: int = 200,
        n_elite: int = 20,
        n_iterations: int = 5,
        action_space: dict | None = None,
    ) -> HierarchicalPlanResult:
        """Plan a long-horizon action sequence via hierarchical subgoal decomposition.

        Encodes current and goal observations, interpolates subgoal latents,
        then uses CEM planning to connect consecutive subgoals.

        Args:
            current_state: Current observation as numpy array (H, W, C).
            goal_state: Goal observation as numpy array (H, W, C).
            max_subgoals: Number of intermediate subgoals to create.
            steps_per_subgoal: Planning horizon for each segment.
            n_candidates: CEM candidate count per segment.
            n_elite: CEM elite count per segment.
            n_iterations: CEM iterations per segment.
            action_space: Optional dict with ``"low"`` and ``"high"`` keys.

        Returns:
            HierarchicalPlanResult with full action sequence and subgoal info.
        """
        was_training = self._model.training
        self._model.eval()

        planner = HierarchicalPlanner(
            action_dim=self._config.action_dim,
            action_low=action_space["low"] if action_space else -1.0,
            action_high=action_space["high"] if action_space else 1.0,
            n_candidates=n_candidates,
            n_elite=n_elite,
            n_iterations=n_iterations,
        )

        current_tensor = self._prepare_observation(current_state)
        goal_tensor = self._prepare_observation(goal_state)

        # Encode both observations to latent space
        current_latent = self._backend.encode(
            self._model, current_tensor.unsqueeze(0)
        ).squeeze(0)  # (D,)
        goal_latent = self._backend.encode(
            self._model, goal_tensor.unsqueeze(0)
        ).squeeze(0)  # (D,)

        ctx_pixels = current_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
        ctx_actions = torch.zeros(
            1, 1, self._config.action_dim, device=self._device
        )

        result = planner.plan(
            model=self._model,
            context_pixels=ctx_pixels,
            context_actions=ctx_actions,
            current_latent=current_latent,
            goal_latent=goal_latent,
            max_subgoals=max_subgoals,
            steps_per_subgoal=steps_per_subgoal,
            context_length=1,
            device=self._device,
        )

        if was_training:
            self._model.train()
        return result

    @classmethod
    def auto_config(
        cls,
        data: str | Path,
        max_training_time: str = "2h",
        target_device: str | None = None,
        trial_epochs: int = 5,
        device: str = "cpu",
    ) -> tuple[ModelConfig, str]:
        """Recommend a model configuration based on data and constraints.

        Samples the data, runs quick training trials, and picks the best
        config that fits within time and device constraints.

        Args:
            data: Path to HDF5 training data.
            max_training_time: Time budget (e.g. ``"2h"``, ``"30m"``).
            target_device: Deployment target (``"jetson"``, ``"browser"``, etc.).
            trial_epochs: Epochs per trial run.
            device: Device for running trials.

        Returns:
            Tuple of (recommended ModelConfig, explanation string).
        """
        from .auto_config import auto_config as _auto_config

        return _auto_config(
            data=data,
            max_training_time=max_training_time,
            target_device=target_device,
            trial_epochs=trial_epochs,
            device=device,
        )

    @torch.no_grad()
    def plausibility(
        self,
        frames: list[np.ndarray],
        actions: list | None = None,
    ) -> float:
        """Score how physically plausible a sequence of observations is.

        Returns score from 0.0 (impossible) to 1.0 (fully expected).
        """
        if not frames or len(frames) < 2:
            return 1.0

        was_training = self._model.training
        self._model.eval()

        try:
            latents = torch.stack([self.encode(frame) for frame in frames])

            # Compute consecutive-frame prediction errors
            diffs = latents[1:] - latents[:-1]
            errors = (diffs ** 2).mean(dim=-1)
            avg_error = errors.mean().item()

            score = float(np.clip(np.exp(-avg_error * 10), 0.0, 1.0))
            return score
        finally:
            if was_training:
                self._model.train()

    # ─── Evaluation ─────────────────────────────────────

    def probe(
        self,
        data: str | Path,
        properties: list[str],
        labels: str | Path,
        alpha: float = 1.0,
        test_fraction: float = 0.2,
        seed: int = 42,
    ) -> ProbeResult:
        """Train linear probes to measure what the latent space has learned.

        Encodes observations through the frozen encoder, then fits Ridge
        regression from latent vectors to each target property. Reports
        R² and MSE per property.

        Args:
            data: Path to HDF5 file with pixel observations.
            properties: Property names to probe (must exist in labels file).
            labels: Path to CSV or HDF5 file with per-frame property values.
            alpha: Ridge regularization strength.
            test_fraction: Fraction of data held out for evaluation.
            seed: Random seed for train/test split.

        Returns:
            ProbeResult with R² scores, MSE, trained probes, and summary.
        """
        from worldkit.eval.probing import LinearProbe

        prober = LinearProbe(self)
        return prober.fit(
            data_path=data,
            properties=properties,
            labels_path=labels,
            alpha=alpha,
            test_fraction=test_fraction,
            seed=seed,
        )

    def visualize_latent_space(
        self,
        data: str | Path,
        method: str = "pca",
        color_by: str | None = None,
        save_to: str | Path | None = None,
    ):
        """Plot a dimensionality-reduced view of the latent space.

        Args:
            data: Path to HDF5 file with pixel observations.
            method: Reduction method — ``"pca"``, ``"tsne"``, or ``"umap"``.
            color_by: Color points by ``"episode"`` or ``"timestep"``.
            save_to: If provided, save the figure to this path.

        Returns:
            matplotlib Figure.
        """
        from worldkit.eval.visualize import LatentVisualizer

        viz = LatentVisualizer(self)
        plot_fn = {
            "pca": viz.plot_pca,
            "tsne": viz.plot_tsne,
            "umap": viz.plot_umap,
        }
        if method not in plot_fn:
            raise ValueError(
                f"Unknown method '{method}'. Use 'pca', 'tsne', or 'umap'."
            )
        return plot_fn[method](data, color_by=color_by, save_to=save_to)

    def rollout_gif(
        self,
        observation: np.ndarray,
        actions: list,
        save_to: str | Path = "rollout.gif",
        fps: int = 10,
    ) -> Path:
        """Generate an animated GIF of a latent-space rollout trajectory.

        Args:
            observation: Initial observation as numpy array (H, W, C).
            actions: List of action arrays.
            save_to: Output path for the GIF file.
            fps: Frames per second.

        Returns:
            Path to the saved GIF.
        """
        from worldkit.eval.rollout_gif import RolloutGIFGenerator

        gen = RolloutGIFGenerator(self)
        return gen.generate(
            observation, actions, save_to=save_to, fps=fps
        )

    @classmethod
    def compare(
        cls,
        models: dict[str, "WorldModel"],
        data_path: str | Path,
        episodes: int = 50,
        save_to: str | Path = "comparison.html",
    ):
        """Compare multiple world models and generate an HTML report.

        Args:
            models: Dict mapping model names to WorldModel instances.
            data_path: Path to HDF5 evaluation data.
            episodes: Number of episodes to evaluate.
            save_to: Output path for the HTML report.

        Returns:
            ComparisonResult with per-model metrics.
        """
        from worldkit.eval.comparison import ModelComparator

        comparator = ModelComparator(models)
        result = comparator.compare(data_path, episodes=episodes)
        comparator.report(result, save_to=save_to)
        return result

    # ─── Export ─────────────────────────────────────────

    def export(
        self,
        format: str = "onnx",
        output: str | Path = "./export/",
        optimize: bool = True,
        fp16: bool = True,
        int8: bool = False,
    ) -> Path:
        """Export the model for deployment.

        Args:
            format: Export format — ``"onnx"``, ``"torchscript"``,
                ``"tensorrt"``, or ``"coreml"``.
            output: Output directory for the exported file.
            optimize: Apply optimizations (ONNX only).
            fp16: Enable FP16 precision (TensorRT only, default True).
            int8: Enable INT8 quantization (TensorRT only).

        Returns:
            Path to the exported file.
        """
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)

        if format == "onnx":
            return self._export_onnx(output, optimize)
        elif format == "torchscript":
            return self._export_torchscript(output)
        elif format == "tensorrt":
            from worldkit.export.tensorrt_export import export_tensorrt

            return export_tensorrt(self, output, fp16=fp16, int8=int8)
        elif format == "coreml":
            from worldkit.export.coreml_export import export_coreml

            return export_coreml(self, output)
        else:
            raise ValueError(
                f"Unsupported format: {format}. "
                "Use 'onnx', 'torchscript', 'tensorrt', or 'coreml'."
            )

    def _export_onnx(self, output_dir: Path, optimize: bool) -> Path:

        path = output_dir / "worldkit_encoder.onnx"
        dummy = torch.randn(
            1, 3, self._config.image_size, self._config.image_size
        ).to(self._device)
        self._model.eval()

        torch.onnx.export(
            self._model.encoder,
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
        self._model.eval()
        traced = torch.jit.trace(self._model.encoder, dummy, check_trace=False)
        traced.save(str(path))
        print(f"WorldKit | Exported encoder to {path}")
        return path

    # ─── Distillation ──────────────────────────────────

    @classmethod
    def distill(
        cls,
        teacher: "WorldModel",
        student_config: str | ModelConfig = "nano",
        data: str | Path = "",
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 3e-4,
        device: str = "auto",
        seed: int = 42,
    ) -> "WorldModel":
        """Distill a teacher model into a smaller student.

        The student is trained to match the teacher's latent predictions
        via MSE loss, producing a smaller model with similar behaviour.

        Args:
            teacher: Trained WorldModel to distill from.
            student_config: Config name or ModelConfig for the student.
            data: Path to HDF5 training data.
            epochs: Number of distillation epochs.
            batch_size: Batch size.
            lr: Learning rate.
            device: Device string.
            seed: Random seed.

        Returns:
            Trained student WorldModel.
        """
        from .distillation import distill as _distill

        return _distill(
            teacher=teacher,
            student_config=student_config,
            data=data,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            seed=seed,
        )

    # ─── Online Learning ───────────────────────────────

    def enable_online_learning(
        self,
        lr: float = 1e-5,
        buffer_size: int = 1000,
        batch_size: int = 16,
        update_every: int = 4,
        ema_decay: float = 0.0,
    ) -> None:
        """Enable incremental online learning for this model.

        After calling this, use :meth:`update` to stream new experience
        and incrementally update the model weights.

        Args:
            lr: Learning rate for online updates.
            buffer_size: Maximum replay buffer capacity.
            batch_size: Mini-batch size sampled from the buffer.
            update_every: Perform a gradient step every N updates.
            ema_decay: EMA decay rate (0 disables EMA).
        """
        from .online import OnlineLearner

        self._online_learner = OnlineLearner(
            model=self,
            lr=lr,
            buffer_size=buffer_size,
            batch_size=batch_size,
            update_every=update_every,
            ema_decay=ema_decay,
        )

    def update(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
    ) -> float | None:
        """Add a transition and optionally perform an online gradient step.

        Requires :meth:`enable_online_learning` to be called first.

        Args:
            observation: Current observation (H, W, C).
            action: Action taken, shape (action_dim,).
            next_observation: Resulting observation.

        Returns:
            Loss value if a gradient step was performed, else None.
        """
        if not hasattr(self, "_online_learner"):
            raise RuntimeError(
                "Online learning not enabled. Call model.enable_online_learning() first."
            )
        return self._online_learner.step(observation, action, next_observation)

    # ─── Internal helpers ───────────────────────────────

    def _prepare_observation(self, obs: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert observation to model-ready tensor."""
        if isinstance(obs, np.ndarray):
            tensor = torch.from_numpy(obs).float()
        else:
            tensor = obs.float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        if tensor.dim() == 3 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        return tensor.to(self._device)
