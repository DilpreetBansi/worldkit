"""Linear probing suite for evaluating world model latent representations.

Trains lightweight linear probes (Ridge regression) on frozen latent embeddings
to measure what physical properties the model has learned (position, velocity,
angle, etc.). This is the standard evaluation protocol from the LeWM paper.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from worldkit.core.model import WorldModel


@dataclass
class ProbeResult:
    """Result from linear probing evaluation.

    Attributes:
        property_scores: R² score per probed property (higher is better, max 1.0).
        mse_scores: Mean squared error per probed property.
        probes: Trained sklearn Ridge models, keyed by property name.
        summary: Human-readable summary of all probe results.
    """

    property_scores: dict[str, float]
    mse_scores: dict[str, float]
    probes: dict = field(default_factory=dict, repr=False)
    summary: str = ""


class LinearProbe:
    """Train linear probes on a world model's latent space.

    Encodes observations through the frozen encoder, then fits a Ridge
    regression from latent vectors to each target property. Reports R²
    and MSE per property.

    Args:
        model: A WorldModel instance (used only for encoding).

    Usage:
        prober = LinearProbe(model)
        result = prober.fit("data.h5", ["agent_x", "agent_y"], "labels.csv")
        print(result.summary)
    """

    def __init__(self, model: WorldModel) -> None:
        self._model = model

    def fit(
        self,
        data_path: str | Path,
        properties: list[str],
        labels_path: str | Path,
        alpha: float = 1.0,
        test_fraction: float = 0.2,
        seed: int = 42,
    ) -> ProbeResult:
        """Fit linear probes and evaluate R² per property.

        Args:
            data_path: Path to HDF5 file with pixel observations.
            properties: List of property names to probe (must exist in labels).
            labels_path: Path to CSV or HDF5 file with per-frame property values.
            alpha: Ridge regularization strength.
            test_fraction: Fraction of data held out for evaluation.
            seed: Random seed for train/test split.

        Returns:
            ProbeResult with R² scores, MSE scores, trained probes, and summary.
        """
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        # 1. Load observations and encode them
        latents = self._encode_dataset(data_path)  # (N, latent_dim)

        # 2. Load labels
        labels = self._load_labels(labels_path, properties)  # (N, len(properties))

        # Ensure latents and labels align
        n_latents = latents.shape[0]
        n_labels = labels.shape[0]
        n_samples = min(n_latents, n_labels)
        if n_latents != n_labels:
            import warnings

            warnings.warn(
                f"Observation count ({n_latents}) != label count ({n_labels}). "
                f"Using first {n_samples} samples.",
                stacklevel=2,
            )
        latents = latents[:n_samples]
        labels = labels[:n_samples]

        # 3. Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            latents, labels, test_size=test_fraction, random_state=seed
        )

        # 4. Fit one Ridge probe per property
        property_scores: dict[str, float] = {}
        mse_scores: dict[str, float] = {}
        probes: dict = {}

        for i, prop in enumerate(properties):
            probe = Ridge(alpha=alpha)
            probe.fit(X_train, y_train[:, i])
            y_pred = probe.predict(X_test)

            r2 = float(r2_score(y_test[:, i], y_pred))
            mse = float(mean_squared_error(y_test[:, i], y_pred))

            property_scores[prop] = r2
            mse_scores[prop] = mse
            probes[prop] = probe

        # 5. Build summary
        summary = self._build_summary(property_scores, mse_scores)

        return ProbeResult(
            property_scores=property_scores,
            mse_scores=mse_scores,
            probes=probes,
            summary=summary,
        )

    def predict(self, observation: np.ndarray, probes: dict) -> dict[str, float]:
        """Predict property values for a single observation using trained probes.

        Args:
            observation: A single frame as numpy array (H, W, C) or (C, H, W).
            probes: Dict of trained sklearn Ridge models from ProbeResult.probes.

        Returns:
            Dict mapping property name to predicted value.
        """
        z = self._model.encode(observation)  # (latent_dim,)
        z_np = z.numpy().reshape(1, -1)  # (1, latent_dim)

        predictions: dict[str, float] = {}
        for prop, probe in probes.items():
            predictions[prop] = float(probe.predict(z_np)[0])
        return predictions

    def _encode_dataset(self, data_path: str | Path) -> np.ndarray:
        """Encode all observations in an HDF5 file to latent vectors.

        Returns:
            Numpy array of shape (N, latent_dim).
        """
        import h5py

        data_path = Path(data_path)
        with h5py.File(data_path, "r") as f:
            pixels_key = None
            for key in ("pixels", "observations", "obs", "images"):
                if key in f:
                    pixels_key = key
                    break
            if pixels_key is None:
                found = list(f.keys())
                raise KeyError(
                    f"No pixel data found in HDF5 file. "
                    f"Expected one of: 'pixels', 'observations', 'obs', 'images'. "
                    f"Found keys: {found}"
                )
            pixels = np.array(f[pixels_key])

        # Handle multi-episode layout: (N_episodes, T, H, W, C) → (N_episodes*T, H, W, C)
        if pixels.ndim == 5:
            n_eps, t_steps = pixels.shape[:2]
            pixels = pixels.reshape(n_eps * t_steps, *pixels.shape[2:])

        # Encode in batches to avoid OOM
        latents_list: list[np.ndarray] = []
        batch_size = 128
        self._model._model.eval()

        with torch.no_grad():
            for start in range(0, len(pixels), batch_size):
                batch = pixels[start : start + batch_size]
                batch_latents = []
                for frame in batch:
                    z = self._model.encode(frame)  # (latent_dim,)
                    batch_latents.append(z.numpy())
                latents_list.append(np.stack(batch_latents))

        return np.concatenate(latents_list, axis=0)  # (N, latent_dim)

    def _load_labels(
        self, labels_path: str | Path, properties: list[str]
    ) -> np.ndarray:
        """Load property labels from CSV or HDF5.

        CSV format: must have a header row with property names as columns.
        HDF5 format: each property is a dataset key with shape (N,) or (N_eps, T).

        Returns:
            Numpy array of shape (N, len(properties)).
        """
        labels_path = Path(labels_path)

        if labels_path.suffix in (".h5", ".hdf5"):
            return self._load_labels_hdf5(labels_path, properties)
        elif labels_path.suffix == ".csv":
            return self._load_labels_csv(labels_path, properties)
        else:
            raise ValueError(
                f"Unsupported labels format: {labels_path.suffix}. Use .csv or .h5"
            )

    def _load_labels_hdf5(
        self, path: Path, properties: list[str]
    ) -> np.ndarray:
        """Load labels from HDF5 file."""
        import h5py

        columns: list[np.ndarray] = []
        with h5py.File(path, "r") as f:
            found_keys = list(f.keys())
            for prop in properties:
                if prop not in f:
                    raise KeyError(
                        f"Property '{prop}' not found in labels HDF5. "
                        f"Available keys: {found_keys}"
                    )
                col = np.array(f[prop])
                # Flatten multi-episode layout: (N_eps, T) → (N_eps*T,)
                if col.ndim == 2:
                    col = col.reshape(-1)
                columns.append(col)

        return np.column_stack(columns)  # (N, len(properties))

    def _load_labels_csv(
        self, path: Path, properties: list[str]
    ) -> np.ndarray:
        """Load labels from CSV file."""
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file {path} has no header row")

            missing = [p for p in properties if p not in reader.fieldnames]
            if missing:
                raise KeyError(
                    f"Properties not found in CSV: {missing}. "
                    f"Available columns: {list(reader.fieldnames)}"
                )

            rows = list(reader)

        columns: list[np.ndarray] = []
        for prop in properties:
            columns.append(np.array([float(row[prop]) for row in rows]))

        return np.column_stack(columns)  # (N, len(properties))

    @staticmethod
    def _build_summary(
        property_scores: dict[str, float],
        mse_scores: dict[str, float],
    ) -> str:
        """Build a human-readable summary string."""
        lines = ["Linear Probe Results", "=" * 40]
        for prop in property_scores:
            r2 = property_scores[prop]
            mse = mse_scores[prop]
            bar = "#" * max(0, int(r2 * 20))
            lines.append(f"  {prop:<20s}  R²={r2:+.4f}  MSE={mse:.6f}  [{bar:<20s}]")

        avg_r2 = np.mean(list(property_scores.values()))
        lines.append("=" * 40)
        lines.append(f"  Average R²: {avg_r2:.4f}")
        return "\n".join(lines)
