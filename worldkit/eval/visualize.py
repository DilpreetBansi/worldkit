"""Latent space visualization tools for WorldKit models.

Provides t-SNE, UMAP, and PCA projections of latent representations
encoded from HDF5 observation data. Useful for understanding what
the world model has learned and how it organizes its representations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from worldkit.core.model import WorldModel


class LatentVisualizer:
    """Visualize a world model's latent space via dimensionality reduction.

    Encodes observations from an HDF5 dataset, applies a reduction
    algorithm (PCA, t-SNE, or UMAP), and produces a scatter plot.

    Args:
        model: A WorldModel instance used for encoding.

    Usage:
        viz = LatentVisualizer(model)
        fig = viz.plot_pca("data.h5", color_by="episode", save_to="pca.png")
    """

    def __init__(self, model: WorldModel) -> None:
        self._model = model

    def plot_pca(
        self,
        data_path: str | Path,
        color_by: str | None = None,
        save_to: str | Path | None = None,
        max_frames: int = 5000,
    ):
        """Plot a PCA projection of latent embeddings.

        Args:
            data_path: Path to HDF5 file with pixel observations.
            color_by: Color points by ``"episode"`` or ``"timestep"``,
                or ``None`` for a single color.
            save_to: If provided, save the figure to this path.
            max_frames: Maximum number of frames to encode.

        Returns:
            matplotlib Figure.
        """
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError(
                "PCA requires scikit-learn. "
                "Install it with: pip install scikit-learn"
            )

        latents, colors, label = self._encode_dataset(
            data_path, max_frames, color_by
        )
        reducer = PCA(n_components=2)
        return self._reduce_and_plot(
            latents, colors, label, "PCA", reducer, save_to
        )

    def plot_tsne(
        self,
        data_path: str | Path,
        color_by: str | None = None,
        save_to: str | Path | None = None,
        max_frames: int = 5000,
    ):
        """Plot a t-SNE projection of latent embeddings.

        Args:
            data_path: Path to HDF5 file with pixel observations.
            color_by: Color points by ``"episode"`` or ``"timestep"``,
                or ``None`` for a single color.
            save_to: If provided, save the figure to this path.
            max_frames: Maximum number of frames to encode.

        Returns:
            matplotlib Figure.
        """
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            raise ImportError(
                "t-SNE requires scikit-learn. "
                "Install it with: pip install scikit-learn"
            )

        latents, colors, label = self._encode_dataset(
            data_path, max_frames, color_by
        )
        reducer = TSNE(n_components=2, perplexity=min(30, len(latents) - 1))
        return self._reduce_and_plot(
            latents, colors, label, "t-SNE", reducer, save_to
        )

    def plot_umap(
        self,
        data_path: str | Path,
        color_by: str | None = None,
        save_to: str | Path | None = None,
        max_frames: int = 5000,
    ):
        """Plot a UMAP projection of latent embeddings.

        Args:
            data_path: Path to HDF5 file with pixel observations.
            color_by: Color points by ``"episode"`` or ``"timestep"``,
                or ``None`` for a single color.
            save_to: If provided, save the figure to this path.
            max_frames: Maximum number of frames to encode.

        Returns:
            matplotlib Figure.
        """
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError(
                "UMAP requires the umap-learn package. "
                "Install it with: pip install umap-learn"
            )

        latents, colors, label = self._encode_dataset(
            data_path, max_frames, color_by
        )
        reducer = UMAP(n_components=2)
        return self._reduce_and_plot(
            latents, colors, label, "UMAP", reducer, save_to
        )

    # ─── Internal helpers ───────────────────────────────

    def _encode_dataset(
        self,
        data_path: str | Path,
        max_frames: int,
        color_by: str | None,
    ) -> tuple[np.ndarray, np.ndarray | None, str | None]:
        """Load HDF5 data, encode all frames, return latents and color info.

        Returns:
            Tuple of (latents, colors, color_label) where latents is
            (N, latent_dim), colors is (N,) floats or None, and
            color_label is the colorbar label string or None.
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
                    f"Expected one of: 'pixels', 'observations', "
                    f"'obs', 'images'. Found keys: {found}"
                )
            pixels = np.array(f[pixels_key])

        # Build episode/timestep indices before flattening
        if pixels.ndim == 5:
            n_eps, t_steps = pixels.shape[:2]
            episode_ids = np.repeat(np.arange(n_eps), t_steps)
            timestep_ids = np.tile(np.arange(t_steps), n_eps)
            pixels = pixels.reshape(n_eps * t_steps, *pixels.shape[2:])
        else:
            episode_ids = np.zeros(len(pixels))
            timestep_ids = np.arange(len(pixels))

        # Subsample if too many frames
        if len(pixels) > max_frames:
            indices = np.linspace(
                0, len(pixels) - 1, max_frames, dtype=int
            )
            pixels = pixels[indices]
            episode_ids = episode_ids[indices]
            timestep_ids = timestep_ids[indices]

        # Encode all frames
        latents = self._batch_encode(pixels)  # (N, latent_dim)

        # Build color array
        colors: np.ndarray | None = None
        label: str | None = None
        if color_by == "episode":
            colors = episode_ids.astype(np.float32)
            label = "Episode"
        elif color_by == "timestep":
            colors = timestep_ids.astype(np.float32)
            label = "Timestep"

        return latents, colors, label

    def _batch_encode(self, pixels: np.ndarray) -> np.ndarray:
        """Encode pixel frames to latent vectors in batches.

        Args:
            pixels: Array of shape (N, H, W, C) or (N, C, H, W).

        Returns:
            Numpy array of shape (N, latent_dim).
        """
        latents_list: list[np.ndarray] = []
        batch_size = 64
        self._model._model.eval()

        with torch.no_grad():
            for start in range(0, len(pixels), batch_size):
                batch = pixels[start : start + batch_size]
                batch_latents = []
                for frame in batch:
                    z = self._model.encode(frame)  # (latent_dim,)
                    batch_latents.append(z.numpy())
                latents_list.append(np.stack(batch_latents))

        return np.concatenate(latents_list, axis=0)

    def _reduce_and_plot(
        self,
        latents: np.ndarray,
        colors: np.ndarray | None,
        color_label: str | None,
        method_name: str,
        reducer,
        save_to: str | Path | None,
    ):
        """Apply dimensionality reduction and create a scatter plot.

        Args:
            latents: Array of shape (N, latent_dim).
            colors: Optional array of shape (N,) for coloring points.
            color_label: Label for the colorbar.
            method_name: Name of the reduction method (for the title).
            reducer: A fitted or unfitted sklearn-style transformer.
            save_to: If provided, save the figure to this path.

        Returns:
            matplotlib Figure.
        """
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Plotting requires matplotlib. "
                "Install it with: pip install matplotlib"
            )

        if save_to is not None:
            matplotlib.use("Agg")

        projected = reducer.fit_transform(latents)  # (N, 2)

        fig, ax = plt.subplots(figsize=(8, 6))

        scatter_kwargs: dict = {"s": 8, "alpha": 0.6}
        if colors is not None:
            scatter_kwargs["c"] = colors
            scatter_kwargs["cmap"] = "viridis"

        sc = ax.scatter(projected[:, 0], projected[:, 1], **scatter_kwargs)

        if colors is not None and color_label is not None:
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(color_label)

        ax.set_title(f"Latent Space — {method_name}")
        ax.set_xlabel(f"{method_name} 1")
        ax.set_ylabel(f"{method_name} 2")

        if save_to is not None:
            fig.savefig(str(save_to), dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

        return fig
