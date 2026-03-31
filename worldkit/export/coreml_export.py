"""CoreML export for WorldKit models.

Converts the encoder to a CoreML model for on-device Apple inference.
Requires: pip install coremltools
"""

from __future__ import annotations

from pathlib import Path

import torch


def export_coreml(
    model: object,
    output_dir: Path,
) -> Path:
    """Export the encoder to a CoreML ``.mlmodel`` file.

    First traces the encoder with TorchScript, then converts using
    ``coremltools``.

    Args:
        model: A WorldModel instance.
        output_dir: Directory to write the ``.mlmodel`` file.

    Returns:
        Path to the saved ``.mlmodel`` file.

    Raises:
        ImportError: If ``coremltools`` is not installed.
    """
    try:
        import coremltools as ct
    except ImportError:
        raise ImportError(
            "CoreML export requires: pip install coremltools"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = model._config.image_size
    dummy = torch.randn(1, 3, image_size, image_size).to(model._device)

    # Trace the encoder via TorchScript
    model._model.eval()
    traced = torch.jit.trace(model._model.encoder, dummy, check_trace=False)

    # Convert to CoreML
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="pixels",
                shape=(1, 3, image_size, image_size),
            )
        ],
        convert_to="mlprogram",
    )

    mlmodel_path = output_dir / "worldkit_encoder.mlpackage"
    mlmodel.save(str(mlmodel_path))

    print(f"WorldKit | Exported CoreML model to {mlmodel_path}")
    return mlmodel_path
