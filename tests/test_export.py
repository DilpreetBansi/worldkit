"""Tests for WorldKit export module — TensorRT, CoreML, ONNX, TorchScript."""

from __future__ import annotations

import pytest

# ─── Helpers ───────────────────────────────────────────


def _tensorrt_available() -> bool:
    try:
        import tensorrt  # noqa: F401

        return True
    except ImportError:
        return False


def _onnx_available() -> bool:
    try:
        import onnx  # noqa: F401

        return True
    except ImportError:
        return False


def _coremltools_available() -> bool:
    try:
        import coremltools  # noqa: F401

        return True
    except ImportError:
        return False


class _MockConfig:
    image_size = 32
    action_dim = 2


class _MockModel:
    """Minimal stand-in for WorldModel used in import-error tests."""

    _config = _MockConfig()
    _device = "cpu"
    _model = None


def _make_mock_model() -> _MockModel:
    return _MockModel()


def _make_real_model():
    """Build a real nano WorldModel for integration tests."""
    from worldkit import WorldModel
    from worldkit.core.backends import backend_registry
    from worldkit.core.config import get_config

    config = get_config("nano", action_dim=2)
    backend_cls = backend_registry.get(config.backend)
    backend = backend_cls()
    model_module = backend.build(config)
    return WorldModel(model_module, config, device="cpu", backend=backend)


# ─── Tests ─────────────────────────────────────────────


class TestTensorRTExport:
    """Tests for TensorRT export functionality."""

    def test_tensorrt_export_without_tensorrt(self, tmp_path, monkeypatch):
        """Importing tensorrt should raise ImportError with a helpful message."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "tensorrt":
                raise ImportError("No module named 'tensorrt'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from worldkit.export.tensorrt_export import export_tensorrt

        model = _make_mock_model()

        with pytest.raises(ImportError, match="pip install tensorrt"):
            export_tensorrt(model, tmp_path)

    @pytest.mark.skipif(
        not _tensorrt_available(),
        reason="tensorrt not installed",
    )
    def test_tensorrt_export_produces_engine(self, tmp_path):
        """If tensorrt is available, export should produce an .engine file."""
        from worldkit.export.tensorrt_export import export_tensorrt

        model = _make_real_model()
        path = export_tensorrt(model, tmp_path, fp16=False)
        assert path.exists()
        assert path.suffix == ".engine"


class TestCoreMLExport:
    """Tests for CoreML export functionality."""

    def test_coreml_export_without_coremltools(self, tmp_path, monkeypatch):
        """Importing coremltools should raise ImportError with a helpful message."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "coremltools":
                raise ImportError("No module named 'coremltools'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from worldkit.export.coreml_export import export_coreml

        model = _make_mock_model()

        with pytest.raises(ImportError, match="pip install coremltools"):
            export_coreml(model, tmp_path)

    @pytest.mark.skipif(
        not _coremltools_available(),
        reason="coremltools not installed",
    )
    def test_coreml_export_produces_mlpackage(self, tmp_path):
        """If coremltools is available, export should produce an .mlpackage."""
        from worldkit.export.coreml_export import export_coreml

        model = _make_real_model()
        path = export_coreml(model, tmp_path)
        assert path.exists()
        assert path.name == "worldkit_encoder.mlpackage"


class TestExportDispatch:
    """Tests for the WorldModel.export() dispatch logic."""

    def test_export_rejects_unknown_format(self, tmp_path):
        """Unknown format should raise ValueError with available options."""
        model = _make_real_model()
        with pytest.raises(ValueError, match="tensorrt"):
            model.export(format="flatbuffers", output=tmp_path)

    @pytest.mark.skipif(
        not _onnx_available(),
        reason="onnx not installed",
    )
    def test_export_onnx(self, tmp_path):
        """ONNX export should produce an .onnx file."""
        model = _make_real_model()
        path = model.export(format="onnx", output=tmp_path)
        assert path.exists()
        assert path.suffix == ".onnx"

    def test_export_torchscript(self, tmp_path):
        """TorchScript export should produce a .pt file."""
        model = _make_real_model()
        path = model.export(format="torchscript", output=tmp_path)
        assert path.exists()
        assert path.suffix == ".pt"
