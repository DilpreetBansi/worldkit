"""TensorRT export for WorldKit models.

Converts the encoder to a TensorRT engine for high-throughput GPU inference.
Requires: pip install tensorrt
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch


def export_tensorrt(
    model: object,
    output_dir: Path,
    fp16: bool = True,
    int8: bool = False,
) -> Path:
    """Export the encoder to a TensorRT engine file.

    First exports to ONNX, then converts to TensorRT. Requires the
    ``tensorrt`` Python package.

    Args:
        model: A WorldModel instance.
        output_dir: Directory to write the engine file.
        fp16: Enable FP16 precision (default True, much faster on modern GPUs).
        int8: Enable INT8 quantization (requires calibration data).

    Returns:
        Path to the saved ``.engine`` file.

    Raises:
        ImportError: If ``tensorrt`` is not installed.
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError(
            "TensorRT export requires: pip install tensorrt. "
            "See https://developer.nvidia.com/tensorrt"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Export to ONNX as intermediate format
    onnx_path = output_dir / "worldkit_encoder.onnx"
    image_size = model._config.image_size
    dummy = torch.randn(1, 3, image_size, image_size).to(model._device)
    model._model.eval()

    torch.onnx.export(
        model._model.encoder,
        dummy,
        str(onnx_path),
        input_names=["pixels"],
        output_names=["latent"],
        dynamic_axes={"pixels": {0: "batch"}, "latent": {0: "batch"}},
        opset_version=17,
    )

    # Step 2: Build TensorRT engine from ONNX
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = "\n".join(
                str(parser.get_error(i)) for i in range(parser.num_errors)
            )
            raise RuntimeError(f"Failed to parse ONNX model:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)

    # Build the serialized engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(
            "TensorRT engine build failed. Check that your GPU supports "
            "the requested precision (fp16/int8)."
        )

    engine_path = output_dir / "worldkit_encoder.engine"
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    # Clean up intermediate ONNX
    onnx_path.unlink()

    print(
        f"WorldKit | Exported TensorRT engine to {engine_path} "
        f"(fp16={fp16}, int8={int8})"
    )
    return engine_path


def benchmark_tensorrt(
    engine_path: Path,
    input_shape: tuple,
    n_runs: int = 100,
) -> dict:
    """Run an inference benchmark on a TensorRT engine.

    Args:
        engine_path: Path to the ``.engine`` file.
        input_shape: Input tensor shape, e.g. ``(1, 3, 96, 96)``.
        n_runs: Number of inference iterations (default 100).

    Returns:
        Dict with ``avg_latency_ms`` and ``throughput_fps``.

    Raises:
        ImportError: If ``tensorrt`` is not installed.
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError(
            "TensorRT benchmarking requires: pip install tensorrt. "
            "See https://developer.nvidia.com/tensorrt"
        )

    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate device memory
    input_data = np.random.randn(*input_shape).astype(np.float32)
    d_input = cuda.mem_alloc(input_data.nbytes)

    # Determine output shape from engine
    output_shape = tuple(engine.get_tensor_shape(engine.get_tensor_name(1)))
    # Replace dynamic -1 dims with the batch size from input
    output_shape = tuple(
        input_shape[0] if s == -1 else s for s in output_shape
    )
    output_data = np.empty(output_shape, dtype=np.float32)
    d_output = cuda.mem_alloc(output_data.nbytes)

    stream = cuda.Stream()

    # Warm up
    for _ in range(10):
        cuda.memcpy_htod_async(d_input, input_data, stream)
        context.execute_async_v2(
            bindings=[int(d_input), int(d_output)], stream_handle=stream.handle
        )
        stream.synchronize()

    # Benchmark
    latencies = []
    for _ in range(n_runs):
        cuda.memcpy_htod_async(d_input, input_data, stream)
        start = time.perf_counter()
        context.execute_async_v2(
            bindings=[int(d_input), int(d_output)], stream_handle=stream.handle
        )
        stream.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    avg_latency = sum(latencies) / len(latencies)
    batch_size = input_shape[0]
    throughput = (batch_size / avg_latency) * 1000  # frames per second

    return {
        "avg_latency_ms": round(avg_latency, 3),
        "throughput_fps": round(throughput, 1),
    }
