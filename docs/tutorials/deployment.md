# Deployment Guide

## Save and Load

```python
model.save("my_model.wk")
loaded = WorldModel.load("my_model.wk")
```

## Export to ONNX

```python
model.export(format="onnx", output="./deploy/")
```

## Export to TorchScript

```python
model.export(format="torchscript", output="./deploy/")
```

## REST API Server

```bash
worldkit serve --model my_model.wk --port 8000
```
