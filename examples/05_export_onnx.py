"""Export a WorldKit model to ONNX or TorchScript.

Requires: pip install worldkit[export]
"""

from worldkit import WorldModel
from worldkit.core.config import get_config
from worldkit.core.jepa import JEPA
import tempfile
import os

config = get_config("nano", action_dim=2)
jepa = JEPA.from_config(config)
model = WorldModel(jepa, config, device="cpu")

with tempfile.TemporaryDirectory() as tmpdir:
    path = model.export(format="torchscript", output=tmpdir)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"TorchScript export: {path} ({size_mb:.1f} MB)")

print()
print("For ONNX export, install: pip install worldkit[export]")
print("Then use: model.export(format='onnx', output='./deploy/')")
