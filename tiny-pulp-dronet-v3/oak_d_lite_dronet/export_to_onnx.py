"""
Export the tiny-PULP-DroNet v3 PyTorch model to ONNX format for deployment
on the OAK-D Lite MyriadX VPU via depthai.

Usage:
    python export_to_onnx.py

Output:
    dronet_tiny.onnx  -- ready for blobconverter (see convert_to_blob.py)

Model used: tiny-pulp-dronet-v3-dw-pw-0.125.pth
  - block_type: Depthwise_Separable
  - depth_mult: 0.125
  - bypass: False   <-- no NEMO dependency needed

The full ResBlock model (bypass=True) uses nemo.quant.pact.PACT_IntegerAdd,
a custom PyTorch op that cannot be cleanly exported to ONNX. Use the tiny
model for OAK-D Lite deployment.

Input: float32 (1, 1, 200, 200) in [0, 1]. The /255 normalization is applied
by OpenVINO Model Optimizer during blob compilation (scale_values=255) so the
VPU can receive raw GRAY8 uint8 frames directly from ImageManip.
"""

import sys
import os
from pathlib import Path

import torch
import torch.nn as nn

# Locate the tiny-pulp-dronet-v3 root (one level up from this script)
SCRIPT_DIR = Path(__file__).resolve().parent
DRONET_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(DRONET_ROOT))

from model.dronet_v3 import dronet, Depthwise_Separable

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WEIGHTS_PATH = DRONET_ROOT / "model" / "tiny-pulp-dronet-v3-dw-pw-0.125.pth"
OUTPUT_ONNX  = SCRIPT_DIR / "dronet_tiny.onnx"

def load_model(weights_path: Path, device: torch.device) -> nn.Module:
    net = dronet(depth_mult=0.125, block_class=Depthwise_Separable, bypass=False)

    state = torch.load(str(weights_path), map_location=device)
    # Handle checkpoints that store the state dict under a key
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Strip any "module." prefix from DataParallel saves
    state = {k.replace("module.", ""): v for k, v in state.items()}

    net.load_state_dict(state)
    net.eval()
    return net


def export(weights_path: Path = WEIGHTS_PATH, output_path: Path = OUTPUT_ONNX):
    print(f"Weights : {weights_path}")
    print(f"Output  : {output_path}")

    device = torch.device("cpu")
    model = load_model(weights_path, device)
    model.eval()

    # Float32 input — normalization (/255) is handled by OpenVINO Model
    # Optimizer (scale_values=255) during blob compilation, so the VPU
    # receives raw uint8 from ImageManip and scales internally.
    dummy = torch.zeros(1, 1, 200, 200, dtype=torch.float32)

    print("Exporting to ONNX (opset 12, legacy TorchScript exporter)...")
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["steer", "coll"],
        opset_version=12,
        do_constant_folding=True,
        dynamic_axes=None,  # fixed batch size 1 for MyriadX
        dynamo=False,       # use legacy exporter: opset 12, single file, no onnxscript
    )
    print(f"Saved: {output_path}")

    # -----------------------------------------------------------------------
    # Sanity-check with onnxruntime
    # -----------------------------------------------------------------------
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(str(output_path),
                                    providers=["CPUExecutionProvider"])
        # Simulate normalized float32 input (as VPU will see after /255)
        dummy_np = np.zeros((1, 1, 200, 200), dtype=np.float32)
        steer_out, coll_out = sess.run(None, {"input": dummy_np})
        print(f"OnnxRuntime check OK  steer={steer_out[0]:.4f}  coll={coll_out[0]:.4f}")
    except ImportError:
        print("onnxruntime not installed — skipping verification.")

    # -----------------------------------------------------------------------
    # Compare against PyTorch output for correctness
    # -----------------------------------------------------------------------
    with torch.no_grad():
        s_pt, c_pt = model(dummy)
    print(f"PyTorch output        steer={s_pt.item():.4f}  coll={c_pt.item():.4f}")


if __name__ == "__main__":
    export()
