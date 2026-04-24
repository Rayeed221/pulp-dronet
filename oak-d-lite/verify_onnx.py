#!/usr/bin/env python3
"""
Numerically verify that the exported ONNX model matches the PyTorch model.

Runs the same random input through both runtimes and asserts outputs agree
within FP32 tolerance before you commit to blob conversion.

Usage:
    python verify_onnx.py \\
        --weights ../tiny-pulp-dronet-v3/model/pulp-dronet-v3-resblock-1.0.pth \\
        --onnx dronet_v3.onnx \\
        --block_type ResBlock --depth_mult 1.0 --bypass True

Requirements:
    pip install onnxruntime numpy torch
"""

import argparse
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Same NEMO monkeypatch as export_onnx.py
# ---------------------------------------------------------------------------
class _PlainAdd(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.add(x, y)

_fake_nemo = types.ModuleType("nemo")
_fake_nemo.quant = types.ModuleType("nemo.quant")
_fake_nemo.quant.pact = types.ModuleType("nemo.quant.pact")
_fake_nemo.quant.pact.PACT_IntegerAdd = _PlainAdd
sys.modules["nemo"] = _fake_nemo
sys.modules["nemo.quant"] = _fake_nemo.quant
sys.modules["nemo.quant.pact"] = _fake_nemo.quant.pact

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tiny-pulp-dronet-v3"))
from model.dronet_v3 import (  # noqa: E402
    dronet,
    ResBlock,
    Depthwise_Separable,
    Inverted_Linear_Bottleneck,
)

BLOCK_MAP = {
    "ResBlock": ResBlock,
    "Depthwise": Depthwise_Separable,
    "IRLB": Inverted_Linear_Bottleneck,
}


def run_pytorch(weights: str, block_type: str, depth_mult: float,
                bypass: bool, dummy_np: np.ndarray) -> np.ndarray:
    model = dronet(
        depth_mult=depth_mult,
        block_class=BLOCK_MAP[block_type],
        bypass=bypass,
        nemo=False,
    )
    ckpt = torch.load(weights, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.eval()

    with torch.no_grad():
        steer, coll = model(torch.from_numpy(dummy_np))
    return np.array([steer.item(), coll.item()], dtype=np.float32)


def run_onnx(onnx_path: str, dummy_np: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": dummy_np})[0]  # shape (1, 2)
    return out[0].astype(np.float32)              # shape (2,)


def main():
    p = argparse.ArgumentParser(description="Verify ONNX output vs PyTorch")
    p.add_argument("--weights", required=True)
    p.add_argument("--onnx", required=True)
    p.add_argument("--block_type", choices=list(BLOCK_MAP.keys()), default="ResBlock")
    p.add_argument("--depth_mult", type=float, default=1.0)
    p.add_argument("--bypass", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--atol", type=float, default=1e-4,
                   help="Absolute tolerance for output comparison")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    dummy_np = rng.random((1, 1, 200, 200), dtype=np.float32)

    pt_out  = run_pytorch(args.weights, args.block_type, args.depth_mult,
                          args.bypass, dummy_np)
    ort_out = run_onnx(args.onnx, dummy_np)

    print(f"PyTorch  : yaw_rate={pt_out[0]:.6f}  coll_prob={pt_out[1]:.6f}")
    print(f"ONNX RT  : yaw_rate={ort_out[0]:.6f}  coll_prob={ort_out[1]:.6f}")
    print(f"Max diff : {np.abs(pt_out - ort_out).max():.2e}")

    if np.allclose(pt_out, ort_out, atol=args.atol):
        print(f"PASS: outputs match within atol={args.atol}")
    else:
        print(f"FAIL: outputs differ beyond atol={args.atol}")
        sys.exit(1)


if __name__ == "__main__":
    main()
