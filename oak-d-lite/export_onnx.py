#!/usr/bin/env python3
"""
Export PULP-DroNet v3 to ONNX for OpenVINO / OAK-D Lite deployment.

dronet_v3.py uses nemo.quant.pact.PACT_IntegerAdd inside block __init__ when
bypass=True.  If nemo is not installed the name is unbound and instantiation
raises NameError.  We inject a fake nemo module into sys.modules before the
import so PACT_IntegerAdd resolves to a plain torch.add wrapper instead.

Usage (pretrained ResBlock 1.0):
    python export_onnx.py \\
        --weights ../tiny-pulp-dronet-v3/model/pulp-dronet-v3-resblock-1.0.pth \\
        --block_type ResBlock --depth_mult 1.0 --bypass True \\
        --output dronet_v3.onnx

Usage (tiny Depthwise 0.125):
    python export_onnx.py \\
        --weights ../tiny-pulp-dronet-v3/model/tiny-pulp-dronet-v3-dw-pw-0.125.pth \\
        --block_type Depthwise --depth_mult 0.125 --bypass False \\
        --output tiny_dronet_v3.onnx
"""

import argparse
import os
import sys
import types

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fake NEMO injection — must happen before dronet_v3 is imported
# ---------------------------------------------------------------------------
class _PlainAdd(nn.Module):
    """Drop-in for nemo.quant.pact.PACT_IntegerAdd for FP32 inference."""
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.add(x, y)

_fake_nemo = types.ModuleType("nemo")
_fake_nemo.quant = types.ModuleType("nemo.quant")
_fake_nemo.quant.pact = types.ModuleType("nemo.quant.pact")
_fake_nemo.quant.pact.PACT_IntegerAdd = _PlainAdd
sys.modules["nemo"] = _fake_nemo
sys.modules["nemo.quant"] = _fake_nemo.quant
sys.modules["nemo.quant.pact"] = _fake_nemo.quant.pact

# Add v3 source to path
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


# ---------------------------------------------------------------------------
# ONNX-exportable wrapper
# ---------------------------------------------------------------------------
class DronetONNXWrapper(nn.Module):
    """
    Merges dronet's [steer, coll] list output into a single (batch, 2) tensor
    so OpenVINO and DepthAI see one output layer rather than two.
    Output layout: [:, 0] = yaw_rate (linear), [:, 1] = coll_prob (sigmoid).
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        steer, coll = self.model(x)
        return torch.stack([steer, coll], dim=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_model(block_type: str, depth_mult: float, bypass: bool) -> nn.Module:
    return dronet(
        depth_mult=depth_mult,
        block_class=BLOCK_MAP[block_type],
        bypass=bypass,
        nemo=False,
    )


def load_checkpoint(model: nn.Module, weights_path: str) -> nn.Module:
    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] Missing keys (expected for FP32 checkpoints): {missing}")
    if unexpected:
        print(f"[warn] Unexpected keys: {unexpected}")
    model.eval()
    return model


def export_onnx(model: nn.Module, output_path: str, opset: int = 11):
    wrapper = DronetONNXWrapper(model)
    wrapper.eval()
    dummy = torch.zeros(1, 1, 200, 200)
    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
    )
    print(f"ONNX model exported → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export PULP-DroNet v3 to ONNX")
    p.add_argument("--weights", required=True, help="Path to .pth checkpoint")
    p.add_argument(
        "--block_type",
        choices=list(BLOCK_MAP.keys()),
        default="ResBlock",
        help="Block type used during training",
    )
    p.add_argument("--depth_mult", type=float, default=1.0,
                   help="Channel depth multiplier used during training")
    p.add_argument(
        "--bypass",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether bypass connections were used (True/False)",
    )
    p.add_argument("--output", default="dronet_v3.onnx",
                   help="Output ONNX file path")
    p.add_argument("--opset", type=int, default=11,
                   help="ONNX opset version (11+ required for OpenVINO)")
    return p


def main():
    args = create_parser().parse_args()
    print(f"Building model: block={args.block_type}, depth_mult={args.depth_mult}, "
          f"bypass={args.bypass}")
    model = build_model(args.block_type, args.depth_mult, args.bypass)
    model = load_checkpoint(model, args.weights)
    export_onnx(model, args.output, opset=args.opset)


if __name__ == "__main__":
    main()
