"""
Convert dronet_tiny.onnx to a MyriadX blob for the OAK-D Lite VPU.

Usage:
    python convert_to_blob.py [--onnx PATH] [--output PATH] [--shaves N]

Prerequisites:
    pip install blobconverter

The blobconverter library uploads the ONNX model to Luxonis' compilation
service (tools.luxonis.com) and downloads the resulting blob. An internet
connection is required.

Blob output: dronet_tiny.blob (same directory as this script by default).
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def convert(onnx_path: Path, output_path: Path, shaves: int):
    try:
        import blobconverter
    except ImportError:
        print("ERROR: blobconverter is not installed.")
        print("Install with:  pip install blobconverter")
        sys.exit(1)

    if not onnx_path.exists():
        print(f"ERROR: ONNX model not found at {onnx_path}")
        print("Run export_to_onnx.py first.")
        sys.exit(1)

    print(f"ONNX input  : {onnx_path}")
    print(f"Blob output : {output_path}")
    print(f"SHAVE cores : {shaves}  (OAK-D Lite has 6 available)")
    print("Uploading to Luxonis blob compiler (requires internet)...")

    blob_path = blobconverter.from_onnx(
        model=str(onnx_path),
        data_type="FP16",
        shaves=shaves,
        use_cache=False,
        output_dir=str(output_path.parent),
    )

    # blobconverter names the file after the model; rename to our target name
    blob_path = Path(blob_path)
    if blob_path.resolve() != output_path.resolve():
        blob_path.rename(output_path)

    print(f"Blob saved  : {output_path}")
    print()
    print("Next step:")
    print(f"  python run_dronet_oak.py --blob {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DroNet ONNX model to MyriadX blob"
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=SCRIPT_DIR / "dronet_tiny.onnx",
        help="Path to input ONNX file (default: dronet_tiny.onnx)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "dronet_tiny.blob",
        help="Path for output blob file (default: dronet_tiny.blob)",
    )
    parser.add_argument(
        "--shaves",
        type=int,
        default=6,
        help="Number of SHAVE cores to use (default: 6, OAK-D Lite maximum)",
    )
    args = parser.parse_args()
    convert(args.onnx, args.output, args.shaves)


if __name__ == "__main__":
    main()
