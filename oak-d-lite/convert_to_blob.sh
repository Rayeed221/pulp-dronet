#!/usr/bin/env bash
# Convert OpenVINO IR → .blob for DepthAI / Myriad X
#
# Two modes:
#   1. From IR  (recommended after running convert_to_openvino.sh):
#        bash convert_to_blob.sh --from-ir ir_output/dronet_v3.xml ir_output/dronet_v3.bin
#
#   2. From ONNX directly (skips the separate 'mo' step via blobconverter API):
#        bash convert_to_blob.sh --from-onnx dronet_v3.onnx
#
# Options:
#   --shaves N      Number of SHAVE cores to allocate (default: 6, range 1-16)
#   --output-dir D  Output directory for .blob file (default: blob_output)
#
# Requirements:
#   pip install blobconverter

set -euo pipefail

MODE=""
ONNX_PATH=""
XML_PATH=""
BIN_PATH=""
SHAVES=6
OUTPUT_DIR="blob_output"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from-ir)
            MODE="ir"
            XML_PATH="${2:?'--from-ir requires xml path'}"
            BIN_PATH="${3:?'--from-ir requires bin path'}"
            shift 3
            ;;
        --from-onnx)
            MODE="onnx"
            ONNX_PATH="${2:?'--from-onnx requires onnx path'}"
            shift 2
            ;;
        --shaves)
            SHAVES="${2:?'--shaves requires a number'}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="${2:?'--output-dir requires a path'}"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash convert_to_blob.sh --from-ir <xml> <bin> [--shaves N] [--output-dir D]"
            echo "       bash convert_to_blob.sh --from-onnx <onnx>   [--shaves N] [--output-dir D]"
            exit 1
            ;;
    esac
done

if [[ -z "${MODE}" ]]; then
    echo "ERROR: specify --from-ir or --from-onnx"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

python3 - "${MODE}" "${XML_PATH}" "${BIN_PATH}" "${ONNX_PATH}" "${SHAVES}" "${OUTPUT_DIR}" <<'PYEOF'
import sys
import blobconverter

mode       = sys.argv[1]
xml_path   = sys.argv[2]
bin_path   = sys.argv[3]
onnx_path  = sys.argv[4]
shaves     = int(sys.argv[5])
output_dir = sys.argv[6]

if mode == "ir":
    blob_path = blobconverter.from_openvino(
        xml=xml_path,
        bin=bin_path,
        data_type="FP16",
        shaves=shaves,
        output_dir=output_dir,
        optimizer_params=[],
    )
else:
    # Chains mo + blobconverter server-side; bakes normalisation into the blob
    blob_path = blobconverter.from_onnx(
        model=onnx_path,
        data_type="FP16",
        shaves=shaves,
        output_dir=output_dir,
        optimizer_params=[
            "--mean_values=[0]",
            "--scale_values=[255]",
            "--input_shape=[1,1,200,200]",
            "--static_shape",
        ],
    )

print(f"Blob written → {blob_path}")
PYEOF
