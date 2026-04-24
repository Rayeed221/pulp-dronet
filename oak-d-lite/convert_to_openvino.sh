#!/usr/bin/env bash
# Convert dronet_v3.onnx → OpenVINO IR (dronet_v3.xml + dronet_v3.bin)
#
# The --mean_values/--scale_values flags bake the training normalisation
# (transforms.ToTensor() divides uint8 [0,255] by 255) into the IR graph.
# This lets inference_pipeline.py send raw uint8 pixels without any
# Python-side preprocessing.
#
# Requirements:
#   source /opt/intel/openvino/setupvars.sh   (or equivalent activation)
#   mo --version  # should print OpenVINO version
#
# Usage:
#   bash convert_to_openvino.sh [ONNX_MODEL] [OUTPUT_DIR] [PRECISION]
#
# Examples:
#   bash convert_to_openvino.sh dronet_v3.onnx ir_output FP16
#   bash convert_to_openvino.sh tiny_dronet_v3.onnx ir_tiny FP16

set -euo pipefail

ONNX_MODEL="${1:-dronet_v3.onnx}"
OUTPUT_DIR="${2:-ir_output}"
PRECISION="${3:-FP16}"   # FP16 is native on Myriad X; use FP32 for desktop debug

if ! command -v mo &>/dev/null; then
    echo "ERROR: 'mo' not found. Activate the OpenVINO environment first:"
    echo "  source /opt/intel/openvino/setupvars.sh"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Converting ${ONNX_MODEL} → ${OUTPUT_DIR}/dronet_v3.{xml,bin} [${PRECISION}]"

mo \
    --input_model "${ONNX_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name dronet_v3 \
    --input_shape "[1,1,200,200]" \
    --mean_values "[0]" \
    --scale_values "[255]" \
    --data_type "${PRECISION}" \
    --static_shape

echo ""
echo "Done. IR files written to ${OUTPUT_DIR}/"
echo "Next step: run convert_to_blob.sh to compile the .blob for DepthAI"
