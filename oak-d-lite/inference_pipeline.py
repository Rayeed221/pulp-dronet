#!/usr/bin/env python3
"""
Live inference pipeline for PULP-DroNet v3 on OAK-D Lite.

The pipeline runs entirely on-device:
  MonoCamera (LEFT, 400×400 grayscale)
    → ImageManip (center-crop 400×400 → 200×200, GRAY8)
    → NeuralNetwork (.blob)
    → host: parse [yaw_rate, coll_prob]

The normalisation (÷255) that training applies via transforms.ToTensor() is
baked into the .blob by convert_to_openvino.sh (--mean_values/--scale_values),
so raw uint8 pixels are passed to the network without Python-side processing.

Usage:
    python inference_pipeline.py --blob blob_output/dronet_v3.blob [--show]

Output per frame:
    yaw_rate  : float ∈ [-1, 1]  (multiply by 90 for deg/s)
    coll_prob : float ∈ [0, 1]

Requirements:
    pip install depthai numpy opencv-python
"""

import argparse

import depthai as dai
import numpy as np

# OAK-D Lite mono sensor resolution.
# THE_400_P gives a 640×400 frame; we pick a 400×400 square by cropping the
# horizontal extent in ImageManip before passing to the network.
# If THE_400_P is unavailable on your firmware, see FALLBACK_RES below.
MONO_RES = dai.MonoCameraProperties.SensorResolution.THE_400_P

# Center-crop parameters for MONO_RES = THE_400_P (640 × 400 sensor)
# We want the central 400 columns out of 640, then crop to 200×200.
# Step 1 – crop horizontally to get 400×400:
#   xmin = (640-400)/(2*640) = 0.1875, xmax = 1-0.1875 = 0.8125
# Step 2 – crop vertically/horizontally to 200×200 (centre of 400×400):
#   Combined in one setCropRect operating on the 640×400 sensor:
#   x centre = 0.5, half-width  = 200/640 = 0.3125 → [0.1875, 0.8125]
#   y centre = 0.5, half-height = 200/400 = 0.25   → [0.25,   0.75  ]
CROP_XMIN = (640 - 200) / (2 * 640)   # 0.34375
CROP_XMAX = 1.0 - CROP_XMIN           # 0.65625
CROP_YMIN = (400 - 200) / (2 * 400)   # 0.25
CROP_YMAX = 1.0 - CROP_YMIN           # 0.75

CROP_SIZE = 200


def build_pipeline(blob_path: str) -> dai.Pipeline:
    pipeline = dai.Pipeline()

    # --- Mono camera ---
    mono = pipeline.create(dai.node.MonoCamera)
    mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono.setResolution(MONO_RES)

    # --- Center-crop to 200×200, keep GRAY8 ---
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setCropRect(CROP_XMIN, CROP_YMIN, CROP_XMAX, CROP_YMAX)
    manip.initialConfig.setResize(CROP_SIZE, CROP_SIZE)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
    manip.setMaxOutputFrameSize(CROP_SIZE * CROP_SIZE)

    # --- Neural network ---
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(blob_path)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    nn.input.setQueueSize(1)  # always use the newest frame

    # --- Outputs to host ---
    nn_out = pipeline.create(dai.node.XLinkOut)
    nn_out.setStreamName("nn_out")

    preview_out = pipeline.create(dai.node.XLinkOut)
    preview_out.setStreamName("preview")

    # --- Links ---
    mono.out.link(manip.inputImage)
    manip.out.link(nn.input)
    nn.out.link(nn_out.input)
    manip.out.link(preview_out.input)

    return pipeline


def parse_output(packet: dai.NNData):
    """
    Unpack the single (1, 2) output tensor into (yaw_rate, coll_prob).
    getFirstLayerFp16() returns a flat Python list of floats.
    Index 0 = yaw_rate (linear), index 1 = coll_prob (sigmoid).
    """
    raw = packet.getFirstLayerFp16()
    yaw_rate  = float(raw[0])
    coll_prob = float(raw[1])
    # Guard against FP16 extremes on unbounded yaw_rate output
    yaw_rate = max(-1.0, min(1.0, yaw_rate))
    return yaw_rate, coll_prob


def run(blob_path: str, show_preview: bool = False):
    pipeline = build_pipeline(blob_path)

    with dai.Device(pipeline) as device:
        nn_q      = device.getOutputQueue("nn_out",  maxSize=1, blocking=False)
        preview_q = device.getOutputQueue("preview", maxSize=1, blocking=False)

        print("Pipeline running. Press Ctrl-C to stop.")
        try:
            while True:
                nn_pkt = nn_q.tryGet()
                if nn_pkt is not None:
                    yaw_rate, coll_prob = parse_output(nn_pkt)
                    print(
                        f"yaw_rate={yaw_rate:+.4f} "
                        f"({yaw_rate * 90.0:+6.1f} deg/s)  "
                        f"coll_prob={coll_prob:.4f}"
                    )

                if show_preview:
                    import cv2
                    frame_pkt = preview_q.tryGet()
                    if frame_pkt is not None:
                        frame = frame_pkt.getCvFrame()
                        cv2.imshow("DroNet input (200x200)", frame)
                        if cv2.waitKey(1) == ord("q"):
                            break

        except KeyboardInterrupt:
            print("\nStopped.")


def main():
    p = argparse.ArgumentParser(
        description="Live PULP-DroNet v3 inference on OAK-D Lite"
    )
    p.add_argument("--blob", required=True,
                   help="Path to compiled .blob model file")
    p.add_argument("--show", action="store_true",
                   help="Display the 200×200 input crop in a window")
    args = p.parse_args()
    run(args.blob, show_preview=args.show)


if __name__ == "__main__":
    main()
