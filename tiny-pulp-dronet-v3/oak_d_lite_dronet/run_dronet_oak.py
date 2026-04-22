"""
Run tiny-PULP-DroNet v3 on the OAK-D Lite camera using depthai v3.

Usage:
    python run_dronet_oak.py [--blob PATH]

Prerequisites:
    1. pip install -r requirements.txt
    2. python export_to_onnx.py           # creates dronet_tiny.onnx
    3. python convert_to_blob.py          # creates dronet_tiny.blob
    4. Connect OAK-D Lite via USB

Pipeline:
    Camera (RGB, CAM_A)
        └─► ImageManip  (resize 200×200, convert to GRAY8)
                ├─► NeuralNetwork (dronet_tiny.blob)
                │       └─► XLinkOut "nn"     → steer + coll outputs
                └─► XLinkOut "preview"        → camera frame for display

Model outputs:
    steer  : yaw-rate in [-1, +1]  (negative = right, positive = left)
    coll   : collision probability in [0, 1]
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BLOB = SCRIPT_DIR / "dronet_tiny.blob"

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
FRAME_W, FRAME_H = 200, 200
DISP_SCALE = 3          # upscale for readability (200→600 px)
DISP_W = FRAME_W * DISP_SCALE
DISP_H = FRAME_H * DISP_SCALE + 120   # extra strip at bottom for gauges

# Colours (BGR)
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
GREEN  = (0,   200, 0)
RED    = (0,   0,   200)
YELLOW = (0,   200, 200)
BLUE   = (200, 80,  0)


def draw_overlay(canvas: np.ndarray, steer: float, coll: float, fps: float):
    """Draw steering arrow, collision bar and FPS onto canvas (in-place)."""
    h, w = canvas.shape[:2]
    gauge_y = FRAME_H * DISP_SCALE  # top of the gauge strip

    # --- Background for gauge strip ---
    cv2.rectangle(canvas, (0, gauge_y), (w, h), (30, 30, 30), -1)

    # --- Steering arrow ---
    # Centre origin at bottom of image area
    cx = w // 2
    cy = FRAME_H * DISP_SCALE - 20
    arrow_len = 80
    # steer > 0 → left, steer < 0 → right
    end_x = int(cx - steer * arrow_len)
    end_y = cy
    color = YELLOW if abs(steer) < 0.3 else (GREEN if steer > 0 else BLUE)
    cv2.arrowedLine(canvas, (cx, cy), (end_x, end_y), color, 3, tipLength=0.3)
    cv2.putText(canvas,
                f"Steer: {steer:+.3f}",
                (10, gauge_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

    # --- Collision probability bar ---
    bar_x0, bar_y0 = 10, gauge_y + 50
    bar_w, bar_h = w - 20, 22
    filled = int(bar_w * np.clip(coll, 0.0, 1.0))
    bar_color = RED if coll > 0.5 else GREEN
    cv2.rectangle(canvas, (bar_x0, bar_y0),
                  (bar_x0 + bar_w, bar_y0 + bar_h), (80, 80, 80), -1)
    cv2.rectangle(canvas, (bar_x0, bar_y0),
                  (bar_x0 + filled, bar_y0 + bar_h), bar_color, -1)
    cv2.rectangle(canvas, (bar_x0, bar_y0),
                  (bar_x0 + bar_w, bar_y0 + bar_h), WHITE, 1)
    label = f"Collision: {coll:.3f}" + ("  !!!" if coll > 0.5 else "")
    cv2.putText(canvas, label, (bar_x0, bar_y0 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, bar_color, 1)

    # --- FPS ---
    cv2.putText(canvas, f"FPS: {fps:.1f}", (w - 110, gauge_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

    # --- Direction label ---
    if steer > 0.1:
        dir_label, dir_color = "LEFT", GREEN
    elif steer < -0.1:
        dir_label, dir_color = "RIGHT", BLUE
    else:
        dir_label, dir_color = "STRAIGHT", YELLOW
    cv2.putText(canvas, dir_label, (w - 130, gauge_y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, dir_color, 2)


def build_pipeline(blob_path: Path):
    """
    Construct and return a depthai v3 Pipeline.

    Camera (RGB CAM_A) → ImageManip (200×200 GRAY8) → NeuralNetwork
    """
    import depthai as dai

    pipeline = dai.Pipeline()

    # ------------------------------------------------------------------
    # Camera node  (depthai v3: dai.node.Camera replaces ColorCamera)
    # ------------------------------------------------------------------
    cam = pipeline.create(dai.node.Camera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)  # RGB centre camera
    # Request a 300×300 preview output — ImageManip handles exact crop/resize
    cam_out = cam.requestOutput((300, 300), type=dai.ImgFrame.Type.NV12)

    # ------------------------------------------------------------------
    # ImageManip: resize to 200×200 and convert colour → grayscale
    # ------------------------------------------------------------------
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(200, 200)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
    manip.setMaxOutputFrameSize(200 * 200)
    cam_out.link(manip.inputImage)

    # ------------------------------------------------------------------
    # NeuralNetwork: run DroNet inference on MyriadX
    # ------------------------------------------------------------------
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(blob_path)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    nn.input.setQueueSize(1)
    manip.out.link(nn.input)

    # ------------------------------------------------------------------
    # XLinkOut nodes
    # ------------------------------------------------------------------
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("nn")
    xout_nn.input.setBlocking(False)
    xout_nn.input.setQueueSize(1)
    nn.out.link(xout_nn.input)

    xout_prev = pipeline.create(dai.node.XLinkOut)
    xout_prev.setStreamName("preview")
    xout_prev.input.setBlocking(False)
    xout_prev.input.setQueueSize(1)
    manip.out.link(xout_prev.input)

    return pipeline


def run(blob_path: Path):
    import depthai as dai

    if not blob_path.exists():
        print(f"ERROR: blob not found at {blob_path}")
        print("Run the following steps first:")
        print("  python export_to_onnx.py")
        print("  python convert_to_blob.py")
        return

    print(f"Loading blob : {blob_path}")
    print("Building pipeline...")
    pipeline = build_pipeline(blob_path)

    print("Connecting to OAK-D Lite...")
    with dai.Device(pipeline) as device:
        print(f"Device  : {device.getDeviceName()}")
        print(f"USB     : {device.getUsbSpeed().name}")
        print("Press 'q' or ESC to quit.")

        q_nn   = device.getOutputQueue("nn",      maxSize=4, blocking=False)
        q_prev = device.getOutputQueue("preview", maxSize=4, blocking=False)

        fps_counter = 0
        fps_t0      = time.time()
        fps         = 0.0

        # Canvas: upscaled camera frame + gauge strip
        canvas = np.zeros((DISP_H, DISP_W, 3), dtype=np.uint8)

        while True:
            nn_data = q_nn.tryGet()
            frame   = q_prev.tryGet()

            if frame is not None:
                # GRAY8 → BGR for display
                gray = frame.getCvFrame()                    # (200, 200) uint8
                bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                upsc = cv2.resize(bgr, (DISP_W, FRAME_H * DISP_SCALE),
                                  interpolation=cv2.INTER_NEAREST)
                canvas[: FRAME_H * DISP_SCALE, :] = upsc

            if nn_data is not None:
                # Retrieve outputs by layer name
                # blobconverter preserves the ONNX output names "steer"/"coll"
                try:
                    steer = nn_data.getLayerFp16("steer")[0]
                    coll  = nn_data.getLayerFp16("coll")[0]
                except Exception:
                    # Fallback: first and second element of first layer
                    raw = nn_data.getFirstLayerFp16()
                    steer, coll = float(raw[0]), float(raw[1])

                fps_counter += 1
                elapsed = time.time() - fps_t0
                if elapsed >= 1.0:
                    fps      = fps_counter / elapsed
                    fps_counter = 0
                    fps_t0  = time.time()

                draw_overlay(canvas, steer, coll, fps)

                print(f"\rsteer={steer:+.4f}  coll={coll:.4f}  fps={fps:.1f}   ",
                      end="", flush=True)

            cv2.imshow("PULP-DroNet v3 — OAK-D Lite", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print()
                break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Run PULP-DroNet v3 on OAK-D Lite via depthai v3"
    )
    parser.add_argument(
        "--blob",
        type=Path,
        default=DEFAULT_BLOB,
        help=f"Path to MyriadX blob file (default: {DEFAULT_BLOB})",
    )
    args = parser.parse_args()
    run(args.blob)


if __name__ == "__main__":
    main()
