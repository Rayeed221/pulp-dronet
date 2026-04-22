"""
Run tiny-PULP-DroNet v3 on the OAK-D Lite camera using depthai v3.

Usage:
    python run_dronet_oak.py [--blob PATH] [--no-depth]

Prerequisites:
    1. pip install -r requirements.txt
    2. python export_to_onnx.py           # creates dronet_tiny.onnx
    3. python convert_to_blob.py          # creates dronet_tiny.blob
    4. Connect OAK-D Lite via USB

Pipeline:
    LEFT mono camera (CAM_B, 1280×800, native grayscale)
        ├─► ImageManip  (crop 245,72→1035,728 then resize 200×200)
        │       ├─► NeuralNetwork (dronet_tiny.blob)
        │       │       └─► XLinkOut "nn"       → steer + coll from CNN
        │       └─► XLinkOut "preview"          → 200×200 frame for display
        └─► StereoDepth (with RIGHT camera)
                └─► XLinkOut "depth"            → depth map for safety override

== Why LEFT mono camera instead of RGB ==
The model was trained on images from the Himax HM01B0 — a monochrome
QVGA camera on the CrazyFlie AI-deck. The OAK-D Lite's LEFT/RIGHT stereo
cameras (OV9282) are already monochrome, which is a much closer match to
the training sensor than converting the RGB camera to grayscale.

== Why this specific crop ==
Training used transforms.CenterCrop(200) on 324×244 Himax images, which
keeps the centre 61.73% of horizontal and 81.97% of vertical pixels.
Applying those same fractions to the mono camera's native 1280×800 gives
a 790×656 centre crop, which we then resize to 200×200. This is the
closest achievable match to the training preprocessing geometry.

    Himax 324×244 → CenterCrop(200):  x0=62, y0=22 → x1=262, y1=222
    Mono  1280×800 → equiv crop:       x0=245, y0=72 → x1=1035, y1=728
                                       (790×656) → resize → 200×200

== Depth safety override ==
The collision labels in training were set to 1 when an obstacle was
within 2 m (measured by the VL53L1x ToF sensor). The OAK-D Lite's stereo
depth provides the same kind of frontal distance measurement. If the
median depth in the central ROI of the depth map is < 2000 mm, we force
collision = 1.0, regardless of what the CNN outputs. This is a hard
safety guarantee on top of the learned model.

Model outputs:
    steer : yaw-rate in [-1, +1]  (positive = left, negative = right)
    coll  : collision probability in [0, 1]
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR   = Path(__file__).resolve().parent
DEFAULT_BLOB = SCRIPT_DIR / "dronet_tiny.blob"

# ---------------------------------------------------------------------------
# Preprocessing constants (match training CenterCrop(200) from 324×244 Himax)
# ---------------------------------------------------------------------------
# Fraction of Himax frame used by the 200×200 centre crop:
#   horiz = 200/324 = 0.6173,  vert = 200/244 = 0.8197
# Applied to mono native resolution 1280×800:
MONO_W, MONO_H       = 1280, 800
CROP_X0, CROP_Y0     = 245, 72
CROP_X1, CROP_Y1     = 1035, 728   # 790×656 crop → resize → 200×200
MODEL_INPUT_SIZE     = 200

# Collision threshold matching training label definition (2 m = 2000 mm)
DEPTH_COLLISION_MM   = 2000
# Central ROI of the depth map used for obstacle distance (fraction of w/h)
DEPTH_ROI_FRAC       = 0.25

# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------
DISP_SCALE = 3
DISP_W     = MODEL_INPUT_SIZE * DISP_SCALE          # 600
DISP_H     = MODEL_INPUT_SIZE * DISP_SCALE + 130    # 730

# BGR colours
WHITE  = (255, 255, 255)
GREEN  = (0,   200, 0)
RED    = (0,   0,   220)
YELLOW = (0,   200, 200)
BLUE   = (220, 100, 0)
ORANGE = (0,   140, 255)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def draw_overlay(
    canvas: np.ndarray,
    steer: float,
    coll: float,
    coll_raw: float,
    depth_mm: float,
    fps: float,
    depth_override: bool,
):
    h, w   = canvas.shape[:2]
    img_h  = MODEL_INPUT_SIZE * DISP_SCALE
    gauge_y = img_h

    cv2.rectangle(canvas, (0, gauge_y), (w, h), (30, 30, 30), -1)

    # --- Steering arrow ---
    cx  = w // 2
    cy  = img_h - 20
    end_x = int(cx - steer * 90)
    arrow_col = YELLOW if abs(steer) < 0.15 else (GREEN if steer > 0 else BLUE)
    cv2.arrowedLine(canvas, (cx, cy), (end_x, cy), arrow_col, 3, tipLength=0.3)

    # --- Direction label ---
    if steer > 0.15:
        dir_txt, dir_col = "LEFT",     GREEN
    elif steer < -0.15:
        dir_txt, dir_col = "RIGHT",    BLUE
    else:
        dir_txt, dir_col = "STRAIGHT", YELLOW
    cv2.putText(canvas, dir_txt, (w - 120, gauge_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, dir_col, 2)

    cv2.putText(canvas, f"Steer: {steer:+.3f}",
                (10, gauge_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 1)

    # --- Collision bar (CNN output) ---
    bar_x0, bar_y0 = 10, gauge_y + 42
    bar_w,  bar_h  = w - 20, 20
    filled = int(bar_w * float(np.clip(coll, 0, 1)))
    bar_col = RED if coll > 0.5 else GREEN
    cv2.rectangle(canvas, (bar_x0, bar_y0),
                  (bar_x0 + bar_w, bar_y0 + bar_h), (70, 70, 70), -1)
    cv2.rectangle(canvas, (bar_x0, bar_y0),
                  (bar_x0 + filled, bar_y0 + bar_h), bar_col, -1)
    cv2.rectangle(canvas, (bar_x0, bar_y0),
                  (bar_x0 + bar_w, bar_y0 + bar_h), WHITE, 1)

    coll_label = f"Collision: {coll:.3f}"
    if depth_override:
        coll_label += f"  [depth override: {depth_mm/1000:.2f}m]"
    elif coll > 0.5:
        coll_label += "  !!!"
    cv2.putText(canvas, coll_label, (bar_x0, bar_y0 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, bar_col, 1)

    # --- Depth readout ---
    depth_str = f"Depth: {depth_mm/1000:.2f} m" if depth_mm > 0 else "Depth: --"
    depth_col = RED if depth_mm < DEPTH_COLLISION_MM else WHITE
    cv2.putText(canvas, depth_str, (10, gauge_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, depth_col, 1)

    # --- CNN raw (before override) label ---
    if depth_override:
        cv2.putText(canvas, f"CNN raw: {coll_raw:.3f}",
                    (10, gauge_y + 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, ORANGE, 1)

    # --- FPS ---
    cv2.putText(canvas, f"FPS: {fps:.1f}", (w - 100, gauge_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

    # --- Depth override banner ---
    if depth_override:
        cv2.rectangle(canvas, (0, 0), (w, 26), RED, -1)
        cv2.putText(canvas, "! OBSTACLE DETECTED (depth < 2 m) !",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(blob_path: Path, use_depth: bool = True):
    """
    Build depthai v3 Pipeline.

    LEFT mono camera → crop+resize to 200×200 → NeuralNetwork
    LEFT + RIGHT mono → StereoDepth               (if use_depth=True)
    """
    import depthai as dai

    pipeline = dai.Pipeline()

    # ------------------------------------------------------------------
    # LEFT mono camera  (OV9282, native grayscale — closest to Himax)
    # ------------------------------------------------------------------
    left_cam = pipeline.create(dai.node.Camera)
    left_cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
    # Request full native resolution for accurate crop geometry
    left_out = left_cam.requestOutput(
        (MONO_W, MONO_H), type=dai.ImgFrame.Type.GRAY8
    )

    # ------------------------------------------------------------------
    # ImageManip: centre-crop 790×656 then resize to 200×200
    # Matches training: CenterCrop(200) from 324×244 Himax QVGA image
    # ------------------------------------------------------------------
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setCropAbsolute(CROP_X0, CROP_Y0, CROP_X1, CROP_Y1)
    manip.initialConfig.setResize(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
    manip.setMaxOutputFrameSize(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE)
    left_out.link(manip.inputImage)

    # ------------------------------------------------------------------
    # NeuralNetwork: DroNet on MyriadX VPU
    # Input:  (1, 1, 200, 200) uint8 — norm /255 is baked into the blob
    # Output: steer (fp16 scalar), coll (fp16 scalar)
    # ------------------------------------------------------------------
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(blob_path)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    nn.input.setQueueSize(1)
    manip.out.link(nn.input)

    # ------------------------------------------------------------------
    # StereoDepth: uses LEFT + RIGHT mono cameras
    # Provides frontal depth to mirror the VL53L1x ToF sensor used for
    # generating collision labels during training.
    # ------------------------------------------------------------------
    if use_depth:
        right_cam = pipeline.create(dai.node.Camera)
        right_cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        right_out = right_cam.requestOutput(
            (MONO_W, MONO_H), type=dai.ImgFrame.Type.GRAY8
        )

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(
            dai.node.StereoDepth.PresetType.HIGH_DENSITY
        )
        stereo.setDepthAlign(dai.CameraBoardSocket.LEFT)
        stereo.setOutputSize(MONO_W, MONO_H)

        left_out.link(stereo.left)
        right_out.link(stereo.right)

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        xout_depth.input.setBlocking(False)
        xout_depth.input.setQueueSize(1)
        stereo.depth.link(xout_depth.input)

    # ------------------------------------------------------------------
    # XLinkOut: NN results
    # ------------------------------------------------------------------
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("nn")
    xout_nn.input.setBlocking(False)
    xout_nn.input.setQueueSize(1)
    nn.out.link(xout_nn.input)

    # ------------------------------------------------------------------
    # XLinkOut: preprocessed camera frame (200×200) for display
    # ------------------------------------------------------------------
    xout_prev = pipeline.create(dai.node.XLinkOut)
    xout_prev.setStreamName("preview")
    xout_prev.input.setBlocking(False)
    xout_prev.input.setQueueSize(1)
    manip.out.link(xout_prev.input)

    return pipeline


def get_center_depth_mm(depth_frame: np.ndarray) -> float:
    """
    Return the median non-zero depth (mm) inside a central ROI of the
    depth map. Returns 0.0 if no valid measurements exist.
    """
    h, w = depth_frame.shape
    roi_h = int(h * DEPTH_ROI_FRAC)
    roi_w = int(w * DEPTH_ROI_FRAC)
    y0 = (h - roi_h) // 2
    x0 = (w - roi_w) // 2
    roi = depth_frame[y0:y0 + roi_h, x0:x0 + roi_w]
    valid = roi[roi > 0]
    return float(np.median(valid)) if valid.size > 0 else 0.0


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run(blob_path: Path, use_depth: bool = True):
    import depthai as dai

    if not blob_path.exists():
        print(f"ERROR: blob not found at {blob_path}")
        print("Run the following steps first:")
        print("  python export_to_onnx.py")
        print("  python convert_to_blob.py")
        return

    print(f"Blob        : {blob_path}")
    print(f"Depth layer : {'enabled' if use_depth else 'disabled'}")
    print("Building pipeline...")
    pipeline = build_pipeline(blob_path, use_depth=use_depth)

    print("Connecting to OAK-D Lite...")
    with dai.Device(pipeline) as device:
        print(f"Device      : {device.getDeviceName()}")
        print(f"USB speed   : {device.getUsbSpeed().name}")
        print("Press 'q' or ESC to quit.\n")

        q_nn    = device.getOutputQueue("nn",      maxSize=4, blocking=False)
        q_prev  = device.getOutputQueue("preview", maxSize=4, blocking=False)
        q_depth = (
            device.getOutputQueue("depth", maxSize=4, blocking=False)
            if use_depth else None
        )

        fps_counter  = 0
        fps_t0       = time.time()
        fps          = 0.0
        depth_mm     = 0.0
        canvas       = np.zeros((DISP_H, DISP_W, 3), dtype=np.uint8)

        while True:
            nn_data    = q_nn.tryGet()
            frame      = q_prev.tryGet()
            depth_data = q_depth.tryGet() if q_depth else None

            # ---- Update depth measurement ----
            if depth_data is not None:
                depth_frame = depth_data.getCvFrame()   # uint16 mm
                depth_mm    = get_center_depth_mm(depth_frame)

            # ---- Update camera display ----
            if frame is not None:
                gray = frame.getCvFrame()               # (200, 200) uint8
                bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                upsc = cv2.resize(
                    bgr,
                    (DISP_W, MODEL_INPUT_SIZE * DISP_SCALE),
                    interpolation=cv2.INTER_NEAREST,
                )
                canvas[: MODEL_INPUT_SIZE * DISP_SCALE, :] = upsc

            # ---- Process NN output ----
            if nn_data is not None:
                try:
                    steer_raw = nn_data.getLayerFp16("steer")[0]
                    coll_raw  = nn_data.getLayerFp16("coll")[0]
                except Exception:
                    raw       = nn_data.getFirstLayerFp16()
                    steer_raw = float(raw[0])
                    coll_raw  = float(raw[1])

                steer = float(np.clip(steer_raw, -1.0, 1.0))
                coll_raw = float(np.clip(coll_raw, 0.0, 1.0))

                # Depth safety override: if stereo depth detects an obstacle
                # within 2 m, force collision = 1.0 (mirrors training label logic
                # which used the VL53L1x ToF sensor with a 2 m threshold).
                depth_override = (
                    use_depth
                    and depth_mm > 0
                    and depth_mm < DEPTH_COLLISION_MM
                )
                coll = 1.0 if depth_override else coll_raw

                fps_counter += 1
                elapsed = time.time() - fps_t0
                if elapsed >= 1.0:
                    fps          = fps_counter / elapsed
                    fps_counter  = 0
                    fps_t0       = time.time()

                draw_overlay(
                    canvas, steer, coll, coll_raw, depth_mm, fps, depth_override
                )
                print(
                    f"\rsteer={steer:+.4f}  coll={coll:.4f}"
                    f"  depth={depth_mm/1000:.2f}m  fps={fps:.1f}   ",
                    end="",
                    flush=True,
                )

            cv2.imshow("PULP-DroNet v3 — OAK-D Lite", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print()
                break

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run PULP-DroNet v3 on OAK-D Lite via depthai v3"
    )
    parser.add_argument(
        "--blob",
        type=Path,
        default=DEFAULT_BLOB,
        help=f"Path to MyriadX blob (default: {DEFAULT_BLOB})",
    )
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Disable stereo depth safety layer (CNN-only mode)",
    )
    args = parser.parse_args()
    run(args.blob, use_depth=not args.no_depth)


if __name__ == "__main__":
    main()
