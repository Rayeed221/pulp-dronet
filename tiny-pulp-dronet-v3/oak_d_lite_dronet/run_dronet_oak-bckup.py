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
# OAK-D Lite OV9282 mono cameras only support GRAY8 output in binned 640×400
# mode (not 1280×800). We use 640×400 and halve the crop coordinates:
#   Fraction from Himax 324×244:  horiz=200/324=0.6173, vert=200/244=0.8197
#   Applied to 1280×800: x0=245,y0=72, x1=1035,y1=728 → 790×656
#   Halved for 640×400:  x0=122,y0=36, x1=517,y1=364  → 395×328 → 200×200
MONO_W, MONO_H       = 640, 400
CROP_X0, CROP_Y0     = 122, 36
CROP_X1, CROP_Y1     = 517, 364   # 395×328 crop → resize → 200×200
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
    lateral: float,
    coll: float,
    coll_raw: float,
    depth_mm: float,
    fps: float,
    depth_override: bool,
):
    h, w    = canvas.shape[:2]
    img_h   = MODEL_INPUT_SIZE * DISP_SCALE
    gauge_y = img_h

    cv2.rectangle(canvas, (0, gauge_y), (w, h), (30, 30, 30), -1)

    # -----------------------------------------------------------------------
    # Horizontal avoidance arrow drawn on the camera image
    # steer : yaw from model  (+1 = left obstacle → go right)
    # lateral : depth left/right asymmetry  (+1 = left obstacle → go right)
    # Arrow is horizontal only — model has no trained vertical output.
    # -----------------------------------------------------------------------
    cx  = w // 2
    cy  = img_h // 2

    ARROW_SCALE = 100
    dx_raw = (steer + lateral) * 0.5   # average two horizontal sources → [-1, +1]
    dx     = int(dx_raw * ARROW_SCALE)

    magnitude = abs(dx)
    arrow_col = YELLOW if magnitude < 15 else (RED if coll > 0.5 else GREEN)

    if magnitude > 5:
        cv2.arrowedLine(canvas, (cx, cy), (cx + dx, cy),
                        arrow_col, 4, tipLength=0.35)
    else:
        cv2.circle(canvas, (cx, cy), 10, YELLOW, 2)

    # Crosshair at centre
    cv2.line(canvas, (cx - 15, cy), (cx + 15, cy), (80, 80, 80), 1)
    cv2.line(canvas, (cx, cy - 15), (cx, cy + 15), (80, 80, 80), 1)

    # -----------------------------------------------------------------------
    # Direction label (bottom of image, above gauge)
    # -----------------------------------------------------------------------
    dir_txt = "RIGHT" if dx_raw > 0.15 else ("LEFT" if dx_raw < -0.15 else "STRAIGHT")
    dir_col = YELLOW if dir_txt == "STRAIGHT" else arrow_col
    cv2.putText(canvas, dir_txt, (w - 140, img_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, dir_col, 2)

    # -----------------------------------------------------------------------
    # Gauge panel
    # -----------------------------------------------------------------------
    cv2.putText(canvas, f"Steer: {steer:+.3f}  Lateral: {lateral:+.3f}",
                (10, gauge_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

    # --- Collision bar ---
    bar_x0, bar_y0 = 10, gauge_y + 42
    bar_w,  bar_h  = w - 20, 20
    filled  = int(bar_w * float(np.clip(coll, 0, 1)))
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
    depth_col = RED if 0 < depth_mm < DEPTH_COLLISION_MM else WHITE
    cv2.putText(canvas, depth_str, (10, gauge_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, depth_col, 1)

    # --- CNN raw (before override) ---
    if depth_override:
        cv2.putText(canvas, f"CNN raw: {coll_raw:.3f}",
                    (10, gauge_y + 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, ORANGE, 1)

    # --- FPS ---
    cv2.putText(canvas, f"FPS: {fps:.1f}", (w - 100, gauge_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

    # --- Obstacle banner ---
    if depth_override:
        cv2.rectangle(canvas, (0, 0), (w, 26), RED, -1)
        cv2.putText(canvas, "! OBSTACLE DETECTED (depth < 2 m) !",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(blob_path: Path, use_depth: bool = True):
    """
    Build depthai v3 Pipeline and return (pipeline, q_nn, q_prev, q_depth).

    LEFT mono camera (CAM_B) → ImageManip crop+resize 200×200 → NeuralNetwork
    LEFT + RIGHT mono (CAM_C) → StereoDepth             (if use_depth=True)

    Uses v3 API: Camera.build(socket), createOutputQueue(), with pipeline context.
    """
    import depthai as dai

    pipeline = dai.Pipeline()

    # ------------------------------------------------------------------
    # LEFT mono camera  (CAM_B = OV9282, native grayscale)
    # ------------------------------------------------------------------
    left_cam = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_B
    )
    left_out = left_cam.requestOutput(
        (MONO_W, MONO_H), type=dai.ImgFrame.Type.GRAY8
    )

    # ------------------------------------------------------------------
    # ImageManip: centre-crop 790×656 then resize to 200×200
    # v3 API: addCrop(x, y, w, h) + setOutputSize(w, h)
    # ------------------------------------------------------------------
    CROP_W = CROP_X1 - CROP_X0   # 790
    CROP_H = CROP_Y1 - CROP_Y0   # 656
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.addCrop(CROP_X0, CROP_Y0, CROP_W, CROP_H)
    manip.initialConfig.setOutputSize(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
    manip.setMaxOutputFrameSize(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE)
    left_out.link(manip.inputImage)

    # ------------------------------------------------------------------
    # NeuralNetwork: DroNet on MyriadX VPU
    # blob input: (1,1,200,200) — scale_values=255 applied by VPU
    # ------------------------------------------------------------------
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(blob_path)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    nn.input.setMaxSize(1)
    manip.out.link(nn.input)

    # ------------------------------------------------------------------
    # StereoDepth: CAM_B (left) + CAM_C (right)
    # ------------------------------------------------------------------
    q_depth = None
    if use_depth:
        right_cam = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_C
        )
        right_out = right_cam.requestOutput(
            (MONO_W, MONO_H), type=dai.ImgFrame.Type.GRAY8
        )

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(
            dai.node.StereoDepth.PresetMode.DENSITY
        )
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)
        stereo.setOutputSize(MONO_W, MONO_H)

        left_out.link(stereo.left)
        right_out.link(stereo.right)

        q_depth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

    # ------------------------------------------------------------------
    # Output queues (v3: createOutputQueue replaces XLinkOut)
    # ------------------------------------------------------------------
    q_nn   = nn.out.createOutputQueue(maxSize=4, blocking=False)
    q_prev = manip.out.createOutputQueue(maxSize=4, blocking=False)

    return pipeline, q_nn, q_prev, q_depth


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


def get_horizontal_bias(depth_frame: np.ndarray) -> float:
    """
    Estimate horizontal avoidance direction from stereo depth.

    Splits the central ROI into left and right halves and compares their
    median depths. Returns a value in [-1, +1]:
      +1 = right half much closer → obstacle right → go LEFT
      -1 = left half much closer  → obstacle left  → go RIGHT
       0 = symmetric / no clear horizontal threat

    Sign convention matches steer: positive = go right (avoid left obstacle).
    """
    h, w = depth_frame.shape
    roi_h = int(h * DEPTH_ROI_FRAC)
    roi_w = int(w * DEPTH_ROI_FRAC)
    y0 = (h - roi_h) // 2
    x0 = (w - roi_w) // 2
    roi = depth_frame[y0:y0 + roi_h, x0:x0 + roi_w]

    mid = roi_w // 2
    left_valid  = roi[:, :mid][roi[:, :mid] > 0]
    right_valid = roi[:, mid:][roi[:, mid:] > 0]

    if left_valid.size == 0 or right_valid.size == 0:
        return 0.0

    left_med  = float(np.median(left_valid))
    right_med = float(np.median(right_valid))

    total = left_med + right_med
    if total < 1.0:
        return 0.0
    # positive = left closer = go right (matches steer sign convention)
    bias = (right_med - left_med) / total   # [-1, +1]

    if abs(bias) < 0.05:
        return 0.0
    return float(np.clip(bias, -1.0, 1.0))


def get_vertical_bias(depth_frame: np.ndarray) -> float:
    """
    Estimate vertical avoidance direction from stereo depth.

    Splits the central ROI into top and bottom halves and compares their
    median depths. Returns a value in [-1, +1]:
      +1 = top half much closer  → obstacle above → go DOWN
      -1 = bottom half much closer → obstacle below → go UP
       0 = symmetric / no clear vertical threat

    Only returns a non-zero value when depth is valid and the asymmetry
    exceeds a minimum threshold (avoids noise-driven corrections).
    """
    h, w = depth_frame.shape
    roi_h = int(h * DEPTH_ROI_FRAC)
    roi_w = int(w * DEPTH_ROI_FRAC)
    y0 = (h - roi_h) // 2
    x0 = (w - roi_w) // 2
    roi = depth_frame[y0:y0 + roi_h, x0:x0 + roi_w]

    mid = roi_h // 2
    top_valid    = roi[:mid, :][roi[:mid, :] > 0]
    bottom_valid = roi[mid:, :][roi[mid:, :] > 0]

    if top_valid.size == 0 or bottom_valid.size == 0:
        return 0.0

    top_med    = float(np.median(top_valid))
    bottom_med = float(np.median(bottom_valid))

    # Normalise asymmetry: positive = top closer = go down
    total = top_med + bottom_med
    if total < 1.0:
        return 0.0
    bias = (bottom_med - top_med) / total   # [-1, +1]

    # Suppress small noise (< 5 % asymmetry)
    if abs(bias) < 0.05:
        return 0.0
    return float(np.clip(bias, -1.0, 1.0))


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
    pipeline, q_nn, q_prev, q_depth = build_pipeline(blob_path, use_depth=use_depth)

    print("Connecting to OAK-D Lite...")
    pipeline.start()
    print("Connected. Press 'q' or ESC to quit.\n")

    try:
        fps_counter  = 0
        fps_t0       = time.time()
        fps          = 0.0
        depth_mm     = 0.0
        lateral      = 0.0   # horizontal avoidance bias from depth left/right asymmetry
        canvas       = np.zeros((DISP_H, DISP_W, 3), dtype=np.uint8)

        while pipeline.isRunning():
            nn_data    = q_nn.tryGet()
            frame      = q_prev.tryGet()
            depth_data = q_depth.tryGet() if q_depth else None

            # ---- Update depth measurements ----
            if depth_data is not None:
                depth_frame = depth_data.getCvFrame()   # uint16 mm
                depth_mm    = get_center_depth_mm(depth_frame)
                lateral     = get_horizontal_bias(depth_frame)

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
                # print(nn_data)
                
                try:
                    steer_raw = float(nn_data.getTensor("steer").flat[0])
                    coll_raw  = float(nn_data.getTensor("coll").flat[0])
                except Exception:
                    t         = nn_data.getFirstTensor()
                    steer_raw = float(t.flat[0])
                    coll_raw  = float(t.flat[1])

                steer    = float(np.clip(steer_raw, -1.0, 1.0))
                coll_raw = float(np.clip(coll_raw,  0.0, 1.0))

                # Depth safety override: obstacle within 2 m → force coll=1
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
                    canvas, steer, lateral, coll, coll_raw, depth_mm, fps, depth_override
                )
                print(
                    f"\rsteer={steer:+.4f}  lateral={lateral:+.4f}"
                    f"  coll={coll:.4f}  depth={depth_mm/1000:.2f}m  fps={fps:.1f}   ",
                    end="",
                    flush=True,
                )

            cv2.imshow("PULP-DroNet v3 — OAK-D Lite", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print()
                break

    finally:
        pipeline.stop()
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
