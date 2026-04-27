# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

End-to-end deployment pipeline for **tiny-PULP-DroNet v3** on the **OAK-D Lite** (MyriadX VPU) via DepthAI v3. Three sequential steps: export PyTorch weights → compile to MyriadX blob → run live inference.

Verified: Python 3.13, depthai 3.5.0, torch 2.11.0+cpu, Windows 11.

## Commands

All commands require `PYTHONUTF8=1` on Windows to avoid Unicode errors in DepthAI logs.

```bash
# Activate venv (Git Bash / Windows)
source venv/Scripts/activate

# Step 1 — Export PyTorch → ONNX (opset 12, legacy exporter)
PYTHONUTF8=1 python export_to_onnx.py

# Step 2 — Compile ONNX → MyriadX blob (uploads to Luxonis cloud, requires internet)
PYTHONUTF8=1 python convert_to_blob.py

# Step 3 — Run live inference on OAK-D Lite
PYTHONUTF8=1 python run_dronet_oak.py
PYTHONUTF8=1 python run_dronet_oak.py --no-depth   # CNN-only, no stereo safety layer
PYTHONUTF8=1 python run_dronet_oak.py --blob path/to/custom.blob
```

If C: drive is low on space during pip install, redirect pip temp dirs:
```bash
TEMP=D:/_tmp_pip TMP=D:/_tmp_pip TMPDIR=D:/_tmp_pip PIP_CACHE_DIR=D:/_pip_cache \
  pip install -r requirements.txt onnxscript \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

`onnxscript` must be installed separately — required by torch 2.9+ but not in `requirements.txt`.

## Pipeline architecture

```
PyTorch weights (.pth)
    └─► export_to_onnx.py  →  dronet_tiny.onnx  (19.6 KB, opset 12)
            └─► convert_to_blob.py  →  dronet_tiny.blob  (32.2 KB, FP16, 6 SHAVE cores)
                    └─► run_dronet_oak.py  →  live OpenCV window

Runtime pipeline (run_dronet_oak.py):
    CAM_B (LEFT, OV9282, 640×400 GRAY8)
        ├─► ImageManip: crop(122,36,395,328) → resize(200,200)
        │       ├─► NeuralNetwork (blob on VPU) → steer ∈[-1,+1], coll ∈[0,1]
        │       └─► preview queue → display
        └─► StereoDepth (CAM_B + CAM_C, DENSITY preset, depth aligned to CAM_B)
                └─► depth queue (uint16 mm) → safety override + directional biases
```

## Key design decisions

**Why LEFT mono camera (not RGB):** The model was trained on the Himax HM01B0 monochrome sensor. OV9282 is already monochrome — a much closer match than converting RGB.

**Crop geometry:** Training used `CenterCrop(200)` on 324×244 Himax images (61.73% horiz, 81.97% vert). Scaled to 640×400 binned mode: `crop(122,36) → (517,364)` = 395×328, then resized to 200×200.

**VPU normalization:** `/255` normalisation is **not** done in Python — it is baked into the blob via `--scale_values=input[255]` at compile time (`convert_to_blob.py`). Do not add float normalization in the pipeline.

**ONNX opset 12:** Required because Luxonis uses OpenVINO 2022.1 which supports opset ≤ 15. The legacy exporter (`dynamo=False`) must be used.

**Depth safety override:** Median depth in the central 25% ROI < 2000 mm → force `coll = 1.0`. Mirrors the training label definition (VL53L1x ToF sensor threshold).

**Directional biases from depth:** `get_horizontal_bias()` and `get_vertical_bias()` split the central ROI into left/right and top/bottom halves. Median depth asymmetry → avoidance direction in `[-1, +1]`. Both feed the 2D avoidance arrow in the display. Arrow `dx = (steer + lateral) * 0.5 * ARROW_SCALE`.

**Model output tensor access:** Named access (`getTensor("steer")`, `getTensor("coll")`) is tried first; falls back to `getFirstTensor()` with index offsets if unavailable.

## DepthAI v3 API (critical)

The parent project CLAUDE.md has the full v3 API patterns. Key points specific to this directory:

- `pipeline.start()` is called explicitly — **not** `with pipeline:` (the context manager does not start the device here).
- Depth output is `RAW16 uint16 in mm`. Do **not** treat as GRAY8.
- `ImageManip.initialConfig.addCrop(x, y, w, h)` — arguments are `(x0, y0, width, height)`, not `(x0, y0, x1, y1)`.
- The run loop **must** call `cv2.waitKey(1)` each iteration or the device watchdog will time out and crash.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Device not found (Windows) | Install WinUSB driver via Zadig for "Movidius MyriadX" |
| `ModuleNotFoundError: onnxscript` | `pip install onnxscript` |
| `UnicodeEncodeError` in depthai logs | Set `PYTHONUTF8=1` |
| Stereo init error | Run with `--no-depth` to bypass |
| Low FPS (< 5) | Use USB 3 port with a quality USB-C cable; avoid hubs |
| Device drops after 1–2 min | Known hardware/cable issue; use powered hub or direct motherboard port |
| Wrong depth values | Confirm depth output is uint16 mm, not normalized float |
