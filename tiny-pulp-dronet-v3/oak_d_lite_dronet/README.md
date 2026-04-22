# OAK-D Lite · PULP-DroNet v3 · DepthAI v3

Run the **tiny-PULP-DroNet v3** model on an **OAK-D Lite** camera in real time using the **DepthAI v3** Python library. The pipeline predicts a steering angle and collision probability from the camera feed and overlays them on a live OpenCV window.

**Verified working on:** Python 3.13, depthai 3.5.0, torch 2.11.0+cpu, Windows 11, OAK-D Lite USB3 SUPER.

---

## Contents

```
oak_d_lite_dronet/
├── export_to_onnx.py   Step 1 — Export PyTorch model → ONNX
├── convert_to_blob.py  Step 2 — Convert ONNX → MyriadX blob (VPU)
├── run_dronet_oak.py   Step 3 — Live inference on OAK-D Lite
├── requirements.txt    Python dependencies
└── README.md           This file
```

---

## Quick-start commands (copy-paste)

```bash
# 1. Create venv and install dependencies
#    NOTE: if C: drive has less than ~600 MB free, redirect pip temp/cache to another drive:
TEMP=D:/_tmp_pip TMP=D:/_tmp_pip TMPDIR=D:/_tmp_pip PIP_CACHE_DIR=D:/_pip_cache \
  python -m venv venv

# Activate (Windows bash/Git Bash)
source venv/Scripts/activate

# Install all dependencies including torch CPU
TEMP=D:/_tmp_pip TMP=D:/_tmp_pip TMPDIR=D:/_tmp_pip PIP_CACHE_DIR=D:/_pip_cache \
  pip install -r requirements.txt onnxscript \
  --extra-index-url https://download.pytorch.org/whl/cpu

# 2. Export PyTorch weights to ONNX (opset 12, legacy exporter)
PYTHONUTF8=1 python export_to_onnx.py

# 3. Convert ONNX to MyriadX blob (requires internet — uploads to Luxonis cloud)
PYTHONUTF8=1 python convert_to_blob.py

# 4. Run live inference on OAK-D Lite
PYTHONUTF8=1 python run_dronet_oak.py

# Run without stereo depth safety layer (CNN predictions only)
PYTHONUTF8=1 python run_dronet_oak.py --no-depth
```

**Windows CMD / PowerShell equivalents:**

```cmd
:: Activate venv
venv\Scripts\activate

:: Export
set PYTHONUTF8=1
python export_to_onnx.py

:: Convert
python convert_to_blob.py

:: Run
python run_dronet_oak.py
```

---

## How it works

```
LEFT mono camera (CAM_B, 640×400 binned GRAY8, 30 fps)
    ├─► ImageManip  addCrop(122,36,395,328) → setOutputSize(200,200)
    │       ├─► NeuralNetwork (dronet_tiny.blob, MyriadX VPU)
    │       │       └─► output queue "nn"      → steer + coll
    │       └─► output queue "preview"         → frame for display
    └─► StereoDepth (CAM_B + CAM_C, DENSITY preset)
            └─► output queue "depth"           → safety override
```

**Outputs**

| Output | Range | Meaning |
|---|---|---|
| `steer` | −1 … +1 | Yaw-rate command. Positive = turn left, negative = turn right |
| `coll` | 0 … 1 | Collision probability. >0.5 = obstacle ahead |

**Depth safety layer** — if stereo depth reads < 2 m in the central field of view, `coll` is forced to 1.0, regardless of the CNN output. This mirrors how collision labels were generated during training (VL53L1x ToF sensor, 2 m threshold).

---

## Requirements

| Item | Notes |
|---|---|
| OAK-D Lite | Connected via USB 3 (USB-C cable) |
| Python 3.9 – 3.13 | Tested on Python 3.13 |
| Internet access | `convert_to_blob.py` uploads to Luxonis cloud compiler |
| Pre-trained weights | `../model/tiny-pulp-dronet-v3-dw-pw-0.125.pth` |
| ~600 MB free disk | For torch + depthai wheels download |

---

## Setup — Windows

### 1. Install USB driver (Windows only)

DepthAI requires **WinUSB** on Windows. Do this **before** plugging in the camera.

1. Download and run **Zadig** from https://zadig.akeo.ie
2. Plug in the OAK-D Lite
3. In Zadig: **Options → List All Devices**
4. Select **"Luxonis Device"** (or similar OAK device name)
5. Select driver: **WinUSB**
6. Click **Install Driver** and wait for completion
7. Unplug and re-plug the camera

> **Note:** If you later use the camera with another application (Depth AI Viewer, etc.) and the driver reverts, repeat the Zadig step.

### 2. Create a virtual environment and install

Open **Git Bash**, **Command Prompt**, or **PowerShell** in the `oak_d_lite_dronet/` directory:

```bash
python -m venv venv
source venv/Scripts/activate          # Git Bash
# OR: venv\Scripts\activate           # CMD / PowerShell

pip install --upgrade pip
pip install -r requirements.txt onnxscript --extra-index-url https://download.pytorch.org/whl/cpu
```

> **Disk space note:** torch and depthai together download ~280 MB. If your system drive (`C:`) has less than ~600 MB free, pip's temp directory will run out of space. Fix by redirecting before running pip:
> ```bash
> TEMP=D:/_tmp_pip TMP=D:/_tmp_pip TMPDIR=D:/_tmp_pip PIP_CACHE_DIR=D:/_pip_cache \
>   pip install -r requirements.txt onnxscript --extra-index-url https://download.pytorch.org/whl/cpu
> ```

---

## Setup — Linux / macOS

```bash
cd tiny-pulp-dronet-v3/oak_d_lite_dronet

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt onnxscript --extra-index-url https://download.pytorch.org/whl/cpu
```

On Linux you may also need udev rules for USB access:

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## Step-by-step: Export → Convert → Run

### Step 1 — Export PyTorch model to ONNX

```bash
PYTHONUTF8=1 python export_to_onnx.py
```

This loads `../model/tiny-pulp-dronet-v3-dw-pw-0.125.pth`, exports a **float32 input** model (opset 12, legacy TorchScript exporter via `dynamo=False`), and validates it with onnxruntime. The `/255` normalisation is applied by the VPU blob at inference time (via `scale_values=255` in the blob compiler).

Expected output:
```
Weights : ...\model\tiny-pulp-dronet-v3-dw-pw-0.125.pth
Output  : ...\oak_d_lite_dronet\dronet_tiny.onnx
Exporting to ONNX (opset 12, legacy TorchScript exporter)...
Saved: dronet_tiny.onnx
OnnxRuntime check OK  steer=-0.0096  coll=0.6908
PyTorch output        steer=-0.0096  coll=0.6908
```

> **torch 2.9+ note:** torch 2.11 requires the `onnxscript` package for its new ONNX exporter. Installing via `pip install onnxscript` is handled by the setup command above. The legacy exporter (`dynamo=False`) is used here because the Luxonis blob compiler (OpenVINO 2022.1) supports ONNX opset ≤ 15.

### Step 2 — Convert ONNX to MyriadX blob

> **Internet required.** The ONNX is uploaded to the Luxonis cloud compiler.

```bash
PYTHONUTF8=1 python convert_to_blob.py
```

This produces `dronet_tiny.blob` (FP16, 6 SHAVE cores, `scale_values=255` baked in so the VPU handles uint8→float normalisation).

Options:
```bash
python convert_to_blob.py --onnx dronet_tiny.onnx --output dronet_tiny.blob --shaves 6
```

Expected output:
```
ONNX input  : dronet_tiny.onnx
Blob output : dronet_tiny.blob
SHAVE cores : 6  (OAK-D Lite has 6 available)
Uploading to Luxonis blob compiler (requires internet)...
[==================================================]
Done
Blob saved  : dronet_tiny.blob
```

### Step 3 — Run live inference

Plug in the OAK-D Lite via USB 3, then:

```bash
PYTHONUTF8=1 python run_dronet_oak.py
```

A window opens showing the camera feed with overlays:
- **Arrow** — steering direction and magnitude
- **Bar** — collision probability (green = safe, red = obstacle)
- **Depth** — stereo distance reading in metres
- **FPS** counter

Press **`q`** or **`ESC`** to quit.

**Verified console output (steady state, camera pointed at room ~7.5 m away):**
```
steer=+0.0063  coll=0.4329  depth=7.50m  fps=0.0
steer=+0.0848  coll=0.4741  depth=7.92m  fps=0.0
steer=-0.0752  coll=0.6123  depth=7.44m  fps=26.5
steer=-0.0142  coll=0.7090  depth=7.34m  fps=30.1
steer=+0.0175  coll=0.5020  depth=7.54m  fps=30.0
```
FPS stabilises at ~30 Hz after ~1 second warmup.

#### Options

| Flag | Effect |
|---|---|
| `--blob PATH` | Use a custom blob file |
| `--no-depth` | Disable stereo depth safety layer (CNN predictions only) |

```bash
python run_dronet_oak.py --no-depth
python run_dronet_oak.py --blob /path/to/custom.blob
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `depthai.XLinkError: Failed to find device` | Re-run Zadig and install WinUSB driver. Unplug/replug camera. |
| `OSError: [Errno 28] No space left on device` during pip | C: drive full. Redirect pip temp/cache to another drive (see setup note above). |
| `ModuleNotFoundError: No module named 'onnxscript'` | `pip install onnxscript` — required by torch 2.9+. |
| `UnicodeEncodeError: 'charmap' ... ✅` | Set `PYTHONUTF8=1` before running. |
| `No available sensor config ... 1280x800 ... resizeMode: CROP` | Expected — OV9282 mono cameras only support GRAY8 at 640×400 (binned mode). Already handled in script. |
| `'depthai.NNData' has no attribute 'getLayerFp16'` | depthai v3 API change — use `getTensor(name)`. Already fixed in script. |
| `'depthai.node.Camera' has no attribute 'setBoardSocket'` | depthai v3 API change — use `.build(socket)`. Already fixed in script. |
| `pipeline.isRunning()` always False, no frames | depthai v3 needs explicit `pipeline.start()` — `with pipeline:` does not boot the device. Already fixed in script. |
| `400 Client Error: BAD REQUEST` from blobconverter | ONNX exported with opset > 15, or 3-channel mean/scale applied to 1-channel input. Re-export with `dynamo=False` (opset 12). |
| `FileNotFoundError: dronet_tiny.blob` | Run Steps 1 and 2 first. |
| Low FPS (< 10) | Use USB 3 port. Try a shorter, higher-quality USB-C cable. |
| `StereoDepth` init error | Run with `--no-depth` flag to bypass stereo depth node. |
| Device crash / watchdog timeout after ~15s | depthai v3 requires the host to consume frames in a tight loop. Ensure `cv2.waitKey(1)` (not `time.sleep(1)`) drives the loop. |

---

## depthai v3 API changes (v2 → v3 migration notes)

The scripts in this directory use the **depthai v3 API** throughout. Key differences from v2:

| v2 (old) | v3 (new) | Notes |
|---|---|---|
| `cam = pipeline.create(dai.node.Camera)` + `cam.setBoardSocket(socket)` | `cam = pipeline.create(dai.node.Camera).build(socket)` | `.build()` sets socket and initialises camera |
| `dai.CameraBoardSocket.LEFT` / `RIGHT` | `dai.CameraBoardSocket.CAM_B` / `CAM_C` | CAM_A=RGB, CAM_B=left mono, CAM_C=right mono |
| `manip.initialConfig.setCropAbsolute(x0,y0,x1,y1)` | `manip.initialConfig.addCrop(x0,y0,w,h)` | Width+height instead of two corners |
| `manip.initialConfig.setResize(w,h)` | `manip.initialConfig.setOutputSize(w,h)` | Renamed |
| `nn.input.setQueueSize(n)` | `nn.input.setMaxSize(n)` | Renamed |
| `pipeline.create(dai.node.XLinkOut)` + stream name | `out.createOutputQueue(maxSize, blocking)` | No XLinkOut nodes needed |
| `with dai.Device(pipeline) as device:` + `device.getOutputQueue(name)` | `pipeline.start()` + hold queue references from `createOutputQueue()` | Pipeline IS the device manager in v3 |
| `device.getDeviceName()`, `device.getUsbSpeed()` | Same methods, called on `dai.Device()` opened separately | `dai.Device(pipeline)` constructor no longer valid in v3 |
| `nn_data.getLayerFp16("name")` | `nn_data.getTensor("name").flat[0]` | Returns numpy array |
| `nn_data.getFirstLayerFp16()` | `nn_data.getFirstTensor().flat[n]` | Returns numpy array |
| `dai.node.StereoDepth.PresetType.HIGH_DENSITY` | `dai.node.StereoDepth.PresetMode.DENSITY` | Renamed enum |

---

## Camera resolution note

The OAK-D Lite's OV9282 mono cameras (CAM_B, CAM_C) **do not support GRAY8 output at 1280×800**. The only GRAY8 mode available via `requestOutput` is **640×400** (2×2 binned). The crop coordinates are halved accordingly:

```
Himax 324×244  → CenterCrop(200):  x0=62,  y0=22  → x1=262, y1=222
Mono  1280×800 → equiv crop:        x0=245, y0=72  → x1=1035,y1=728  (790×656)
Mono  640×400  → halved crop:       x0=122, y0=36  → x1=517, y1=364  (395×328) → 200×200
```

The geometric relationship to the training data is preserved — only absolute pixel coordinates change.

---

## Model details

| Property | Value |
|---|---|
| Architecture | Depthwise-Separable CNN (`depth_mult=0.125`, `bypass=False`) |
| Input | 200×200 grayscale, float32 [0–1] (VPU applies `/255` from raw uint8) |
| Output | `steer` float ∈ [−1,1], `coll` float ∈ [0,1] |
| Weights | `tiny-pulp-dronet-v3-dw-pw-0.125.pth` |
| Originally deployed on | Himax HM01B0 camera → GAP8 SoC (139 fps, 2.9 kB RAM) |
| ONNX opset | 12 (legacy TorchScript exporter, required for OpenVINO 2022.1) |
| Blob | FP16, 6 SHAVE cores, `scale_values=255` baked in |

### Why LEFT mono camera?

Training images came from the **Himax HM01B0** — a native monochrome sensor.
The OAK-D Lite's LEFT/RIGHT stereo cameras (OV9282) are also native
monochrome, making them a much closer match than converting the RGB camera
to greyscale.

---

## What about a depth-aware model?

No depth-input model exists in this repository. The VL53L1x ToF sensor
used during data collection only provided collision **labels** — it was
never fed into the CNN as an input channel.

The stereo depth safety layer added here is the practical equivalent:
it reads the OAK-D Lite's built-in stereo depth and overrides the
collision output when an obstacle is geometrically confirmed within 2 m,
without requiring any retraining.

If you want a model that actually uses depth as a second input channel,
you would need to:
1. Collect new training data with paired (image, depth) samples
2. Modify `model/dronet_v3.py` to accept a 2-channel input
3. Retrain from scratch (or fine-tune) on the new dataset
