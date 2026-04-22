# OAK-D Lite · PULP-DroNet v3 · DepthAI v3

Run the **tiny-PULP-DroNet v3** model on an **OAK-D Lite** camera in real time using the **DepthAI v3** Python library. The pipeline predicts a steering angle and collision probability from the camera feed and overlays them on a live OpenCV window.

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

## How it works

```
LEFT mono camera (CAM_B, 1280×800, native grayscale)
    ├─► ImageManip  crop 790×656 from centre → resize 200×200
    │       ├─► NeuralNetwork (dronet_tiny.blob, MyriadX VPU)
    │       │       └─► XLinkOut "nn"      → steer + coll
    │       └─► XLinkOut "preview"         → frame for display
    └─► StereoDepth (LEFT + RIGHT cameras)
            └─► XLinkOut "depth"           → safety override
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
| Python 3.8 – 3.11 | 3.10 recommended |
| Internet access | `convert_to_blob.py` uploads to Luxonis cloud compiler |
| Pre-trained weights | `../model/tiny-pulp-dronet-v3-dw-pw-0.125.pth` |

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

### 2. Install Python

Download Python 3.10 from https://www.python.org/downloads/  
During installation, check **"Add Python to PATH"**.

Verify in a new Command Prompt:
```cmd
python --version
```

### 3. Create a virtual environment

Open **Command Prompt** (or PowerShell) and run:

```cmd
cd path\to\pulp-dronet\tiny-pulp-dronet-v3\oak_d_lite_dronet

python -m venv venv
venv\Scripts\activate
```

Your prompt should now show `(venv)`.

### 4. Install dependencies

```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch on Windows (CPU is sufficient for the export step):

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> If you have an NVIDIA GPU and want CUDA for faster export:
> ```cmd
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## Setup — Linux / macOS

```bash
cd tiny-pulp-dronet-v3/oak_d_lite_dronet

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

On Linux you may also need udev rules for USB access:

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## Step-by-step: Export → Convert → Run

### Step 1 — Export PyTorch model to ONNX

```cmd
python export_to_onnx.py
```

This loads `../model/tiny-pulp-dronet-v3-dw-pw-0.125.pth`, wraps it with
a `/255` normalisation layer (so the VPU receives raw uint8 pixels), and
exports to `dronet_tiny.onnx`. The script validates the output with
onnxruntime and prints the steering and collision values for a blank frame.

Expected output:
```
Weights : ...\model\tiny-pulp-dronet-v3-dw-pw-0.125.pth
Output  : ...\oak_d_lite_dronet\dronet_tiny.onnx
Exporting to ONNX (opset 12)...
Saved: dronet_tiny.onnx
OnnxRuntime check OK  steer=-0.0123  coll=0.4871
PyTorch output        steer=-0.0123  coll=0.4871
```

### Step 2 — Convert ONNX to MyriadX blob

> **Internet required.** The ONNX is uploaded to the Luxonis cloud compiler.

```cmd
python convert_to_blob.py
```

This produces `dronet_tiny.blob` (FP16, compiled for 6 SHAVE cores).

Options:
```cmd
python convert_to_blob.py --onnx dronet_tiny.onnx --output dronet_tiny.blob --shaves 6
```

Expected output:
```
ONNX input  : dronet_tiny.onnx
Blob output : dronet_tiny.blob
SHAVE cores : 6  (OAK-D Lite has 6 available)
Uploading to Luxonis blob compiler (requires internet)...
Blob saved  : dronet_tiny.blob
```

### Step 3 — Run live inference

Plug in the OAK-D Lite via USB, then:

```cmd
python run_dronet_oak.py
```

A window opens showing the camera feed with overlays:
- **Arrow** — steering direction and magnitude
- **Bar** — collision probability (green = safe, red = obstacle)
- **Depth** — stereo distance reading in metres
- **FPS** counter

Press **`q`** or **`ESC`** to quit.

#### Options

| Flag | Effect |
|---|---|
| `--blob PATH` | Use a custom blob file |
| `--no-depth` | Disable stereo depth safety layer (CNN predictions only) |

```cmd
python run_dronet_oak.py --no-depth
python run_dronet_oak.py --blob C:\path\to\custom.blob
```

---

## Troubleshooting — Windows

| Symptom | Fix |
|---|---|
| `depthai.XLinkError: Failed to find device` | Re-run Zadig and install WinUSB driver. Unplug/replug camera. |
| `OSError: [WinError 10054]` during blob conversion | Network issue. Retry or check firewall. |
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install torch torchvision` inside the venv. |
| `FileNotFoundError: dronet_tiny.blob` | Run Steps 1 and 2 first. |
| Black / blank camera window | Camera may need a moment; wait 2–3 seconds after the window opens. |
| `cv2.error: (-215)` display error | Install a display-capable OpenCV: `pip install opencv-python` |
| Low FPS (< 10) | Use USB 3 port. Try a shorter, higher-quality USB-C cable. |
| `StereoDepth` init error | Run with `--no-depth` flag to bypass stereo depth node. |

---

## Model details

| Property | Value |
|---|---|
| Architecture | Depthwise-Separable CNN (`depth_mult=0.125`, `bypass=False`) |
| Input | 200×200 grayscale, uint8 [0–255] |
| Output | `steer` float, `coll` float (after sigmoid) |
| Weights | `tiny-pulp-dronet-v3-dw-pw-0.125.pth` |
| Originally deployed on | Himax HM01B0 camera → GAP8 SoC (139 fps, 2.9 kB RAM) |

### Why LEFT mono camera?

Training images came from the **Himax HM01B0** — a native monochrome sensor.
The OAK-D Lite's LEFT/RIGHT stereo cameras (OV9282) are also native
monochrome, making them a much closer match than converting the RGB camera
to greyscale.

### Why this specific crop?

Training used `CenterCrop(200)` on 324×244 Himax images, keeping the
centre 61.7 % horizontal and 82.0 % vertical of the frame.
Applying the same fractions to the mono camera's 1280×800 resolution gives
a **790×656** centre crop at offset **(245, 72)**, which is then resized
to 200×200. This replicates the exact field-of-view geometry from training.

```
Himax 324×244  → CenterCrop(200):  x0=62,  y0=22  → x1=262,  y1=222
Mono  1280×800 → equiv crop:        x0=245, y0=72  → x1=1035, y1=728
                                    (790×656) → resize → 200×200
```

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
