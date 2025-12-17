````md
# 🃏 PokerVision

> **PokerVision** is a computer-vision project that detects and classifies playing cards in real time using a YOLO model.  
> It combines **synthetic dataset generation**, **YOLO training**, and a **real-time “Detect / Texas Hold’em” demo** with optional equity estimation.

---

## 🚀 Features

- 🎨 **Synthetic dataset generator** (random backgrounds + card variants)
- 🧠 **YOLO (Ultralytics)** training + inference for card classification
- 🎥 **Real-time inference** from:
  - Laptop webcam (index `0`, `1`, …)
  - OBS Virtual Camera
  - DroidCam / phone streams (HTTP/RTSP if your app provides it)
- 🃏 **Two realtime modes**
  - **Detect mode**: show individual card detections (single boxes)
  - **Hold’em mode**: cluster detections into *board + player hands* and (optionally) compute equities
- 🧩 Built-in camera probing: `--list_cams`
- ⌨️ Quick mode toggle: press **`m`** (and **`q`** to quit)
- ♠♥♦♣ pretty suit symbols (optional, if your OpenCV build supports `cv2.freetype` + you provide a font)

---

## 📂 Project Structure

```plaintext
PokerVision/
├── data/
│   ├── backgrounds/         # Background images for synthetic data
│   ├── generated/           # Auto-generated images (often ignored by git)
│   ├── raw_cards/           # Source card images (normal/inverted/real)
│   └── yolo_dataset/        # YOLO-formatted dataset (train/val images & labels)
│
├── runs/                    # YOLO training results (often ignored by git)
├── src/
│   ├── dataset_gen/         # Dataset generation scripts
│   ├── realtime/            # Real-time demos
│   │   └── webcam_card_corners.py
│   └── utils/               # Shared helper functions (e.g., CUDA check)
│
├── cards.yaml               # YOLO dataset config
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
````

> Note: folders like `runs/`, `data/generated/`, and large weights are typically ignored by git.

---

## 🧩 Installation

### 1) Clone the repo

```bash
git clone https://github.com/Yufan3/PokerVision.git
cd PokerVision
```

### 2) Create a conda environment

```bash
conda create -n pokervision python=3.10 -y
conda activate pokervision
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Verify CUDA (optional)

```bash
python -m src.utils.check_cuda
```

---

## ⚠️ Important: OpenCV GUI support

This project opens a live window (`cv2.namedWindow`, `cv2.imshow`).

If you see an error like:

* `The function is not implemented ... in function 'cvNamedWindow'`

You likely installed a **headless** OpenCV build or you’re in an environment without GUI support.

Fix by installing GUI-enabled OpenCV:

```bash
pip uninstall -y opencv-python-headless
pip install opencv-python
```

---

## 🧠 Dataset Generation

Generate a synthetic dataset (backgrounds + random card placements):

```bash
python -m src.dataset_gen.generate_dataset
```

Outputs:

```bash
data/generated/
data/yolo_dataset/
```

---

## 🏋️‍♂️ Model Training (Ultralytics YOLO)

Train a detector on your dataset:

```bash
yolo detect train model=yolo11s.pt data=cards.yaml epochs=100 imgsz=640 batch=16 device=0
```

Best weights will be saved to:

```bash
runs/detect/train*/weights/best.pt
```

---

## 🎥 Real-Time Demo (Detect + Texas Hold’em)

The main realtime entrypoint is:

* `src/realtime/webcam_card_corners.py`

### Basic run (webcam index 0)

```bash
python -m src.realtime.webcam_card_corners --model runs/detect/train*/weights/best.pt --source 0
```

Controls:

* Press **`m`** to toggle **Detect / Hold’em**
* Press **`q`** to quit

### Start directly in Detect or Hold’em mode

```bash
# Start in Detect mode
python -m src.realtime.webcam_card_corners --model runs/detect/train*/weights/best.pt --source 0 --start_mode detect

# Start in Hold’em mode
python -m src.realtime.webcam_card_corners --model runs/detect/train*/weights/best.pt --source 0 --start_mode holdem
```

### List cameras / virtual cameras (Windows-friendly)

```bash
python -m src.realtime.webcam_card_corners --list_cams
```

If your OBS/DroidCam virtual camera is running, it should appear as a usable index (often `1`, `2`, `3`, ...).

---

## 📷 Using OBS Virtual Camera (Windows)

1. In OBS, click **Start Virtual Camera**
2. Probe indices:

```bash
python -m src.realtime.webcam_card_corners --list_cams
```

3. Run using the correct index:

```bash
python -m src.realtime.webcam_card_corners --model runs/detect/train*/weights/best.pt --source 1
```

You can also try opening by name (depends on your system/OpenCV backend):

```bash
python -m src.realtime.webcam_card_corners --model runs/detect/train*/weights/best.pt --source "OBS Virtual Camera" --backend dshow
```

---

## 📱 Using a Phone Camera over Wi-Fi (DroidCam / iPhone apps)

Your phone must provide a **stream URL** (HTTP/RTSP). Different apps use different URLs.

Typical patterns (examples only — check your app’s UI):

```bash
# Example HTTP stream (some apps)
python -m src.realtime.webcam_card_corners --model runs/detect/train*/weights/best.pt --source "http://192.168.1.92:4747/video"

# Example RTSP stream (some apps)
python -m src.realtime.webcam_card_corners --model runs/detect/train*/weights/best.pt --source "rtsp://192.168.1.92:8554/live"
```

If you’re using an iPhone, you can use apps that expose an IP camera stream (RTSP/HTTP), or route it through OBS and use **OBS Virtual Camera**.

---

## ♠♥♦♣ Pretty Suit Symbols (Optional)

If your OpenCV supports `cv2.freetype`, PokerVision can render suit symbols.

### Option A (recommended): install OpenCV contrib

```bash
pip install opencv-contrib-python
```

### Option B: provide a font that supports ♠♥♦♣

On Windows, a common option is:

```bash
C:\Windows\Fonts\seguisym.ttf
```

Run:

```bash
python -m src.realtime.webcam_card_corners ^
  --model runs/detect/train*/weights/best.pt ^
  --source 0 ^
  --font "C:\Windows\Fonts\seguisym.ttf"
```

If `cv2.freetype` isn’t available, the script will fall back to plain OpenCV text rendering.

---

## 🧯 Troubleshooting

### “Could not open video source”

Try in this order:

1. Confirm the camera exists:

```bash
python -m src.realtime.webcam_card_corners --list_cams
```

2. Use the index that opens successfully:

```bash
python -m src.realtime.webcam_card_corners --model runs/detect/train*/weights/best.pt --source 1
```

3. For Windows virtual cams, force DirectShow:

```bash
python -m src.realtime.webcam_card_corners --model runs/detect/train*/weights/best.pt --source 1 --backend dshow
```

4. Close other apps that might be locking the camera (Zoom/Teams/Chrome/OBS preview windows, etc.)

### Window creation fails

If you see GUI errors, reinstall non-headless OpenCV:

```bash
pip uninstall -y opencv-python-headless
pip install opencv-python
```

---

## 🪪 License

This project is open-source under the **MIT License**.
See `LICENSE` for details.

```
::contentReference[oaicite:0]{index=0}
```
