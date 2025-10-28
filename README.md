# 🃏 PokerVision

> **PokerVision** is a computer-vision project that detects and classifies playing cards in real time using a YOLO-based model.  
> It combines **synthetic dataset generation**, **YOLO training**, and **live detection** through a webcam.

---

## 🚀 Features

- 🎨 Synthetic dataset generator with realistic card positioning and lighting
- 🧠 YOLOv8/YOLO11-based card detection and classification
- 💻 Real-time webcam inference at ~30 FPS
- 🃏 Support for normal, inverted, and real card variants
- 🔄 Easy retraining and dataset regeneration

---

## 📂 Project Structure

PokerVision/
├── data/
│ ├── backgrounds/ # Background images for synthetic data
│ ├── generated/ # Auto-generated images (ignored by git)
│ ├── raw_cards/ # Source card images (normal, inverted, real)
│ └── yolo_dataset/ # YOLO-formatted dataset (train/val images & labels)
│
├── runs/ # YOLO training results (ignored by git)
├── src/
│ ├── dataset_gen/ # Dataset generation scripts
│ ├── realtime/ # Real-time detection (webcam_demo.py)
│ ├── training/ # (Future) model training utils
│ └── utils/ # Shared helper functions
│
├── weights/ # Model checkpoints (ignored by git)
├── cards.yaml # YOLO dataset config
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md

yaml
Copy code

---

## 🧩 Installation

### 1️⃣ Clone the repo
```bash
git clone https://github.com/Yufan3/PokerVision.git
cd PokerVision
2️⃣ Create a Conda environment
bash
Copy code
conda create -n pokervision python=3.10 -y
conda activate pokervision
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Verify GPU setup
bash
Copy code
python -m src.utils.check_cuda
🧠 Dataset Generation
Generate a synthetic dataset by combining random backgrounds and card variants:

bash
Copy code
python -m src.dataset_gen.generate_dataset
The generated data and YOLO labels will appear under:

bash
Copy code
data/generated/
data/yolo_dataset/
🏋️‍♂️ Model Training
Train a YOLO model using your dataset:

bash
Copy code
yolo detect train model=yolo11s.pt data=cards.yaml epochs=100 imgsz=640 batch=16 device=0
After training, the best weights will be saved at:

bash
Copy code
runs/detect/train*/weights/best.pt
🎥 Real-Time Detection
Run live card detection using your webcam:

bash
Copy code
python src/realtime/webcam_demo.py
Press Q to exit the window.

🪪 License

This project is open-source under the MIT License.
See LICENSE for details.
