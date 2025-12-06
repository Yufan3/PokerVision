import argparse
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.realtime.card_tracker import CardTracker

# --- defaults ---
DEFAULT_MODEL = "runs/segment/train/weights/best.pt"
DEFAULT_SOURCE = 0
DEFAULT_CONF = 0.40
DEFAULT_IMG_SZ = 672
FPS_SMOOTH_ALPHA = 0.1

MASK_ALPHA = 0.35
MASK_COLOR = (255, 0, 0)   # BGR


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    model = YOLO(args.model)
    model.to(device)
    model.fuse()

    # --- tracking state ---
    tracker = CardTracker(
        iou_thresh=0.45,
        max_lost=20,
        smooth_factor=0.25,
    )

    # webcam
    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source {args.source}")

    win_name = "PokerVision – Card segmentation (YOLO11s-seg + Tracking)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    fps_disp = 0.0
    t_prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        overlay = frame.copy()

        # --- YOLO inference ---
        res = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=0.5,
            verbose=False,
            device=0 if device == "cuda" else None,
        )[0]

        polys = []
        boxes = []
        confs = []

        if res.masks is not None and len(res.masks) > 0:
            # res.masks.xy is already in image pixel coordinates
            polys = res.masks.xy
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()

        # --- update tracker (returns only active tracks this frame) ---
        tracks = tracker.update(polys, boxes, confs)

        # --- draw tracked cards ---
        for trk in tracks:
            poly = trk.polygon
            box = trk.bbox_xyxy
            score = trk.stable_score
            age = trk.age
            tid = trk.track_id

            if poly is None or len(poly) < 3:
                continue

            pts = np.round(poly).astype(np.int32)

            # segmentation mask overlay
            cv2.fillPoly(overlay, [pts], MASK_COLOR)

            # bbox
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)

            label = f"ID {tid} | s={score:.2f} | age={age}"
            cv2.putText(
                frame,
                label,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (60, 220, 60),
                2,
                cv2.LINE_AA,
            )

        # blend mask overlay
        frame = cv2.addWeighted(overlay, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0)

        # --- FPS display ---
        t_now = time.time()
        fps = 1.0 / max(1e-6, (t_now - t_prev))
        t_prev = t_now
        fps_disp = fps_disp * (1 - FPS_SMOOTH_ALPHA) + fps * FPS_SMOOTH_ALPHA

        cv2.putText(
            frame,
            f"{fps_disp:.1f} FPS | conf={args.conf:.2f}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):
            break
        elif key == ord("r"):
            args.conf = max(0.05, args.conf - 0.05)
            print(f"[INFO] conf -> {args.conf:.2f}")
        elif key == ord("t"):
            args.conf = min(0.95, args.conf + 0.05)
            print(f"[INFO] conf -> {args.conf:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--source", type=int, default=DEFAULT_SOURCE)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SZ)
    args = ap.parse_args()
    main(args)
