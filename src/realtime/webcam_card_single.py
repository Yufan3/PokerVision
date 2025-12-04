import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# --- defaults (can be overridden by CLI) ---
DEFAULT_MODEL = "runs/detect/train/weights/best.pt" # single-class 'card' detector
DEFAULT_SOURCE = 0
DEFAULT_CONF = 0.25
DEFAULT_IMG_SZ = 704          # ↑ slightly bigger improves small/partial cards
IOU_MATCH_THRESHOLD = 0.40    # your tracker threshold (kept)
SMOOTH_FRAMES = 15            # your temporal smoothing (kept)
FPS_SMOOTH_ALPHA = 0.1

def refine_bbox_with_edges(frame, box, pad=8):
    # box: [x1,y1,x2,y2], returns (ok, refined_box)
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W-1, x2 + pad); y2 = min(H-1, y2 + pad)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False, box

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gray, 50, 150)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, box

    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 0.02 * roi.shape[0] * roi.shape[1]:
        return False, box

    rect = cv2.minAreaRect(c)
    (cx, cy), (w, h), ang = rect
    w, h = max(w,1), max(h,1)
    ratio = max(w, h) / max(1.0, min(w, h))
    if ratio < 1.2 or ratio > 1.8:
        return False, box  # not card-like

    box_pts = cv2.boxPoints(rect)
    box_pts[:,0] += x1; box_pts[:,1] += y1
    xA, yA = np.min(box_pts, axis=0).astype(int)
    xB, yB = np.max(box_pts, axis=0).astype(int)
    xA = max(0, xA); yA = max(0, yA); xB = min(W-1, xB); yB = min(H-1, yB)
    if xB <= xA or yB <= yA:
        return False, box
    return True, np.array([xA, yA, xB, yB], dtype=float)

def iou_xyxy(a, b):
    # boxes: [x1,y1,x2,y2]
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    areaB = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    denom = areaA + areaB - inter
    return inter / denom if denom > 0 else 0.0

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)
    model.to(device)
    model.fuse()

    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)  # force DirectShow on Windows
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source {args.source}")

    # simple IOU-based tracker with temporal smoothing
    tracks = {}           # tid -> dict(bbox, conf_hist)
    next_id = 0
    fps_disp = 0.0
    t_prev = time.time()

    def match_tracks(dets):
        nonlocal tracks, next_id
        used = set()
        out = []
        # dets: list[(box, conf)]
        for box, conf in dets:
            best_i, best_id = 0.0, None
            for tid, tr in list(tracks.items()):
                i = iou_xyxy(box, tr["bbox"])
                if i > best_i and i > IOU_MATCH_THRESHOLD:
                    best_i, best_id = i, tid
            if best_id is not None:
                tr = tracks[best_id]
                tr["bbox"] = box
                tr["conf_hist"].append(conf)
                out.append((best_id, box))
                used.add(best_id)
            else:
                tid = next_id
                next_id += 1
                tracks[tid] = {"bbox": box, "conf_hist": deque([conf], maxlen=SMOOTH_FRAMES)}
                out.append((tid, box))
                used.add(tid)
        # prune unmatched
        tracks = {tid: tr for tid, tr in tracks.items() if tid in used}
        return out

    win_name = "PokerVision – Card detector (single-class)"
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Inference tuned for overlaps/fans: higher NMS IoU, larger imgsz, more max_det
        res = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=0.60,           # ↑ allow overlapping boxes to coexist
            max_det=50,        # ↑ crowded scenes
            verbose=False,
            device=0 if device == "cuda" else None,
        )[0]

        # collect detections: (box, conf) in xyxy
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for box, c in zip(boxes, confs):
                dets.append((box, float(c)))

        refined = []
        for box, c in dets:
            ok, tight = refine_bbox_with_edges(frame, box)
            refined.append((tight if ok else box, c))
        dets = refined


        matched = match_tracks(dets)

        # draw
        for tid, box in matched:
            x1, y1, x2, y2 = [int(v) for v in box]
            conf_avg = float(np.mean(tracks[tid]["conf_hist"])) if tracks[tid]["conf_hist"] else 0.0
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)
            cv2.putText(frame, f"card {conf_avg:.2f}", (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 60), 2, cv2.LINE_AA)

        # fps
        t_now = time.time()
        fps = 1.0 / max(1e-6, (t_now - t_prev))
        t_prev = t_now
        fps_disp = fps_disp * (1 - FPS_SMOOTH_ALPHA) + fps * FPS_SMOOTH_ALPHA
        cv2.putText(frame, f"{fps_disp:.1f} FPS", (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 220, 60), 2, cv2.LINE_AA)

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):   # ESC or q to quit
            break
        elif key == ord('r'):       # r: lower conf (more recall)
            args.conf = max(0.05, args.conf - 0.05)
            print(f"[INFO] conf -> {args.conf:.2f}")
        elif key == ord('t'):       # t: raise conf (less false positives)
            args.conf = min(0.90, args.conf + 0.05)
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

