import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque
import time

# --- Parameters ---
MODEL_PATH = "runs/detect/train2/weights/best.pt"
CONF_THRESHOLD = 0.5       # try 0.4–0.6
IOU_MATCH_THRESHOLD = 0.4  # how similar boxes must be to be the same object
SMOOTH_FRAMES = 5          # how many past predictions to remember
FPS_SMOOTH = 15            # FPS smoothing for display

# --- Load model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)
model.to(device)
model.fuse()

# --- Tracker state ---
tracks = {}  # id -> dict(bbox, label_hist, conf_hist)
next_id = 0

def iou(boxA, boxB):
    # boxes in [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

def match_tracks(dets):
    """Assign detections to existing tracks using IoU."""
    global next_id, tracks
    used_ids = set()
    results = []

    for det in dets:
        box, cls, conf = det
        best_iou, best_id = 0, None
        for tid, tr in tracks.items():
            i = iou(box, tr["bbox"])
            if i > best_iou and i > IOU_MATCH_THRESHOLD:
                best_iou, best_id = i, tid

        if best_id is not None:
            used_ids.add(best_id)
            tr = tracks[best_id]
            tr["bbox"] = box
            tr["label_hist"].append(cls)
            tr["conf_hist"].append(conf)
            if len(tr["label_hist"]) > SMOOTH_FRAMES:
                tr["label_hist"].popleft()
                tr["conf_hist"].popleft()
            results.append((best_id, box))
        else:
            # new track
            tid = next_id
            next_id += 1
            tracks[tid] = {
                "bbox": box,
                "label_hist": deque([cls], maxlen=SMOOTH_FRAMES),
                "conf_hist": deque([conf], maxlen=SMOOTH_FRAMES)
            }
            used_ids.add(tid)
            results.append((tid, box))

    # remove lost tracks (not matched for a while)
    tracks = {tid: tr for tid, tr in tracks.items() if tid in used_ids}
    return results

def get_smooth_label(tid):
    tr = tracks[tid]
    if not tr["label_hist"]:
        return "?"
    labels = list(tr["label_hist"])
    confs = list(tr["conf_hist"])
    # majority vote weighted by avg conf
    label = max(set(labels), key=labels.count)
    conf_avg = np.mean(confs)
    return label, conf_avg

# --- Webcam loop ---
cap = cv2.VideoCapture(0)
fps_display = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # inference
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]

    dets = []
    for box, cls, conf in zip(results.boxes.xyxy.cpu().numpy(),
                              results.boxes.cls.cpu().numpy(),
                              results.boxes.conf.cpu().numpy()):
        label = model.names[int(cls)]
        dets.append((box, label, conf))

    matched = match_tracks(dets)

    # draw boxes
    for tid, box in matched:
        label, conf_avg = get_smooth_label(tid)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf_avg:.2f}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

    # FPS counter
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now
    fps_display = 0.9 * fps_display + 0.1 * fps if fps_display else fps
    cv2.putText(frame, f"{fps_display:.1f} FPS",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("PokerVision Live", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
