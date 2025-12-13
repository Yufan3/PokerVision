# src/realtime/webcam_card_corners.py

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


# MUST match the LABEL_ORDER you used in dataset generation / cards_corners.yaml
LABEL_ORDER = [
    "2C", "2D", "2H", "2S",
    "3C", "3D", "3H", "3S",
    "4C", "4D", "4H", "4S",
    "5C", "5D", "5H", "5S",
    "6C", "6D", "6H", "6S",
    "7C", "7D", "7H", "7S",
    "8C", "8D", "8H", "8S",
    "9C", "9D", "9H", "9S",
    "TC", "TD", "TH", "TS",
    "JC", "JD", "JH", "JS",
    "QC", "QD", "QH", "QS",
    "KC", "KD", "KH", "KS",
    "AC", "AD", "AH", "AS",
]
NUM_CLASSES = len(LABEL_ORDER)

# BGR colors for suits (visually pleasant)
SUIT_COLORS = {
    "H": (54, 38, 227),     # red-ish (BGR)
    "S": (8, 12, 16),       # near black
    "C": (131, 193, 157),   # green
    "D": (187, 57, 28),     # blue-ish
}

# Duplicate (impossible) card color – light purple
DUPLICATE_COLOR = (204, 102, 153)  # BGR


# ------------ Tracking & hysteresis parameters ------------

IOU_THRESH = 0.3          # match detection to existing track if IoU >= this
BOX_SMOOTH = 0.3          # EMA for bbox: new = 0.3*old + 0.7*det
MAX_MISSED_FRAMES = 25    # drop track if unseen for this many frames
DRAW_MAX_MISSED = 3       # still draw tracks missed for <= this many frames
MIN_DRAW_SCORE = 0.65     # do not draw weak / unstable tracks

# class-switch hysteresis: prevent flicker between e.g. 2H / 2D
CLASS_SWITCH_MARGIN = 0.18   # new class must beat locked one by at least this
MIN_LOCK_FRAMES    = 5       # need this many frames before allowing switch


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray      # [x1, y1, x2, y2] (float)
    cls_id: int           # locked class index for this track
    score: float          # smoothed confidence for cls_id
    last_seen: int
    age: int = 0
    stable_frames: int = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Webcam demo for YOLOv11 corner detector with temporal smoothing and duplicate highlighting."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/train_corners_v2/weights/best.pt",
        help="Path to trained YOLOv11 detection model (.pt).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: '0' for default webcam, or path to video file.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.72,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=672,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device for inference, e.g. '0', 'cpu'.",
    )
    parser.add_argument(
        "--max_det",
        type=int,
        default=120,
        help="Maximum number of detections per frame.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Process every N-th frame (1 = all frames, 2 = half, etc.).",
    )
    return parser.parse_args()


def convert_source(src_str: str):
    """If source is '0', '1', ... return int for webcam, else string (video path)."""
    if src_str.isdigit():
        return int(src_str)
    return src_str


def iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def update_track_with_detection(track: Track, det_box: np.ndarray, det_cls: int, det_conf: float):
    """
    Update an existing track with a matched detection (one per frame).
    - Bbox is smoothed.
    - Class is kept unless the new class is clearly better for a stable track.
    """
    # 1) bbox EMA
    track.bbox = BOX_SMOOTH * track.bbox + (1.0 - BOX_SMOOTH) * det_box

    # 2) class / score hysteresis
    if det_cls == track.cls_id:
        # same class -> strengthen it
        if track.age == 0:
            track.score = det_conf
        else:
            track.score = 0.7 * track.score + 0.3 * det_conf
        track.stable_frames += 1
    else:
        # candidate wants to flip the class
        if (det_conf >= track.score + CLASS_SWITCH_MARGIN
                and track.stable_frames >= MIN_LOCK_FRAMES):
            # accept switch
            track.cls_id = det_cls
            track.score = det_conf
            track.stable_frames = 1
        else:
            # reject switch, keep current class and let it decay slightly
            track.score = 0.95 * track.score

    track.age += 1


def cluster_tracks_by_iou(track_list: List[Track], iou_thr: float = 0.5) -> List[Track]:
    """
    Merge overlapping tracks of the same class into "corner instances".
    This removes double-detections of the same corner before we count
    how many corners a logical card has.
    """
    if not track_list:
        return []

    # Greedy grouping by IOU, strongest first
    remaining = sorted(track_list, key=lambda t: -t.score)
    used_ids = set()
    groups: List[List[Track]] = []

    for t in remaining:
        tid = id(t)
        if tid in used_ids:
            continue
        group = [t]
        used_ids.add(tid)
        for u in remaining:
            uid = id(u)
            if uid in used_ids:
                continue
            if iou_xyxy(t.bbox, u.bbox) >= iou_thr:
                group.append(u)
                used_ids.add(uid)
        groups.append(group)

    # Representative of each group = highest-score track in that group
    reps = [max(g, key=lambda tr: tr.score) for g in groups]
    return reps


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"[INFO] Loading model from {model_path}")
    model = YOLO(str(model_path))

    source = convert_source(args.source)
    print(f"[INFO] Starting stream from source={source!r}")

    results_generator = model(
        source=source,
        stream=True,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        verbose=False,
        max_det=args.max_det,
        vid_stride=args.frame_stride,
    )

    win_name = "PokerVision - Corners (suit-colored, purple = duplicate card class)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    tracks: List[Track] = []
    next_track_id = 0
    frame_idx = 0

    for result in results_generator:
        frame_idx += 1
        frame = result.orig_img.copy()  # BGR uint8

        # No detections
        if (not hasattr(result, "boxes")
                or result.boxes is None
                or result.boxes.xyxy is None
                or len(result.boxes.xyxy) == 0):
            cv2.imshow(win_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()                # (N,4)
        cls_ids = boxes.cls.cpu().numpy().astype(int)  # (N,)
        confs = boxes.conf.cpu().numpy()               # (N,)

        # ---------- 1) Associate detections to existing tracks (greedy IoU) ----------
        order = np.argsort(-confs)  # high conf first
        used_track_ids = set()

        for det_idx in order:
            det_box = xyxy[det_idx].astype(float)
            det_cls = int(cls_ids[det_idx])
            det_conf = float(confs[det_idx])

            # sanity check
            if det_cls < 0 or det_cls >= NUM_CLASSES:
                continue

            # find best matching track
            best_iou = 0.0
            best_track = None
            for t in tracks:
                if t.track_id in used_track_ids:
                    continue
                iou_val = iou_xyxy(det_box, t.bbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_track = t

            if best_track is not None and best_iou >= IOU_THRESH:
                # update existing track
                update_track_with_detection(best_track, det_box, det_cls, det_conf)
                best_track.last_seen = frame_idx
                used_track_ids.add(best_track.track_id)
            else:
                # create new track
                new_track = Track(
                    track_id=next_track_id,
                    bbox=det_box.copy(),
                    cls_id=det_cls,
                    score=det_conf,
                    last_seen=frame_idx,
                    age=1,
                    stable_frames=1,
                )
                next_track_id += 1
                tracks.append(new_track)
                used_track_ids.add(new_track.track_id)

        # ---------- 2) Prune old tracks ----------
        tracks = [t for t in tracks if (frame_idx - t.last_seen) <= MAX_MISSED_FRAMES]

        # ---------- 3) Group active tracks per class ----------
        active_tracks = [
            t for t in tracks
            if (frame_idx - t.last_seen) <= DRAW_MAX_MISSED
        ]

        per_class_tracks: Dict[int, List[Track]] = defaultdict(list)
        for t in active_tracks:
            if t.cls_id < 0 or t.cls_id >= NUM_CLASSES:
                continue
            if t.score < MIN_DRAW_SCORE:
                continue
            per_class_tracks[t.cls_id].append(t)

        # ---------- 4) Corner-detections → card instances (per class) ----------
        for cls_id, track_list in per_class_tracks.items():
            label = LABEL_ORDER[cls_id]
            suit = label[-1] if len(label) >= 2 else "?"
            base_color = SUIT_COLORS.get(suit, (255, 255, 255))

            # 4a) merge overlapping tracks of this class into "corner instances"
            cluster_reps = cluster_tracks_by_iou(track_list, iou_thr=0.5)

            # duplicate logic based on number of corner instances
            is_duplicate = (len(cluster_reps) > 2)

            if not is_duplicate:
                # ≤ 2 corners total for this logical card:
                # show **only the top-left** representative
                def tl_key(tr: Track):
                    x1, y1, x2, y2 = tr.bbox
                    return (y1, x1)

                chosen = min(cluster_reps, key=tl_key)
                reps_to_draw = [(chosen, base_color)]
            else:
                # 3+ distinct corners → impossible duplicate (or model hallucination)
                reps_to_draw = [(tr, DUPLICATE_COLOR) for tr in cluster_reps]

            # ---------- 5) Draw selected reps ----------
            for tr, color in reps_to_draw:
                x1, y1, x2, y2 = map(int, tr.bbox)
                text = f"{label} {tr.score:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
                (tw, th), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                text_bg_tl = (x1, max(0, y1 - th - baseline - 2))
                text_bg_br = (x1 + tw + 4, y1)
                cv2.rectangle(frame, text_bg_tl, text_bg_br, color, thickness=-1)
                cv2.putText(
                    frame,
                    text,
                    (text_bg_tl[0] + 2, y1 - baseline - 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    print("[INFO] Webcam demo terminated.")


if __name__ == "__main__":
    main()
