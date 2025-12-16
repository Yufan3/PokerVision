# src.realtime.webcam_card_corners

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


# MUST match the LABEL_ORDER you used in dataset generation / cards_corners.yaml
# We'll verify against model.names at runtime and optionally override safely.
LABEL_ORDER = ['TC', 'TD', 'TH', 'TS',
               '2C', '2D', '2H', '2S',
               '3C', '3D', '3H', '3S',
               '4C', '4D', '4H', '4S',
               '5C', '5D', '5H', '5S',
               '6C', '6D', '6H', '6S',
               '7C', '7D', '7H', '7S',
               '8C', '8D', '8H', '8S',
               '9C', '9D', '9H', '9S',
               'AC', 'AD', 'AH', 'AS',
               'JC', 'JD', 'JH', 'JS',
               'KC', 'KD', 'KH', 'KS',
               'QC', 'QD', 'QH', 'QS']

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
# (I am NOT changing your defaults silently; I only add better behavior around them.
#  You can tweak these after you confirm stability.)

IOU_THRESH = 0.30         # slightly stricter than 0.30 to reduce wrong associations
BOX_SMOOTH = 0.50         # EMA weight on old bbox (0.5 old + 0.5 new)
MAX_MISSED_FRAMES = 25    # drop track if unseen for this many frames
DRAW_MAX_MISSED = 6       # still draw tracks missed for <= this many frames
MIN_DRAW_SCORE = 0.55     # do not draw weak / unstable tracks

# class-switch hysteresis: prevent flicker between e.g. 2H / 2D
CLASS_SWITCH_MARGIN = 0.20   # slightly stricter than 0.18
MIN_LOCK_FRAMES    = 6       # slightly stricter than 5

# decay when a track is not matched in the current frame
MISS_SCORE_DECAY = 0.97      # per frame missed: score *= 0.97
MISS_STABLE_DECAY = 1        # per frame missed: stable_frames -= 1 (clamped)


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
        default="runs/detect/train_corners_v1/weights/best.pt",
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
        default=0.40,  # lower detect-threshold; MIN_DRAW_SCORE still prevents junk display
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
        default=80,  # corners are few; reduces overhead
        help="Maximum number of detections per frame.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Process every N-th frame (1 = all frames, 2 = half, etc.).",
    )
    parser.add_argument(
        "--strict_label_check",
        action="store_true",
        help="If set, assert that hardcoded LABEL_ORDER matches model.names exactly.",
    )
    return parser.parse_args()


def convert_source(src_str: str):
    """If source is '0', '1', ... return int for webcam, else string (video path)."""
    if src_str.isdigit():
        return int(src_str)
    return src_str


def get_model_labels(model: YOLO) -> List[str]:
    """Return class labels in index order from Ultralytics YOLO object."""
    names = model.names
    if isinstance(names, dict):
        return [names[i] for i in range(len(names))]
    return list(names)


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
    Update an existing track with a matched detection.
    - Bbox is smoothed (EMA).
    - Class is kept unless new class is clearly better after a stability period.
    """
    # bbox EMA: BOX_SMOOTH is weight on old
    track.bbox = BOX_SMOOTH * track.bbox + (1.0 - BOX_SMOOTH) * det_box

    if det_cls == track.cls_id:
        # same class: strengthen
        if track.age == 0:
            track.score = det_conf
        else:
            track.score = 0.7 * track.score + 0.3 * det_conf
        track.stable_frames += 1
    else:
        # candidate wants to flip
        if (det_conf >= track.score + CLASS_SWITCH_MARGIN) and (track.stable_frames >= MIN_LOCK_FRAMES):
            track.cls_id = det_cls
            track.score = det_conf
            track.stable_frames = 1
        else:
            # reject switch: slight score decay + reduce "stability memory"
            track.score *= 0.95
            track.stable_frames = max(0, track.stable_frames - 1)

    track.age += 1


def cluster_tracks_by_iou(track_list: List[Track], iou_thr: float = 0.5) -> List[Track]:
    """
    Merge overlapping tracks of the same class into "corner instances".
    """
    if not track_list:
        return []

    remaining = sorted(track_list, key=lambda t: -t.score)
    used = set()
    groups: List[List[Track]] = []

    for t in remaining:
        tid = id(t)
        if tid in used:
            continue
        group = [t]
        used.add(tid)
        for u in remaining:
            uid = id(u)
            if uid in used:
                continue
            if iou_xyxy(t.bbox, u.bbox) >= iou_thr:
                group.append(u)
                used.add(uid)
        groups.append(group)

    return [max(g, key=lambda tr: tr.score) for g in groups]


def merge_close_clusters_by_center(cluster_reps: List[Track]) -> List[Track]:
    """
    Second-stage merge to avoid "false 3 clusters" from jittery boxes.
    Threshold is derived from median cluster box size (data-driven per frame).
    """
    if len(cluster_reps) <= 1:
        return cluster_reps

    centers = []
    sizes = []
    for tr in cluster_reps:
        x1, y1, x2, y2 = tr.bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        centers.append((cx, cy))
        sizes.append(0.5 * (w + h))

    med_size = float(np.median(sizes))
    # center distance threshold scaled by box size (not arbitrary pixels)
    dist_thr = max(12.0, 1.25 * med_size)

    kept: List[Track] = []
    kept_centers: List[tuple] = []
    for tr, (cx, cy) in sorted(zip(cluster_reps, centers), key=lambda z: -z[0].score):
        ok = True
        for (kx, ky) in kept_centers:
            if (cx - kx) ** 2 + (cy - ky) ** 2 <= dist_thr ** 2:
                ok = False
                break
        if ok:
            kept.append(tr)
            kept_centers.append((cx, cy))
    return kept


def main():
    global LABEL_ORDER, NUM_CLASSES

    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"[INFO] Loading model from {model_path}")
    model = YOLO(str(model_path))

    # ---- Critical: sync label order from the trained model ----
    model_labels = get_model_labels(model)
    print(f"[INFO] Model nc={len(model_labels)}. First 10 labels: {model_labels[:10]}")

    if args.strict_label_check:
        assert LABEL_ORDER == model_labels, (
            "LABEL_ORDER does not match model.names! "
            "Fix your dataset YAML/names or remove hardcoding and trust model.names."
        )

    # safest: trust the model for mapping ids -> strings
    LABEL_ORDER = model_labels
    NUM_CLASSES = len(LABEL_ORDER)

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
        frame = result.orig_img.copy()

        # robustly handle "no detections" without skipping tracking/drawing
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            xyxy = np.zeros((0, 4), dtype=float)
            cls_ids = np.zeros((0,), dtype=int)
            confs = np.zeros((0,), dtype=float)
        else:
            xyxy = boxes.xyxy.cpu().numpy().astype(float)
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy().astype(float)

        # ---------- 1) Associate detections to tracks (greedy IoU) ----------
        order = np.argsort(-confs) if len(confs) else []
        used_track_ids = set()

        for det_idx in order:
            det_box = xyxy[det_idx]
            det_cls = int(cls_ids[det_idx])
            det_conf = float(confs[det_idx])

            if det_cls < 0 or det_cls >= NUM_CLASSES:
                continue

            best_iou = 0.0
            best_track = None

            # Pass 1: only same-class tracks (most reliable for identity stability)
            for t in tracks:
                if t.track_id in used_track_ids:
                    continue
                if t.cls_id != det_cls:
                    continue
                iou_val = iou_xyxy(det_box, t.bbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_track = t

            # Pass 2: if nothing matched, allow any track (handles true class changes / early unstable tracks)
            if best_track is None:
                for t in tracks:
                    if t.track_id in used_track_ids:
                        continue
                    iou_val = iou_xyxy(det_box, t.bbox)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_track = t


            if best_track is not None and best_iou >= IOU_THRESH:
                update_track_with_detection(best_track, det_box, det_cls, det_conf)
                best_track.last_seen = frame_idx
                used_track_ids.add(best_track.track_id)
            else:
                # new track: IMPORTANT -> age starts at 0 so init branch works
                new_track = Track(
                    track_id=next_track_id,
                    bbox=det_box.copy(),
                    cls_id=det_cls,
                    score=det_conf,
                    last_seen=frame_idx,
                    age=0,
                    stable_frames=1,
                )
                next_track_id += 1
                tracks.append(new_track)
                used_track_ids.add(new_track.track_id)

        # ---------- 1b) Decay tracks not updated this frame ----------
        for t in tracks:
            if t.track_id not in used_track_ids:
                # missed this frame
                t.score *= MISS_SCORE_DECAY
                t.stable_frames = max(0, t.stable_frames - MISS_STABLE_DECAY)

        # ---------- 2) Prune old tracks ----------
        tracks = [t for t in tracks if (frame_idx - t.last_seen) <= MAX_MISSED_FRAMES]

        # ---------- 3) Active tracks per class ----------
        active_tracks = [t for t in tracks if (frame_idx - t.last_seen) <= DRAW_MAX_MISSED]

        per_class_tracks: Dict[int, List[Track]] = defaultdict(list)
        for t in active_tracks:
            if 0 <= t.cls_id < NUM_CLASSES and t.score >= MIN_DRAW_SCORE:
                per_class_tracks[t.cls_id].append(t)

        # ---------- 4) Corner instances per class + duplicate logic ----------
        for cls_id, track_list in per_class_tracks.items():
            label = LABEL_ORDER[cls_id]
            suit = label[-1] if len(label) >= 2 else "?"
            base_color = SUIT_COLORS.get(suit, (255, 255, 255))

            cluster_reps = cluster_tracks_by_iou(track_list, iou_thr=0.5)
            cluster_reps = merge_close_clusters_by_center(cluster_reps)

            is_duplicate = (len(cluster_reps) > 2)

            if not is_duplicate:
                # show only top-left representative (your original behavior)
                def tl_key(tr: Track):
                    x1, y1, x2, y2 = tr.bbox
                    return (y1, x1)

                chosen = min(cluster_reps, key=tl_key)
                reps_to_draw = [(chosen, base_color)]
            else:
                reps_to_draw = [(tr, DUPLICATE_COLOR) for tr in cluster_reps]

            # ---------- 5) Draw ----------
            for tr, color in reps_to_draw:
                x1, y1, x2, y2 = map(int, tr.bbox)
                text = f"{label} {tr.score:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
                (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
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
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cv2.destroyAllWindows()
    print("[INFO] Webcam demo terminated.")


if __name__ == "__main__":
    main()
