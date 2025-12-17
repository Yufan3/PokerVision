# src/realtime/webcam_card_corners.py
# python -m src.realtime.webcam_card_corners --source 2 --backend dshow --model runs/detect/train3/weights/best.pt
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math
import random
import time
import re
import sys

import cv2
import numpy as np
from ultralytics import YOLO

# Poker evaluation
from treys import Card, Evaluator


# ----------------------------
# Camera / capture utilities
# ----------------------------

def _is_url(s: str) -> bool:
    s = s.strip().lower()
    return s.startswith("http://") or s.startswith("https://") or s.startswith("rtsp://")


def _is_probably_windows_path(s: str) -> bool:
    # C:\... or \\server\... or contains drive-like pattern
    return bool(re.match(r"^[a-zA-Z]:\\", s)) or s.startswith("\\\\")


def _backend_from_str(name: str) -> int:
    name = name.lower()
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "msmf":
        return cv2.CAP_MSMF
    if name == "any":
        return cv2.CAP_ANY
    return cv2.CAP_ANY


def get_dshow_device_names() -> Optional[List[str]]:
    """
    Optional: if pygrabber is installed, we can list DirectShow device names.
    pip install pygrabber
    """
    if not sys.platform.startswith("win"):
        return None
    try:
        from pygrabber.dshow_graph import FilterGraph  # type: ignore
        return FilterGraph().get_input_devices()
    except Exception:
        return None


def list_available_cameras(max_index: int = 15, backend: str = "auto"):
    """
    Prints which camera indices OpenCV can open, and (optionally) DirectShow device names.
    """
    print("[INFO] Probing camera indices...")

    if sys.platform.startswith("win"):
        names = get_dshow_device_names()
        if names:
            print("[INFO] DirectShow device names (pygrabber):")
            for i, n in enumerate(names):
                print(f"  dshow_list_index={i}: {n}")
        else:
            print("[INFO] (Tip) Install 'pygrabber' to list device names: pip install pygrabber")

    # choose probing backends
    if sys.platform.startswith("win"):
        backend_order = []
        if backend == "auto":
            backend_order = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]
        else:
            b = _backend_from_str(backend)
            backend_order = [(backend.upper(), b)]
    else:
        backend_order = [("ANY", cv2.CAP_ANY)]

    found_any = False
    for i in range(max_index):
        opened = False
        for bname, b in backend_order:
            cap = None
            try:
                cap = cv2.VideoCapture(i, b)
                if cap.isOpened():
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        h, w = frame.shape[:2]
                        print(f"  index={i} opened ✅  {w}x{h}   backend={bname}")
                        found_any = True
                        opened = True
                        break
            except Exception:
                # Some backends throw noisy exceptions; ignore and continue probing.
                pass
            finally:
                if cap is not None:
                    cap.release()
        if opened:
            continue

    if not found_any:
        print("  (No cameras opened. Check permissions / drivers / virtual cam install.)")


def convert_source(src_str: str):
    """If source is '0', '1', ... return int for webcam, else keep string."""
    if src_str.isdigit():
        return int(src_str)
    return src_str


def open_capture(source, backend: str = "auto") -> cv2.VideoCapture:
    """
    Robust open for:
      - int index
      - Windows DirectShow name: "video=OBS Virtual Camera"
      - Windows plain name: "OBS Virtual Camera" (we try adding video=)
      - URL streams (http/rtsp)
      - file paths
    """
    # backend order
    if sys.platform.startswith("win"):
        if backend == "auto":
            backend_order = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backend_order = [_backend_from_str(backend)]
    else:
        backend_order = [cv2.CAP_ANY] if backend == "auto" else [_backend_from_str(backend)]

    def try_open(src, b) -> Optional[cv2.VideoCapture]:
        cap = None
        try:
            cap = cv2.VideoCapture(src, b)
            if cap.isOpened():
                # reduce latency when supported
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                return cap
        except Exception:
            pass
        if cap is not None:
            cap.release()
        return None

    # ---- Case A: index ----
    if isinstance(source, int):
        for b in backend_order:
            cap = try_open(source, b)
            if cap is not None:
                return cap
        raise RuntimeError(f"Could not open camera index {source} with backends {backend_order}")

    # ---- Case B: string ----
    src = str(source).strip()

    # URLs / files: use normal open first (FFMPEG often kicks in automatically)
    if _is_url(src) or Path(src).exists() or _is_probably_windows_path(src):
        for b in backend_order:
            cap = try_open(src, b)
            if cap is not None:
                return cap
        raise RuntimeError(f"Could not open video source: {src!r}")

    # Windows virtual cams by name (OBS / DroidCam virtual driver)
    if sys.platform.startswith("win"):
        candidates = []
        if src.lower().startswith("video="):
            candidates.append(src)  # already dshow format
            candidates.append(src[len("video="):].strip())  # sometimes works too
        else:
            candidates.append(src)
            candidates.append(f"video={src}")

        # Try open by string on each backend (DSHOW first in auto)
        for cand in candidates:
            for b in backend_order:
                cap = try_open(cand, b)
                if cap is not None:
                    return cap

        # Optional: if pygrabber is installed, try to map name -> index
        names = get_dshow_device_names()
        if names:
            want = src
            if want.lower().startswith("video="):
                want = want[len("video="):].strip()

            # substring match
            match_idx = None
            for i, n in enumerate(names):
                if want.lower() == n.lower():
                    match_idx = i
                    break
            if match_idx is None:
                for i, n in enumerate(names):
                    if want.lower() in n.lower():
                        match_idx = i
                        break

            if match_idx is not None:
                for b in backend_order:
                    cap = try_open(match_idx, b)
                    if cap is not None:
                        print(f"[INFO] Opened by name-match via pygrabber: {names[match_idx]!r} -> index {match_idx}")
                        return cap

        raise RuntimeError(
            f"Could not open video source: {src!r}\n"
            f"Tips:\n"
            f"  - In OBS: click 'Start Virtual Camera' first.\n"
            f"  - Run: python -m src.realtime.webcam_card_corners --list_cams\n"
            f"    Then use --source <index> (e.g. 1,2,3...).\n"
            f"  - Or try: --source \"video=OBS Virtual Camera\" --backend dshow\n"
        )

    # Non-Windows: last resort
    for b in backend_order:
        cap = try_open(src, b)
        if cap is not None:
            return cap
    raise RuntimeError(f"Could not open video source: {src!r}")


# ----------------------------
# Labels (will be overridden by model.names safely)
# ----------------------------
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

DUPLICATE_COLOR = (204, 102, 153)  # BGR

# ------------ Tracking & hysteresis parameters ------------
IOU_THRESH = 0.30
BOX_SMOOTH = 0.50
MAX_MISSED_FRAMES = 25
DRAW_MAX_MISSED = 6
MIN_DRAW_SCORE = 0.45

CLASS_SWITCH_MARGIN = 0.20
MIN_LOCK_FRAMES = 6

MISS_SCORE_DECAY = 0.97
MISS_STABLE_DECAY = 1

SWITCH_CONFIRM_FRAMES = 3

# ------------ Snapshot / poker-state stabilization ------------
EVAL_MIN_TRACK_STABLE_FRAMES = 8
EVAL_MIN_CARD_SCORE = 0.60
EVAL_STABLE_SET_FRAMES = 10
EVAL_SET_JACCARD_MIN = 0.95

DEFAULT_SIMS = 5000


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    cls_id: int
    score: float
    last_seen: int
    age: int = 0
    stable_frames: int = 0
    cand_cls: int = -1
    cand_count: int = 0


@dataclass
class CardInstance:
    label: str
    cls_id: int
    bbox: np.ndarray
    score: float
    center: Tuple[float, float]


@dataclass
class Cluster:
    cards: List[CardInstance]
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    role: str


# ----------------------------
# Pretty suit rendering (♠♥♦♣) + colored tokens
# ----------------------------
SUIT_TO_SYMBOL = {"H": "♥", "S": "♠", "C": "♣", "D": "♦"}

SUIT_COLORS_SYMBOL = {
    "♥": (54, 38, 227),
    "♠": (8, 12, 16),
    "♣": (131, 193, 157),
    "♦": (187, 57, 28),
}

SUIT_COLORS_LETTER = {
    "H": SUIT_COLORS_SYMBOL["♥"],
    "S": SUIT_COLORS_SYMBOL["♠"],
    "C": SUIT_COLORS_SYMBOL["♣"],
    "D": SUIT_COLORS_SYMBOL["♦"],
}


def pretty_label(label: str, unicode_ok: bool) -> Tuple[str, Tuple[int, int, int]]:
    if len(label) != 2:
        return label, (255, 255, 255)
    r, s = label[0], label[1].upper()
    sym = SUIT_TO_SYMBOL.get(s, s)
    if unicode_ok:
        col = SUIT_COLORS_SYMBOL.get(sym, (255, 255, 255))
        return f"{r}{sym}", col
    col = SUIT_COLORS_LETTER.get(s, (255, 255, 255))
    return f"{r}{s}", col


class TextRenderer:
    def __init__(self, font_path: Optional[str] = None):
        self.ft = None
        self.unicode_ok = False
        if hasattr(cv2, "freetype"):
            try:
                ft = cv2.freetype.createFreeType2()
                if font_path is not None and Path(font_path).exists():
                    ft.loadFontData(fontFileName=str(font_path), id=0)
                    self.ft = ft
                    self.unicode_ok = True
            except Exception:
                self.ft = None
                self.unicode_ok = False

    def get_text_size(self, text: str, font_height: int, thickness: int) -> Tuple[int, int, int]:
        if self.ft is not None:
            (w, h), baseline = self.ft.getTextSize(text, fontHeight=font_height, thickness=thickness)
            return int(w), int(h), int(baseline)
        (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_height / 30.0, thickness)
        return int(w), int(h), int(baseline)

    def put_text(self, img, text: str, org: Tuple[int, int], font_height: int,
                 color: Tuple[int, int, int], thickness: int = 2):
        x, y = org
        if self.ft is not None:
            self.ft.putText(img, text, (x, y), fontHeight=font_height,
                            color=color, thickness=thickness, line_type=cv2.LINE_AA, bottomLeftOrigin=False)
        else:
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_height / 30.0, color, thickness, cv2.LINE_AA)


def draw_tag(frame, tr: TextRenderer, x: int, y: int, text: str,
             text_color: Tuple[int, int, int], font_h: int = 22, thickness: int = 2,
             bg_color: Tuple[int, int, int] = (0, 0, 0), pad: int = 4):
    
    # 1. Calculate text size
    tw, th, bl = tr.get_text_size(text, font_height=font_h, thickness=thickness)
    x2 = x + tw + 2 * pad
    y2 = y + th + bl + 2 * pad

    # 2. REMOVED: The background rectangle line
    # cv2.rectangle(frame, (x, y), (x2, y2), bg_color, -1)

    # 3. ADDED: Draw a thick black "stroke" (outline) behind the text
    # We draw the text in black with extra thickness first
    # tr.put_text(frame, text, (x + pad, y + th + pad), font_height=font_h,
    #             color=(0, 0, 0), thickness=thickness + 3)

    # 4. Draw the actual colored text on top
    tr.put_text(frame, text, (x + pad, y + th + pad), font_height=font_h,
                color=text_color, thickness=thickness)

    return x2, y2


def draw_card_box(frame, tr: TextRenderer, ci: CardInstance, show_score: bool = True):
    txt, col = pretty_label(ci.label, unicode_ok=tr.unicode_ok)
    label = f"{txt} {ci.score:.2f}" if show_score else txt

    x1, y1, x2, y2 = map(int, ci.bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

    tag_x = x1
    tag_y = max(0, y1 - 30)
    draw_tag(frame, tr, tag_x, tag_y, label, text_color=col, font_h=22, thickness=2)


def draw_token_line(frame, tr: TextRenderer, tokens: List[str], org: Tuple[int, int],
                    font_h: int = 24, thickness: int = 2, pad: int = 12):
    x, y = org
    for raw in tokens:
        txt, col = pretty_label(raw, unicode_ok=tr.unicode_ok)
        tr.put_text(frame, txt, (x, y), font_height=font_h, color=col, thickness=thickness)
        tw, _, _ = tr.get_text_size(txt, font_height=font_h, thickness=thickness)
        x += tw + pad


# ----------------------------
# Model helpers
# ----------------------------
def get_model_labels(model: YOLO) -> List[str]:
    names = model.names
    if isinstance(names, dict):
        return [names[i] for i in range(len(names))]
    return list(names)


def iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> float:
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
    track.bbox = BOX_SMOOTH * track.bbox + (1.0 - BOX_SMOOTH) * det_box

    if det_cls == track.cls_id:
        track.score = det_conf if track.age == 0 else (0.7 * track.score + 0.3 * det_conf)
        track.stable_frames += 1
        track.cand_cls = -1
        track.cand_count = 0
    else:
        if track.cand_cls != det_cls:
            track.cand_cls = det_cls
            track.cand_count = 1
        else:
            track.cand_count += 1

        if (track.stable_frames >= MIN_LOCK_FRAMES
                and track.cand_count >= SWITCH_CONFIRM_FRAMES
                and det_conf >= track.score + CLASS_SWITCH_MARGIN):
            track.cls_id = det_cls
            track.score = det_conf
            track.stable_frames = 1
            track.cand_cls = -1
            track.cand_count = 0
        else:
            track.score *= 0.95
            track.stable_frames = max(0, track.stable_frames - 1)

    track.age += 1


def cluster_tracks_by_iou(track_list: List[Track], iou_thr: float = 0.5) -> List[Track]:
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
    if len(cluster_reps) <= 1:
        return cluster_reps

    centers, sizes = [], []
    for tr in cluster_reps:
        x1, y1, x2, y2 = tr.bbox
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        centers.append((cx, cy))
        sizes.append(0.5 * (w + h))

    med_size = float(np.median(sizes))
    dist_thr = max(12.0, 1.25 * med_size)

    kept: List[Track] = []
    kept_centers: List[Tuple[float, float]] = []
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


def extract_card_instances_from_tracks(
    tracks: List[Track],
    frame_idx: int,
    num_classes: int,
    labels: List[str],
) -> Tuple[List[CardInstance], List[CardInstance]]:
    active_tracks = [t for t in tracks if (frame_idx - t.last_seen) <= DRAW_MAX_MISSED]

    per_class: Dict[int, List[Track]] = defaultdict(list)
    for t in active_tracks:
        if 0 <= t.cls_id < num_classes and t.score >= MIN_DRAW_SCORE:
            per_class[t.cls_id].append(t)

    draw_cards: List[CardInstance] = []
    eval_cards: List[CardInstance] = []

    for cls_id, tlist in per_class.items():
        reps = cluster_tracks_by_iou(tlist, iou_thr=0.5)
        reps = merge_close_clusters_by_center(reps)

        if len(reps) > 2:
            for tr in reps:
                x1, y1, x2, y2 = tr.bbox
                cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                draw_cards.append(CardInstance(labels[cls_id], cls_id, tr.bbox.copy(), float(tr.score), (cx, cy)))
            continue

        def tl_key(tr: Track):
            x1, y1, x2, y2 = tr.bbox
            return (y1, x1)

        chosen = min(reps, key=tl_key)
        x1, y1, x2, y2 = chosen.bbox
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        ci = CardInstance(labels[cls_id], cls_id, chosen.bbox.copy(), float(chosen.score), (cx, cy))
        draw_cards.append(ci)

        if chosen.stable_frames >= EVAL_MIN_TRACK_STABLE_FRAMES and chosen.score >= EVAL_MIN_CARD_SCORE:
            eval_cards.append(ci)

    return draw_cards, eval_cards


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def union_find_clusters(points: List[Tuple[float, float]], dist_thr: float) -> List[List[int]]:
    n = len(points)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    d2 = dist_thr * dist_thr
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            if (xi - xj) ** 2 + (yi - yj) ** 2 <= d2:
                union(i, j)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def cluster_cards_into_groups(cards: List[CardInstance], frame_shape: Tuple[int, int, int], cluster_scale: float) -> List[List[CardInstance]]:
    if not cards:
        return []

    centers = [c.center for c in cards]
    sizes = []
    for c in cards:
        x1, y1, x2, y2 = c.bbox
        sizes.append(0.5 * (max(1, x2 - x1) + max(1, y2 - y1)))
    med = float(np.median(sizes))
    dist_thr = max(20.0, cluster_scale * med)

    idx_groups = union_find_clusters(centers, dist_thr=dist_thr)
    grouped = [[cards[i] for i in idxs] for idxs in idx_groups]
    grouped.sort(key=lambda g: (-len(g), float(np.mean([c.score for c in g]))))
    return grouped


def pick_board_and_hands(grouped: List[List[CardInstance]], frame_shape: Tuple[int, int, int]) -> Tuple[List[CardInstance], List[List[CardInstance]]]:
    if not grouped:
        return [], []

    H, W = frame_shape[:2]
    cx0, cy0 = W / 2.0, H / 2.0

    def cl_center(cluster):
        xs = [c.center[0] for c in cluster]
        ys = [c.center[1] for c in cluster]
        return (float(np.mean(xs)), float(np.mean(ys)))

    board_candidates = []
    for g in grouped:
        if len(g) >= 3:
            cxc, cyc = cl_center(g)
            dist = (cxc - cx0) ** 2 + (cyc - cy0) ** 2
            board_candidates.append((len(g), -dist, g))

    if board_candidates:
        board_candidates.sort(reverse=True)
        board = board_candidates[0][2]
        remaining = [g for g in grouped if g is not board]
    else:
        board = []
        remaining = grouped[:]

    hands = [g for g in remaining if 2 <= len(g) <= 4]
    hands = hands[:5]
    return board, hands


def cluster_bbox(cluster: List[CardInstance]) -> Tuple[int, int, int, int]:
    x1 = int(min(c.bbox[0] for c in cluster))
    y1 = int(min(c.bbox[1] for c in cluster))
    x2 = int(max(c.bbox[2] for c in cluster))
    y2 = int(max(c.bbox[3] for c in cluster))
    return x1, y1, x2, y2


def cluster_center(cluster: List[CardInstance]) -> Tuple[float, float]:
    xs = [c.center[0] for c in cluster]
    ys = [c.center[1] for c in cluster]
    return float(np.mean(xs)), float(np.mean(ys))


def label_to_treys(label: str) -> int:
    if len(label) != 2:
        raise ValueError(f"Bad label: {label}")
    r = label[0]
    s = label[1].lower()
    return Card.new(r + s)


def compute_equities(board_labels: List[str], hands_labels: List[List[str]], sims: int) -> Optional[List[float]]:
    players = [h for h in hands_labels if len(h) >= 1]
    if len(players) < 2:
        return None

    evaluator = Evaluator()

    known = []
    try:
        for b in board_labels:
            known.append(label_to_treys(b))
        for h in players:
            for c in h:
                known.append(label_to_treys(c))
    except Exception:
        return None

    if len(set(known)) != len(known):
        return None

    ranks = "23456789TJQKA"
    suits = "cdhs"
    full_deck = [Card.new(r + s) for r in ranks for s in suits]

    known_set = set(known)
    remaining = [c for c in full_deck if c not in known_set]

    board_known = [label_to_treys(b) for b in board_labels]
    board_missing = max(0, 5 - len(board_known))

    hole_known = []
    hole_missing = []
    for h in players:
        hk = [label_to_treys(x) for x in h]
        hole_known.append(hk)
        hole_missing.append(max(0, 2 - len(hk)))

    need_total = board_missing + sum(hole_missing)
    if need_total > len(remaining):
        return None

    win_share = [0.0 for _ in players]

    for _ in range(sims):
        drawn = random.sample(remaining, need_total) if need_total > 0 else []
        idx = 0

        board = list(board_known)
        if board_missing:
            board.extend(drawn[idx:idx + board_missing])
            idx += board_missing

        hands = []
        for i in range(len(players)):
            h = list(hole_known[i])
            m = hole_missing[i]
            if m:
                h.extend(drawn[idx:idx + m])
                idx += m
            hands.append(h)

        scores = [evaluator.evaluate(board, h) for h in hands]
        best = min(scores)
        winners = [i for i, sc in enumerate(scores) if sc == best]

        share = 1.0 / len(winners)
        for w in winners:
            win_share[w] += share

    return [ws / sims for ws in win_share]


def sort_hands_for_display(hands: List[List[CardInstance]], frame_shape) -> List[List[CardInstance]]:
    H, W = frame_shape[:2]
    cx0, cy0 = W / 2.0, H / 2.0

    def ang(cluster):
        cx, cy = cluster_center(cluster)
        return math.atan2(cy - cy0, cx - cx0)

    return sorted(hands, key=ang)


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="PokerVision realtime: YOLO + stabilization + clustering + Hold'em equities (OBS/DroidCam supported)."
    )
    p.add_argument("--list_cams", action="store_true", help="List available camera indices and exit.")
    p.add_argument("--backend", type=str, default="auto", choices=["auto", "dshow", "msmf", "any"],
                   help="Force a VideoCapture backend (Windows: dshow/msmf).")

    p.add_argument("--model", type=str, default="runs/detect/train_corners_v1/weights/best.pt")
    p.add_argument("--source", type=str, default="0",
                   help="0/1/... for webcam OR 'video=OBS Virtual Camera' OR 'OBS Virtual Camera' OR http/rtsp URL.")
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--imgsz", type=int, default=672)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--max_det", type=int, default=120)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--sims", type=int, default=DEFAULT_SIMS)

    p.add_argument("--cluster_scale", type=float, default=4.0)
    p.add_argument("--font", type=str, default=None,
                   help="Path to a .ttf font that supports ♠♥♦♣. On Windows: C:\\Windows\\Fonts\\seguisym.ttf")
    p.add_argument("--start_mode", type=str, default="holdem", choices=["detect", "holdem"])

    return p.parse_args()


def main():
    global LABEL_ORDER, NUM_CLASSES

    args = parse_args()

    if args.list_cams:
        list_available_cameras(max_index=20, backend=args.backend)
        return

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Tip: run with --model <your_best.pt> or use --list_cams without needing a model."
        )

    print(f"[INFO] Loading model from {model_path}")
    model = YOLO(str(model_path))

    model_labels = get_model_labels(model)
    print(f"[INFO] Model nc={len(model_labels)}. First 10 labels: {model_labels[:10]}")
    LABEL_ORDER = model_labels
    NUM_CLASSES = len(LABEL_ORDER)

    source = convert_source(args.source)

    cap = open_capture(source, backend=args.backend)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source!r}")

    win_name = "PokerVision (m=toggle mode, q=quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    tr = TextRenderer(font_path=args.font)
    mode = 1 if args.start_mode == "holdem" else 0

    def on_trackbar(_v):
        pass

    try:
        cv2.createTrackbar("Mode 0=Detect 1=Holdem", win_name, mode, 1, on_trackbar)
    except Exception:
        pass

    tracks: List[Track] = []
    next_track_id = 0
    frame_idx = 0

    stable_counter = 0
    prev_label_set: set = set()

    snapshot_board: List[str] = []
    snapshot_hands: List[List[str]] = []
    snapshot_equities: Optional[List[float]] = None
    snapshot_clusters: List[Cluster] = []
    last_calc_board = []
    last_calc_hands = []

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1
        if args.frame_stride > 1 and (frame_idx % args.frame_stride != 0):
            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                mode = 1 - mode
            continue

        try:
            mode = cv2.getTrackbarPos("Mode 0=Detect 1=Holdem", win_name)
        except Exception:
            pass

        res = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            max_det=args.max_det,
            verbose=False
        )[0]

        boxes = getattr(res, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            xyxy = np.zeros((0, 4), dtype=float)
            cls_ids = np.zeros((0,), dtype=int)
            confs = np.zeros((0,), dtype=float)
        else:
            xyxy = boxes.xyxy.cpu().numpy().astype(float)
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy().astype(float)

        order = np.argsort(-confs) if len(confs) else []
        used_track_ids = set()

        for det_idx in order:
            det_box = xyxy[det_idx]
            det_cls = int(cls_ids[det_idx])
            det_conf = float(confs[det_idx])

            if det_cls < 0 or det_cls >= NUM_CLASSES:
                continue

            best_iou, best_track = 0.0, None

            for t in tracks:
                if t.track_id in used_track_ids:
                    continue
                if t.cls_id != det_cls:
                    continue
                iou_val = iou_xyxy(det_box, t.bbox)
                if iou_val > best_iou:
                    best_iou, best_track = iou_val, t

            if best_track is None:
                for t in tracks:
                    if t.track_id in used_track_ids:
                        continue
                    iou_val = iou_xyxy(det_box, t.bbox)
                    if iou_val > best_iou:
                        best_iou, best_track = iou_val, t

            if best_track is not None and best_iou >= IOU_THRESH:
                update_track_with_detection(best_track, det_box, det_cls, det_conf)
                best_track.last_seen = frame_idx
                used_track_ids.add(best_track.track_id)
            else:
                tracks.append(Track(
                    track_id=next_track_id,
                    bbox=det_box.copy(),
                    cls_id=det_cls,
                    score=det_conf,
                    last_seen=frame_idx,
                    age=0,
                    stable_frames=1,
                    cand_cls=-1,
                    cand_count=0
                ))
                used_track_ids.add(next_track_id)
                next_track_id += 1

        for t in tracks:
            if t.track_id not in used_track_ids:
                t.score *= MISS_SCORE_DECAY
                t.stable_frames = max(0, t.stable_frames - MISS_STABLE_DECAY)
                t.cand_cls = -1
                t.cand_count = 0

        tracks = [t for t in tracks if (frame_idx - t.last_seen) <= MAX_MISSED_FRAMES]

        draw_cards, eval_cards = extract_card_instances_from_tracks(
            tracks=tracks,
            frame_idx=frame_idx,
            num_classes=NUM_CLASSES,
            labels=LABEL_ORDER
        )

        if mode == 1:
            eval_label_set = set(ci.label for ci in eval_cards)
            sim = jaccard(eval_label_set, prev_label_set)

            if sim >= EVAL_SET_JACCARD_MIN:
                stable_counter += 1
            else:
                stable_counter = 0
                prev_label_set = eval_label_set

            if stable_counter >= EVAL_STABLE_SET_FRAMES:
                stable_counter = 0
                prev_label_set = eval_label_set

                grouped = cluster_cards_into_groups(eval_cards, frame.shape, cluster_scale=args.cluster_scale)
                board_cards, hand_clusters = pick_board_and_hands(grouped, frame.shape)
                hand_clusters = sort_hands_for_display(hand_clusters, frame.shape)

                # 1. Get the current labels
                current_board_labels = sorted([c.label for c in board_cards])
                current_hand_labels = [sorted([c.label for c in hc]) for hc in hand_clusters]

                # 2. CHECK: Only compute if the cards have changed!
                # We compare current labels against the ones we used for the last calculation.
                if (current_board_labels != last_calc_board or 
                    current_hand_labels != last_calc_hands):
                    
                    snapshot_equities = compute_equities(
                        board_labels=current_board_labels,
                        hands_labels=current_hand_labels,
                        sims=args.sims
                    )
                    
                    # Update the "last calc" cache
                    last_calc_board = current_board_labels
                    last_calc_hands = current_hand_labels
                    
                    # Also update the snapshot lists for display
                    snapshot_board = current_board_labels
                    snapshot_hands = current_hand_labels

                # 3. Always update positions (bbox/center) so the labels follow the cards smoothly
                # even if we didn't re-calculate the percentages.
                snapshot_clusters = []
                if board_cards:
                    bb = cluster_bbox(board_cards)
                    cc = cluster_center(board_cards)
                    snapshot_clusters.append(Cluster(cards=board_cards, bbox=bb, center=cc, role="BOARD"))
                for i, hc in enumerate(hand_clusters[:5], start=1):
                    bb = cluster_bbox(hc)
                    cc = cluster_center(hc)
                    snapshot_clusters.append(Cluster(cards=hc, bbox=bb, center=cc, role=f"P{i}"))

        else:
            stable_counter = 0
            # Optional: Clear cache when switching modes if desired
            # last_calc_board = []
            # last_calc_hands = []

        if mode == 0:
            for ci in draw_cards:
                draw_card_box(frame, tr, ci, show_score=True)
            draw_tag(frame, tr, 10, 10, "MODE: Detect (press m)", (255, 255, 255))
        else:
            draw_tag(frame, tr, 10, 10, "MODE: Holdem (press m)", (0, 255, 255))
            if len(snapshot_clusters) == 0:
                for ci in draw_cards:
                    draw_card_box(frame, tr, ci, show_score=True)
                draw_tag(frame, tr, 10, 42, f"Waiting stable set... {stable_counter}/{EVAL_STABLE_SET_FRAMES}", (0, 255, 255))
            else:
                for cl in snapshot_clusters:
                    x1, y1, x2, y2 = cl.bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                    title = cl.role
                    if cl.role.startswith("P") and snapshot_equities is not None:
                        try:
                            idx = int(cl.role[1:]) - 1
                            if 0 <= idx < len(snapshot_equities):
                                title = f"{cl.role} win~{100.0 * snapshot_equities[idx]:.1f}%"
                        except Exception:
                            pass

                    draw_tag(frame, tr, x1, max(0, y1 - 30), title, (0, 255, 255))

                    cards_sorted = sorted([c.label for c in cl.cards])
                    line_y = min(frame.shape[0] - 10, y2 + 28)
                    draw_token_line(frame, tr, cards_sorted, org=(x1, line_y), font_h=28, thickness=2, pad=14)

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            mode = 1 - mode
            try:
                cv2.setTrackbarPos("Mode 0=Detect 1=Holdem", win_name, mode)
            except Exception:
                pass

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] terminated.")


if __name__ == "__main__":
    main()
