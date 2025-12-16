# src/dataset_gen/generate_dataset.py
import argparse
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from itertools import combinations

from src.utils.config import (
    ensure_dirs,
    STYLE_GROUP_DIRS,
    STYLE_WEIGHTS,
    BACKGROUNDS_DIR,
    CARD_CLASSES,
)
from src.dataset_gen.yolo_label_utils import bbox_to_yolo


# -----------------------
# IMPORTANT: bump whenever you change ROI/segmentation/selection
# so masks are recomputed.
# -----------------------
CACHE_VERSION = "v6_edgeContours_primary_fallbackLab_hysteresis_fixedPlacement_20251214"

# Fixed label order: rank-major (2..A), suit alphabetical (C, D, H, S)
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

# -----------------------
# Dataset paths
# -----------------------
CORNER_BASE_DIR = Path("data/yolo_cards_corners")
CORNER_IMAGES_DIR = CORNER_BASE_DIR / "images"
CORNER_LABELS_DIR = CORNER_BASE_DIR / "labels"
DEBUG_DIR = CORNER_BASE_DIR / "debug"
MASK_CACHE_DIR = CORNER_BASE_DIR / "mask_cache"

# -----------------------
# Debug paths
# -----------------------
ROI_DEBUG_ASSETS_DIR = CORNER_BASE_DIR / "roi_debug_assets"
ROI_DEBUG_GEN_CARDS_DIR = CORNER_BASE_DIR / "roi_debug_generated_cards"
ROI_DEBUG_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
ROI_DEBUG_GEN_CARDS_DIR.mkdir(parents=True, exist_ok=True)

for d in [CORNER_IMAGES_DIR, CORNER_LABELS_DIR, DEBUG_DIR, MASK_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -----------------------
# Corner ROI fractions (raw upright card coords)
# Keep ROI big enough to contain BOTH rank+small suit for all ranks.
# -----------------------
# User-requested ROI
TL_X0_F, TL_Y0_F = 0.022, 0.014
TL_X1_F, TL_Y1_F = 0.220, 0.360

# -----------------------
# Segmentation thresholds
# -----------------------
ALPHA_THR = 10  # ignore transparent pixels in assets and in composed alpha

# LAB-distance fallback candidates (strict -> lenient)
INK_DIST_THR_CANDIDATES = [12, 11, 10, 9, 8, 7]

# Small noise reject
MIN_COMP_AREA = 6
MIN_INK_PIXELS_ROI = 40

# Morphology (kept conservative; dilation merges into pips/patterns)
OPEN_ITERS = 0
CLOSE_ITERS_PRE = 0
CLOSE_ITERS_POST = 0
DILATE_ITERS = 0

# -----------------------
# Edge-based corner extraction (PRIMARY)
# -----------------------
EDGE_BLUR_K = 5              # odd
EDGE_DILATE_ITERS = 1
EDGE_CLOSE_ITERS = 1

EDGE_MIN_AREA_FRAC = 0.0009  # fraction of ROI area
EDGE_MAX_AREA_FRAC = 0.45
EDGE_MIN_SOLIDITY = 0.22

# reject contours touching ROI border (frame lines)
EDGE_BORDER_MARGIN_FRAC = 0.015
EDGE_BORDER_MARGIN_PX_MIN = 2
EDGE_BORDER_MARGIN_PX_MAX = 10

# stripe rejection in ROI coords
STRIPE_THIN_FRAC = 0.06      # thin dimension threshold
STRIPE_LONG_FRAC = 0.55      # long dimension threshold

# -----------------------
# Subset-selection constraints (SECONDARY clean-up)
# -----------------------
TL_CX_MAX_FRAC = 0.92
TL_CY_MAX_FRAC = 0.94

# IMPORTANT: reject components touching ANY ROI border within this many pixels.
# Since ROI starts inside the card edge, valid symbols shouldn't touch ROI borders.
TL_BORDER_PX = 2

UNION_MAX_AREA_FRAC = 0.92
UNION_MIN_W_FRAC = 0.10
UNION_MIN_H_FRAC = 0.10
UNION_MAX_X2_FRAC = 0.93
UNION_MAX_Y2_FRAC = 0.94

REL_AREA_KEEP_FRAC = 0.001
MAX_CANDIDATES = 12
MAX_SUBSET = 7

# rank/suit bands (NON-overlapping)
RANKLIKE_CY_MAX_FRAC = 0.48
SUITLIKE_CY_MIN_FRAC = 0.58

UNION_X1_MAX_FRAC = 0.50
UNION_Y1_MAX_FRAC = 0.70

UNION_MAX_W_FRAC = 0.75
UNION_MAX_H_FRAC = 0.90

# -----------------------
# Label visibility thresholds (on composed image)
# -----------------------
MIN_INK_PIXELS_GLOBAL = 35
MIN_VISIBLE_FRACTION = 0.50
MIN_INFRAME_FRACTION = 0.85

# -----------------------
# Placement distribution
# -----------------------
HARD_NEG_PROB_DEFAULT = 1
NUM_CARDS_MIN = 2
NUM_CARDS_MAX = 7

# -----------------------
# Hysteresis thresholding (LAB fallback)
# -----------------------
USE_HYSTERESIS = True
HYST_DELTA = 6
HYST_MAX_ITERS = 25
DIST_BLUR_K = 3            # must be odd >=3 if used
DIST_BLUR_SIGMA = 0.6


# -----------------------
# Debug helpers
# -----------------------
def _mask_u8_from_bool(m: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if m is None:
        return None
    if m.dtype == np.bool_:
        return (m.astype(np.uint8) * 255)
    if m.dtype != np.uint8:
        return m.astype(np.uint8)
    return m

def _convex_hull_from_mask(mask_u8: np.ndarray) -> Optional[np.ndarray]:
    if mask_u8 is None or mask_u8.ndim != 2:
        return None
    if int((mask_u8 > 0).sum()) < 5:
        return None
    cnts, _ = cv2.findContours((mask_u8 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    pts = np.vstack(cnts)
    if pts.shape[0] < 3:
        return None
    return cv2.convexHull(pts)

def _draw_roi_and_hulls_on_rgba(
    card_rgba: Image.Image,
    roi_rect_mask_u8: Optional[np.ndarray],
    tl_mask_u8: Optional[np.ndarray],
    br_mask_u8: Optional[np.ndarray],
) -> np.ndarray:
    """
    Returns BGR uint8 image with:
      - ROI rectangle outline (blue)
      - TL mask convex hull (green)
      - BR mask convex hull (yellow)
    """
    bgr = cv2.cvtColor(np.array(card_rgba), cv2.COLOR_RGBA2BGR)

    if roi_rect_mask_u8 is not None:
        cnts, _ = cv2.findContours((roi_rect_mask_u8 > 0).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(bgr, cnts, -1, (255, 0, 0), 2)

    if tl_mask_u8 is not None:
        hull = _convex_hull_from_mask(tl_mask_u8)
        if hull is not None:
            cv2.polylines(bgr, [hull], True, (0, 255, 0), 2)

    if br_mask_u8 is not None:
        hull = _convex_hull_from_mask(br_mask_u8)
        if hull is not None:
            cv2.polylines(bgr, [hull], True, (0, 255, 255), 2)

    return bgr


# -----------------------
# Utilities
# -----------------------
def load_backgrounds() -> List[Image.Image]:
    bgs: List[Image.Image] = []
    for bg_path in BACKGROUNDS_DIR.iterdir():
        if bg_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            try:
                bgs.append(Image.open(bg_path).convert("RGB"))
            except Exception as e:
                print(f"[WARN] Failed to load background {bg_path}: {e}")
    if not bgs:
        print("[ERROR] No backgrounds found in data/backgrounds/")
    return bgs

def _pil_to_bgra(pil_rgba: Image.Image) -> np.ndarray:
    arr = np.array(pil_rgba)  # RGBA
    return arr[:, :, [2, 1, 0, 3]].copy()  # BGRA

def _roi_from_frac(w: int, h: int, x0f: float, y0f: float, x1f: float, y1f: float) -> Tuple[int, int, int, int]:
    x0 = int(round(x0f * w))
    y0 = int(round(y0f * h))
    x1 = int(round(x1f * w))
    y1 = int(round(y1f * h))
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(1, min(w, x1))
    y1 = max(1, min(h, y1))
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return x0, y0, x1, y1

def _ensure_odd_ksize(k: int) -> int:
    k = int(k)
    if k < 3:
        return 0
    if k % 2 == 0:
        k += 1
    return k

def _auto_canny(gray: np.ndarray, sigma: float = 0.33) -> Tuple[int, int]:
    v = float(np.median(gray))
    lo = int(max(0, (1.0 - sigma) * v))
    hi = int(min(255, (1.0 + sigma) * v))
    if hi <= lo:
        hi = min(255, lo + 40)
    return lo, hi

def _union_bbox(comps: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    x1 = min(c[0] for c in comps)
    y1 = min(c[1] for c in comps)
    x2 = max(c[2] for c in comps)
    y2 = max(c[3] for c in comps)
    return x1, y1, x2, y2

def _bbox_area(b: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


# -----------------------
# PRIMARY corner mask: edges -> contours -> filled selected contours
# -----------------------
def _edge_contour_ink_mask(zone_bgra: np.ndarray) -> Optional[np.ndarray]:
    zh, zw = zone_bgra.shape[:2]
    if zh < 8 or zw < 8:
        return None

    bgr = zone_bgra[:, :, :3]
    a = zone_bgra[:, :, 3]
    opaque = (a > ALPHA_THR)

    if int(opaque.sum()) < 60:
        return None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Make transparent pixels neutral so they don't create edges
    med = int(np.median(gray[opaque]))
    gray2 = gray.copy()
    gray2[~opaque] = med

    k = _ensure_odd_ksize(EDGE_BLUR_K)
    if k:
        gray2 = cv2.GaussianBlur(gray2, (k, k), 0)

    t1, t2 = _auto_canny(gray2, sigma=0.33)
    edges = cv2.Canny(gray2, t1, t2)
    edges = (edges & (opaque.astype(np.uint8) * 255)).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    if EDGE_DILATE_ITERS > 0:
        edges = cv2.dilate(edges, kernel, iterations=EDGE_DILATE_ITERS)
    if EDGE_CLOSE_ITERS > 0:
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=EDGE_CLOSE_ITERS)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    roi_area = float(zh * zw)
    min_area = max(20.0, EDGE_MIN_AREA_FRAC * roi_area)
    max_area = EDGE_MAX_AREA_FRAC * roi_area

    margin = int(round(EDGE_BORDER_MARGIN_FRAC * min(zh, zw)))
    margin = max(EDGE_BORDER_MARGIN_PX_MIN, min(EDGE_BORDER_MARGIN_PX_MAX, margin))

    keep: List[np.ndarray] = []
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < min_area or area > max_area:
            continue

        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull))
        if hull_area <= 1e-6:
            continue
        solidity = area / hull_area
        if solidity < EDGE_MIN_SOLIDITY:
            continue

        x, y, w, h = cv2.boundingRect(c)
        x2, y2 = x + w, y + h

        # reject anything touching ROI borders (frame lines)
        if x <= margin or y <= margin or x2 >= (zw - margin) or y2 >= (zh - margin):
            continue

        # reject stripes
        if (h / float(zh)) <= STRIPE_THIN_FRAC and (w / float(zw)) >= STRIPE_LONG_FRAC:
            continue
        if (w / float(zw)) <= STRIPE_THIN_FRAC and (h / float(zh)) >= STRIPE_LONG_FRAC:
            continue

        keep.append(c)

    if not keep:
        return None

    mask = np.zeros((zh, zw), np.uint8)
    cv2.drawContours(mask, keep, -1, 255, thickness=-1)

    # very light cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


# -----------------------
# SECONDARY clean-up: connected components + subset selection
# -----------------------
def _select_components_subset_tl(ink_u8: np.ndarray) -> np.ndarray:
    H, W = ink_u8.shape[:2]
    bin_mask = (ink_u8 > 0).astype(np.uint8)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if num <= 1:
        return np.zeros_like(ink_u8)

    cand = []
    for comp_id in range(1, num):
        x = int(stats[comp_id, cv2.CC_STAT_LEFT])
        y = int(stats[comp_id, cv2.CC_STAT_TOP])
        w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        cx, cy = centroids[comp_id]

        if area < MIN_COMP_AREA:
            continue

        x2 = x + w
        y2 = y + h

        # reject components touching ANY ROI border (frame / cut-through)
        if (x <= TL_BORDER_PX) or (y <= TL_BORDER_PX) or (x2 >= (W - TL_BORDER_PX)) or (y2 >= (H - TL_BORDER_PX)):
            continue

        # keep near TL (loose)
        if cx >= TL_CX_MAX_FRAC * W or cy >= TL_CY_MAX_FRAC * H:
            continue

        # reject extreme stripes by bbox aspect
        bw = max(1, w)
        bh = max(1, h)
        aspect = float(max(bw, bh)) / float(min(bw, bh))
        if aspect > 18.0:
            continue

        cand.append((comp_id, area, x, y, x2, y2, float(cx), float(cy)))

    if not cand:
        return np.zeros_like(ink_u8)

    max_area = max(c[1] for c in cand)
    thr_area = max(MIN_COMP_AREA, int(REL_AREA_KEEP_FRAC * max_area))
    cand_pruned = [c for c in cand if c[1] >= thr_area]
    if len(cand_pruned) >= 2 or len(cand) == 1:
        cand = cand_pruned

    if not cand:
        return np.zeros_like(ink_u8)

    roi_area = float(H * W)

    def subset_ok(b):
        x1, y1, x2, y2 = b
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            return False

        if (bw / W) < UNION_MIN_W_FRAC:
            return False
        if (bh / H) < UNION_MIN_H_FRAC:
            return False
        if (bw / W) > UNION_MAX_W_FRAC:
            return False
        if (bh / H) > UNION_MAX_H_FRAC:
            return False

        area_frac = _bbox_area(b) / roi_area
        if area_frac > UNION_MAX_AREA_FRAC:
            return False
        if (x2 / W) > UNION_MAX_X2_FRAC:
            return False
        if (y2 / H) > UNION_MAX_Y2_FRAC:
            return False
        if (x1 / W) > UNION_X1_MAX_FRAC:
            return False
        if (y1 / H) > UNION_Y1_MAX_FRAC:
            return False

        return True

    def is_ranklike(c):
        return (c[7] / H) <= RANKLIKE_CY_MAX_FRAC

    def is_suitlike(c):
        return (c[7] / H) >= SUITLIKE_CY_MIN_FRAC

    def y_top_frac(c):
        return c[3] / H

    def y_bot_frac(c):
        return c[5] / H

    def spans_both(c):
        # for a single connected blob that genuinely covers both
        return (y_top_frac(c) < 0.40) and (y_bot_frac(c) > 0.65)

    def cand_score(c):
        _, area, x, y, x2, y2, cx, cy = c
        bw = max(1, x2 - x)
        bh = max(1, y2 - y)
        fill = float(area) / float(bw * bh)
        aspect = float(max(bw, bh)) / float(min(bw, bh))
        aspect_pen = 1.0 / (1.0 + 0.25 * max(0.0, aspect - 1.0))
        tl_pen = 0.90 * (cx / W) + 0.90 * (cy / H)
        return (float(area) * (0.40 + 0.60 * fill) * aspect_pen) / (1.0 + tl_pen)

    cand_all = sorted(cand, key=cand_score, reverse=True)
    cand_sorted = cand_all[:MAX_CANDIDATES]

    need_rank = any(is_ranklike(c) for c in cand)
    need_suit = any(is_suitlike(c) for c in cand)

    def subset_has_required_parts(subset):
        has_rank = any(is_ranklike(c) for c in subset)
        has_suit = any(is_suitlike(c) for c in subset)
        if need_rank and not has_rank:
            return False
        if need_suit and not has_suit:
            return False
        if need_rank and need_suit and len(subset) == 1 and not spans_both(subset[0]):
            return False
        return True

    def subset_objective(subset, union):
        ink_area = float(sum(c[1] for c in subset))
        bbox_area = float(max(1, _bbox_area(union)))
        density = ink_area / bbox_area

        x1, y1, x2, y2 = union
        corner_bonus = -0.55 * (x1 / W + y1 / H) - 0.08 * (x2 / W + y2 / H)

        align_bonus = 0.0
        rank_candidates = [c for c in subset if is_ranklike(c)]
        suit_candidates = [c for c in subset if is_suitlike(c)]
        if rank_candidates and suit_candidates:
            r = min(rank_candidates, key=lambda c: c[7])
            s = max(suit_candidates, key=lambda c: c[7])
            dx = abs(r[6] - s[6]) / float(W)
            sep = (s[7] - r[7]) / float(H)
            align_bonus += 0.18 * max(0.0, sep - 0.10)
            align_bonus -= 0.32 * dx

        frag_pen = -0.02 * max(0, len(subset) - 3)
        return density + 0.00010 * ink_area + corner_bonus + align_bonus + frag_pen

    best_subset = None
    best_obj = -1e18

    for k in range(1, min(MAX_SUBSET, len(cand_sorted)) + 1):
        for subset in combinations(cand_sorted, k):
            if not subset_has_required_parts(subset):
                continue
            boxes = [(c[2], c[3], c[4], c[5]) for c in subset]
            union = _union_bbox(boxes)
            if not subset_ok(union):
                continue
            obj = subset_objective(subset, union)
            if obj > best_obj:
                best_obj = obj
                best_subset = subset

    if best_subset is None:
        return np.zeros_like(ink_u8)

    keep_mask = np.zeros_like(bin_mask)
    for c in best_subset:
        keep_mask[labels == c[0]] = 1

    return (keep_mask.astype(np.uint8) * 255)


# -----------------------
# LAB-distance fallback (kept, safer)
# -----------------------
def _estimate_bg_lab(zone_bgra: np.ndarray) -> np.ndarray:
    """
    Robust background LAB using k-means on opaque pixels.
    IMPORTANT: to avoid "gold becomes background", we pick the MAJORITY cluster
    but also bias toward the cluster that is more spatially spread (background tends to be spread).
    """
    h, w = zone_bgra.shape[:2]
    a = zone_bgra[:, :, 3]
    opaque = a > ALPHA_THR

    if int(opaque.sum()) < 80:
        return np.array([255, 128, 128], dtype=np.float32)

    bgr = zone_bgra[:, :, :3]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    vals = lab[opaque].reshape(-1, 3)
    if vals.shape[0] < 80:
        return np.array([255, 128, 128], dtype=np.float32)

    # subsample for speed
    if vals.shape[0] > 7000:
        idx = np.random.choice(vals.shape[0], 7000, replace=False)
        vals_s = vals[idx]
    else:
        vals_s = vals

    K = 3 if vals_s.shape[0] >= 3 else 1
    if K == 1:
        return np.median(vals_s, axis=0).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.25)
    _, km_labels, centers = cv2.kmeans(vals_s, K, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    km_labels = km_labels.reshape(-1)

    counts = np.array([(km_labels == i).sum() for i in range(K)], dtype=np.float32)

    # spatial spread proxy: sample positions too (cheap)
    ys, xs = np.where(opaque)
    if ys.size > 7000:
        ii = np.random.choice(ys.size, 7000, replace=False)
        ys_s, xs_s = ys[ii], xs[ii]
    else:
        ys_s, xs_s = ys, xs

    # assign sampled positions to labels (approx by sampling same indices length)
    # (good enough as a bias term)
    lbl_pos = km_labels[:min(km_labels.size, ys_s.size)]
    ys_s = ys_s[:lbl_pos.size]
    xs_s = xs_s[:lbl_pos.size]

    spread = np.zeros((K,), dtype=np.float32)
    for i in range(K):
        m = (lbl_pos == i)
        if m.sum() < 20:
            spread[i] = 0.0
        else:
            spread[i] = float(np.std(xs_s[m]) + np.std(ys_s[m]))

    # score = majority + small spread bonus
    score = counts + 0.15 * spread
    bg_idx = int(np.argmax(score))
    return centers[bg_idx].astype(np.float32)

def _hysteresis_ink_u8(dist: np.ndarray, alpha: np.ndarray, thr_hi: float) -> np.ndarray:
    thr_lo = float(max(0.0, thr_hi - HYST_DELTA))
    opaque = (alpha > ALPHA_THR)

    strong = ((dist > thr_hi) & opaque).astype(np.uint8)
    if strong.sum() == 0:
        return strong * 255

    weak = ((dist > thr_lo) & opaque).astype(np.uint8)

    cur = strong.copy()
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(HYST_MAX_ITERS):
        nxt = (cv2.dilate(cur, kernel, iterations=1) & weak).astype(np.uint8)
        if np.array_equal(nxt, cur):
            break
        cur = nxt

    return (cur * 255).astype(np.uint8)

def _largest_reasonable_ink_mask_lab(zone_bgra: np.ndarray) -> Optional[np.ndarray]:
    zh, zw = zone_bgra.shape[:2]
    if zh < 6 or zw < 6:
        return None

    bg_lab = _estimate_bg_lab(zone_bgra)
    bgr = zone_bgra[:, :, :3]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = zone_bgra[:, :, 3].astype(np.float32)
    dist = np.linalg.norm(lab - bg_lab.reshape(1, 1, 3), axis=2)

    kernel = np.ones((3, 3), np.uint8)
    best = None
    best_obj = -1e9

    for thr in INK_DIST_THR_CANDIDATES:
        dist_use = dist
        k = _ensure_odd_ksize(DIST_BLUR_K)
        if k:
            dist_use = cv2.GaussianBlur(dist_use, (k, k), sigmaX=DIST_BLUR_SIGMA)

        if USE_HYSTERESIS:
            ink_u8 = _hysteresis_ink_u8(dist_use, a, float(thr))
        else:
            ink = (dist_use > float(thr)) & (a > ALPHA_THR)
            ink_u8 = (ink.astype(np.uint8) * 255)

        if OPEN_ITERS > 0:
            ink_u8 = cv2.morphologyEx(ink_u8, cv2.MORPH_OPEN, kernel, iterations=OPEN_ITERS)
        if CLOSE_ITERS_PRE > 0:
            ink_u8 = cv2.morphologyEx(ink_u8, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERS_PRE)

        ink_u8 = _select_components_subset_tl(ink_u8)

        if CLOSE_ITERS_POST > 0:
            ink_u8 = cv2.morphologyEx(ink_u8, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERS_POST)
            ink_u8 = _select_components_subset_tl(ink_u8)

        ink_pixels = int((ink_u8 > 0).sum())
        if ink_pixels < MIN_INK_PIXELS_ROI:
            continue

        ys, xs = np.where(ink_u8 > 0)
        if xs.size == 0 or ys.size == 0:
            continue

        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        density = ink_pixels / float(bbox_area)
        obj = density + 0.00012 * ink_pixels

        if obj > best_obj:
            best_obj = obj
            best = ink_u8.copy()

    return best


# -----------------------
# Combined corner mask selection
# -----------------------
def _largest_reasonable_ink_mask(zone_bgra: np.ndarray) -> Optional[np.ndarray]:
    """
    PRIMARY: edge-contour mask (robust to black/gold, borders)
    SECONDARY: lab-distance mask (your old approach, safer now)
    """
    m1 = _edge_contour_ink_mask(zone_bgra)
    if m1 is not None:
        # secondary cleanup to remove any leftover junk
        m1 = _select_components_subset_tl(m1)
        if int((m1 > 0).sum()) >= MIN_INK_PIXELS_ROI:
            return m1

    m2 = _largest_reasonable_ink_mask_lab(zone_bgra)
    if m2 is not None and int((m2 > 0).sum()) >= MIN_INK_PIXELS_ROI:
        return m2

    return None


# -----------------------
# Find TL/BR masks on raw asset
# -----------------------
def _find_corner_masks_raw(img_pil_rgba: Image.Image) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    img_bgra = _pil_to_bgra(img_pil_rgba)
    H, W = img_bgra.shape[:2]

    x0, y0, x1, y1 = _roi_from_frac(W, H, TL_X0_F, TL_Y0_F, TL_X1_F, TL_Y1_F)
    zone = img_bgra[y0:y1, x0:x1].copy()
    tl_roi_mask = _largest_reasonable_ink_mask(zone)

    tl_full = None
    if tl_roi_mask is not None:
        tl_full = np.zeros((H, W), np.uint8)
        tl_full[y0:y1, x0:x1] = tl_roi_mask

    img_rot = cv2.rotate(img_bgra, cv2.ROTATE_180)
    Hr, Wr = img_rot.shape[:2]
    x0r, y0r, x1r, y1r = _roi_from_frac(Wr, Hr, TL_X0_F, TL_Y0_F, TL_X1_F, TL_Y1_F)
    zone_r = img_rot[y0r:y1r, x0r:x1r].copy()
    br_roi_mask_on_rot = _largest_reasonable_ink_mask(zone_r)

    br_full = None
    if br_roi_mask_on_rot is not None:
        br_rot_full = np.zeros((Hr, Wr), np.uint8)
        br_rot_full[y0r:y1r, x0r:x1r] = br_roi_mask_on_rot
        br_full = cv2.rotate(br_rot_full, cv2.ROTATE_180)

    return tl_full, br_full


def _save_mask_cache(cache_path: Path, tl_mask: Optional[np.ndarray], br_mask: Optional[np.ndarray]):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(cache_path),
        tl=(tl_mask if tl_mask is not None else np.zeros((1, 1), np.uint8)),
        br=(br_mask if br_mask is not None else np.zeros((1, 1), np.uint8)),
        tl_valid=(1 if tl_mask is not None else 0),
        br_valid=(1 if br_mask is not None else 0),
    )

def _load_mask_cache(cache_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not cache_path.exists():
        return None, None
    data = np.load(str(cache_path), allow_pickle=False)
    tl_valid = int(data["tl_valid"])
    br_valid = int(data["br_valid"])
    tl = data["tl"]
    br = data["br"]
    tl_mask = tl if tl_valid == 1 and tl.size > 1 else None
    br_mask = br if br_valid == 1 and br.size > 1 else None
    return tl_mask, br_mask


@dataclass
class CardAsset:
    cls: str
    style: str
    path: Path
    img: Image.Image
    tl_mask_raw: Optional[np.ndarray]
    br_mask_raw: Optional[np.ndarray]


def load_card_assets() -> Dict[str, Dict[str, List[CardAsset]]]:
    style_names = list(STYLE_GROUP_DIRS.keys())
    assets: Dict[str, Dict[str, List[CardAsset]]] = {cls: {s: [] for s in style_names} for cls in CARD_CLASSES}

    for cls in CARD_CLASSES:
        for style, dir_list in STYLE_GROUP_DIRS.items():
            for d in dir_list:
                p = d / f"{cls}.png"
                if not p.exists():
                    continue
                try:
                    img = Image.open(p).convert("RGBA")
                except Exception as e:
                    print(f"[WARN] Failed loading {p}: {e}")
                    continue

                cache_name = f"{CACHE_VERSION}__{style}__{cls}__{p.parent.name}.npz"
                cache_path = MASK_CACHE_DIR / cache_name

                tl_mask, br_mask = _load_mask_cache(cache_path)
                if tl_mask is None and br_mask is None:
                    tl_mask, br_mask = _find_corner_masks_raw(img)

                    # ---- ROI debug dump for raw asset ----
                    try:
                        W0, H0 = img.size
                        x0, y0, x1, y1 = _roi_from_frac(W0, H0, TL_X0_F, TL_Y0_F, TL_X1_F, TL_Y1_F)
                        roi_rect = np.zeros((H0, W0), np.uint8)
                        roi_rect[y0:y1, x0:x1] = 255

                        tl_u8 = _mask_u8_from_bool(tl_mask)
                        br_u8 = _mask_u8_from_bool(br_mask)

                        vis_bgr = _draw_roi_and_hulls_on_rgba(img, roi_rect, tl_u8, br_u8)
                        out_name = f"{style}__{cls}__{p.parent.name}.jpg"
                        cv2.imwrite(str(ROI_DEBUG_ASSETS_DIR / out_name), vis_bgr)
                    except Exception as e:
                        print(f"[WARN] ROI debug dump failed for {p}: {e}")

                    _save_mask_cache(cache_path, tl_mask, br_mask)

                assets[cls][style].append(CardAsset(
                    cls=cls, style=style, path=p, img=img, tl_mask_raw=tl_mask, br_mask_raw=br_mask
                ))

    for cls in CARD_CLASSES:
        if all(len(v) == 0 for v in assets[cls].values()):
            print(f"[WARN] No variants found for {cls}")

    return assets


def choose_card_asset(assets_for_cls: Dict[str, List[CardAsset]]) -> Optional[CardAsset]:
    styles = [s for s, lst in assets_for_cls.items() if len(lst) > 0]
    if not styles:
        return None
    weights = [STYLE_WEIGHTS.get(s, 0.0) for s in styles]
    if sum(weights) <= 0:
        weights = [1.0] * len(styles)
    style_choice = random.choices(styles, weights=weights, k=1)[0]
    return random.choice(assets_for_cls[style_choice])

def _available_classes(card_assets) -> List[str]:
    return [cls for cls, style_dict in card_assets.items() if any(len(lst) > 0 for lst in style_dict.values())]


# -----------------------
# Hard negative: draw random rank OR suit symbols (separately)
# -----------------------
_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
_SUITS = ["♣", "♦", "♥", "♠"]

def _try_load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\seguisym.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()

def draw_fake_occluders(bg_img: Image.Image, num_shapes=4) -> Image.Image:
    bg = bg_img.convert("RGB").copy()
    W, H = bg.size

    for _ in range(num_shapes):
        color_choice = random.choice([(0, 0, 0), (200, 0, 0), (205, 170, 60)])
        text = random.choice(_RANKS) if random.random() < 0.5 else random.choice(_SUITS)

        fs = int(random.uniform(0.045, 0.16) * min(W, H))
        fs = max(18, min(fs, 170))
        font = _try_load_font(fs)

        tmp = Image.new("L", (10, 10), 0)
        d0 = ImageDraw.Draw(tmp)
        bbox = d0.textbbox((0, 0), text, font=font)
        tw = max(1, bbox[2] - bbox[0])
        th = max(1, bbox[3] - bbox[1])

        pad = int(0.35 * fs)
        mw, mh = tw + 2 * pad, th + 2 * pad

        mask = Image.new("L", (mw, mh), 0)
        dm = ImageDraw.Draw(mask)
        dm.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=255)

        patch = Image.new("RGBA", (mw, mh), (0, 0, 0, 0))
        color_layer = Image.new("RGBA", (mw, mh), (*color_choice, 255))
        patch = Image.composite(color_layer, patch, mask)

        angle = random.uniform(-40, 40)
        patch = patch.rotate(angle, expand=True, resample=Image.BICUBIC)

        px = random.randint(0, max(0, W - patch.size[0]))
        py = random.randint(0, max(0, H - patch.size[1]))
        bg.paste(patch, (px, py), patch)

    return bg

def make_hard_negative_example(backgrounds: Union[List[Image.Image], Image.Image]) -> np.ndarray:
    bg = backgrounds if isinstance(backgrounds, Image.Image) else random.choice(backgrounds)
    img = draw_fake_occluders(bg, num_shapes=random.randint(3, 10))
    return np.array(img)[:, :, ::-1].copy()

def apply_post_augs(img_bgr: np.ndarray) -> np.ndarray:
    out = img_bgr.astype(np.float32)

    if random.random() < 0.9:
        alpha = random.uniform(0.6, 1.3)
        beta = random.uniform(-45, 35)
        out = out * alpha + beta

    if random.random() < 0.7:
        rb = random.uniform(0.9, 1.1)
        gb = random.uniform(0.9, 1.1)
        bb = random.uniform(0.9, 1.1)
        gain = np.array([bb, gb, rb], dtype=np.float32).reshape(1, 1, 3)
        out = out * gain

    if random.random() < 0.6:
        if random.random() < 0.5:
            k = random.choice([3, 5, 7])
            out = cv2.GaussianBlur(out, (k, k), sigmaX=random.uniform(0.8, 2.0))
        else:
            k = random.choice([3, 5, 7])
            kernel = np.zeros((k, k), np.float32)
            if random.random() < 0.5:
                kernel[int((k - 1) / 2), :] = 1.0 / k
            else:
                kernel[:, int((k - 1) / 2)] = 1.0 / k
            out = cv2.filter2D(out, -1, kernel)

    if random.random() < 0.7:
        sigma = random.uniform(5, 18)
        out = out + np.random.normal(0, sigma, out.shape).astype(np.float32)

    if random.random() < 0.6:
        quality = random.randint(25, 70)
        ok, enc = cv2.imencode(".jpg", np.clip(out, 0, 255).astype(np.uint8),
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)


# -----------------------
# Placements (rotate masks together with card)  [FIXED]
# -----------------------
def random_single_card_instances(card_assets, W, H, num_cards: int):
    placements = []
    available = _available_classes(card_assets)
    if not available:
        return placements

    chosen = random.sample(available, k=min(num_cards, len(available)))
    for cls in chosen:
        asset = choose_card_asset(card_assets[cls])
        if asset is None:
            continue

        orig_w, orig_h = asset.img.size
        if orig_w <= 0 or orig_h <= 0:
            continue

        if random.random() < 0.35:
            frac_min, frac_max = 0.18, 0.30
        else:
            frac_min, frac_max = 0.12, 0.18

        target_w = int(W * random.uniform(frac_min, frac_max))
        scale = target_w / float(orig_w)
        new_w = target_w
        new_h = int(orig_h * scale)
        if new_w < 10 or new_h < 10:
            continue

        card_resized = asset.img.resize((new_w, new_h), Image.BICUBIC)

        tl_mask_resized = None
        br_mask_resized = None
        if asset.tl_mask_raw is not None:
            tl_img = Image.fromarray(asset.tl_mask_raw, mode="L").resize((new_w, new_h), Image.NEAREST)
            tl_mask_resized = (np.array(tl_img) > 0)
        if asset.br_mask_raw is not None:
            br_img = Image.fromarray(asset.br_mask_raw, mode="L").resize((new_w, new_h), Image.NEAREST)
            br_mask_resized = (np.array(br_img) > 0)

        # ROI rectangle mask in resized-card coordinates
        roi_rect_resized = np.zeros((new_h, new_w), np.uint8)
        rx0, ry0, rx1, ry1 = _roi_from_frac(new_w, new_h, TL_X0_F, TL_Y0_F, TL_X1_F, TL_Y1_F)
        roi_rect_resized[ry0:ry1, rx0:rx1] = 255

        angle = random.uniform(-90, 90)

        card_rot = card_resized.rotate(angle, expand=True, resample=Image.BICUBIC)
        rot_w, rot_h = card_rot.size
        if rot_w >= W or rot_h >= H or rot_w < 5 or rot_h < 5:
            continue

        # rotate ROI rect and masks with same params
        roi_rect_rot_u8 = np.array(
            Image.fromarray(roi_rect_resized, mode="L").rotate(
                angle, expand=True, resample=Image.NEAREST, fillcolor=0
            )
        )

        tl_mask_rot = None
        br_mask_rot = None

        if tl_mask_resized is not None:
            tl_rot = Image.fromarray((tl_mask_resized.astype(np.uint8) * 255), mode="L").rotate(
                angle, expand=True, resample=Image.NEAREST, fillcolor=0
            )
            tl_mask_rot = (np.array(tl_rot) > 0)

        if br_mask_resized is not None:
            br_rot = Image.fromarray((br_mask_resized.astype(np.uint8) * 255), mode="L").rotate(
                angle, expand=True, resample=Image.NEAREST, fillcolor=0
            )
            br_mask_rot = (np.array(br_rot) > 0)

        xmin = random.randint(0, W - rot_w)
        ymin = random.randint(0, H - rot_h)
        xmax, ymax = xmin + rot_w, ymin + rot_h

        placements.append({
            "cls": cls,
            "img": card_rot,
            "bbox": (xmin, ymin, xmax, ymax),
            "tl_mask": tl_mask_rot,
            "br_mask": br_mask_rot,
            "roi_rect_u8": roi_rect_rot_u8,
        })

    return placements


# -----------------------
# Compose & label
# -----------------------
def compose_on_background(bg_img: Image.Image, placements: List[dict]):
    bg = bg_img.copy()
    W, H = bg.size

    for p in placements:
        card_img = p["img"]
        xmin, ymin, xmax, ymax = p["bbox"]

        vis_xmin = max(int(xmin), 0)
        vis_ymin = max(int(ymin), 0)
        vis_xmax = min(int(xmax), W)
        vis_ymax = min(int(ymax), H)
        if vis_xmin >= vis_xmax or vis_ymin >= vis_ymax:
            continue

        crop_left = vis_xmin - xmin
        crop_top = vis_ymin - ymin
        crop_right = crop_left + (vis_xmax - vis_xmin)
        crop_bottom = crop_top + (vis_ymax - vis_ymin)

        card_visible = card_img.crop((crop_left, crop_top, crop_right, crop_bottom))
        bg.paste(card_visible, (vis_xmin, vis_ymin), card_visible)

    occupancy = np.zeros((H, W), dtype=bool)
    annotations: List[Tuple[int, float, float, float, float]] = []

    for p in reversed(placements):
        cls = p["cls"]
        card_img = p["img"]
        xmin, ymin, xmax, ymax = p["bbox"]

        class_id = LABEL_ORDER.index(cls)

        alpha_local = np.array(card_img.split()[-1])
        card_local_mask = (alpha_local > ALPHA_THR)
        rot_h, rot_w = card_local_mask.shape

        gx0 = max(0, xmin)
        gy0 = max(0, ymin)
        gx1 = min(W, xmax)
        gy1 = min(H, ymax)
        if gx1 <= gx0 or gy1 <= gy0:
            continue

        lx0 = gx0 - xmin
        ly0 = gy0 - ymin
        lx1 = lx0 + (gx1 - gx0)
        ly1 = ly0 + (gy1 - gy0)

        card_crop = card_local_mask[ly0:ly1, lx0:lx1]
        if card_crop.size == 0 or not card_crop.any():
            continue

        def handle_corner_mask(cmask: Optional[np.ndarray]):
            if cmask is None or cmask.shape != (rot_h, rot_w):
                return

            ink_card = (cmask & card_local_mask)
            full_area = int(ink_card.sum())
            if full_area < MIN_INK_PIXELS_GLOBAL:
                return

            ink_inframe = ink_card[ly0:ly1, lx0:lx1]
            inframe_area = int(ink_inframe.sum())
            if inframe_area < MIN_INK_PIXELS_GLOBAL:
                return

            if (inframe_area / float(full_area)) < MIN_INFRAME_FRACTION:
                return

            occ_crop = occupancy[gy0:gy1, gx0:gx1]
            visible = ink_inframe & (~occ_crop)
            vis_area = int(visible.sum())
            if vis_area < MIN_INK_PIXELS_GLOBAL:
                return

            if (vis_area / float(full_area)) < MIN_VISIBLE_FRACTION:
                return

            ys, xs = np.where(ink_inframe)
            if xs.size == 0 or ys.size == 0:
                return

            x1 = int(xs.min()) + gx0
            y1 = int(ys.min()) + gy0
            x2 = int(xs.max()) + 1 + gx0
            y2 = int(ys.max()) + 1 + gy0
            annotations.append((class_id, float(x1), float(y1), float(x2), float(y2)))

        handle_corner_mask(p.get("tl_mask"))
        handle_corner_mask(p.get("br_mask"))

        occupancy[gy0:gy1, gx0:gx1] |= card_crop

    composite = np.array(bg)[:, :, ::-1].copy()
    return composite, annotations


def save_example(idx: int, img_bgr: np.ndarray, annotations: List[Tuple[int, float, float, float, float]]):
    img_h, img_w = img_bgr.shape[:2]
    img_path = CORNER_IMAGES_DIR / f"img_{idx:06d}.jpg"
    lbl_path = CORNER_LABELS_DIR / f"img_{idx:06d}.txt"

    cv2.imwrite(str(img_path), img_bgr)

    if not annotations:
        lbl_path.write_text("", encoding="utf-8")
        return

    lines = []
    for class_id, xmin, ymin, xmax, ymax in annotations:
        cx, cy, w, h = bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
        if w <= 1e-6 or h <= 1e-6:
            continue
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    lbl_path.write_text("\n".join(lines), encoding="utf-8")


def draw_debug(idx: int, img_bgr: np.ndarray, annotations: List[Tuple[int, float, float, float, float]], is_hard_neg: bool = False):
    out = img_bgr.copy()

    if is_hard_neg:
        cv2.putText(out, "HARD_NEG", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imwrite(str(DEBUG_DIR / f"debug_{idx:06d}.jpg"), out)
        return

    if not annotations:
        cv2.putText(out, "NO_LABEL", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imwrite(str(DEBUG_DIR / f"debug_{idx:06d}.jpg"), out)
        return

    for class_id, x1, y1, x2, y2 in annotations:
        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), (0, 0, 255), 2)
        cv2.putText(out, LABEL_ORDER[class_id], (x1i, max(0, y1i - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(DEBUG_DIR / f"debug_{idx:06d}.jpg"), out)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--debug_n", type=int, default=20)
    ap.add_argument("--hard_neg_prob", type=float, default=HARD_NEG_PROB_DEFAULT)

    # ROI/hull debug for generated cards
    ap.add_argument("--roi_debug", action="store_true", help="Save ROI rectangle + hull overlays per generated card")
    ap.add_argument("--roi_debug_max_imgs", type=int, default=200, help="Max composite images to dump per run (each may contain multiple cards)")
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dirs()

    backgrounds = load_backgrounds()
    if not backgrounds:
        print("[FATAL] Add backgrounds to data/backgrounds/")
        return

    card_assets = load_card_assets()
    available = _available_classes(card_assets)
    if not available:
        print("[FATAL] No card assets found in your STYLE_GROUP_DIRS.")
        return

    labeled_count = 0
    total_labels = 0
    hard_neg_count = 0

    roi_debug_budget = int(args.roi_debug_max_imgs)

    for i in tqdm(range(args.num), desc="Generating corner-mask dataset"):
        bg_img = random.choice(backgrounds)
        W, H = bg_img.size

        # --- hard negative branch ---
        if random.random() < args.hard_neg_prob:
            hard_neg_count += 1
            img_bgr = make_hard_negative_example(bg_img)
            img_bgr = apply_post_augs(img_bgr)
            save_example(i, img_bgr, [])
            if i < args.debug_n:
                draw_debug(i, img_bgr, [], is_hard_neg=True)
            continue

        num_cards = random.randint(NUM_CARDS_MIN, NUM_CARDS_MAX)
        placements = random_single_card_instances(card_assets, W, H, num_cards=num_cards)

        # --- fallback hard negative if placements failed ---
        if not placements:
            hard_neg_count += 1
            img_bgr = make_hard_negative_example(bg_img)
            img_bgr = apply_post_augs(img_bgr)
            save_example(i, img_bgr, [])
            if i < args.debug_n:
                draw_debug(i, img_bgr, [], is_hard_neg=True)
            continue

        # ---- per-generated-card ROI/hull debug (BEFORE composing) ----
        if args.roi_debug and roi_debug_budget > 0:
            for j, p in enumerate(placements):
                card_rgba = p["img"]
                roi_u8 = p.get("roi_rect_u8", None)
                tl_u8 = _mask_u8_from_bool(p.get("tl_mask", None))
                br_u8 = _mask_u8_from_bool(p.get("br_mask", None))

                vis_bgr = _draw_roi_and_hulls_on_rgba(card_rgba, roi_u8, tl_u8, br_u8)
                out_path = ROI_DEBUG_GEN_CARDS_DIR / f"gen_{i:06d}_card{j}_{p['cls']}.jpg"
                cv2.imwrite(str(out_path), vis_bgr)

            roi_debug_budget -= 1

        img_bgr, annotations = compose_on_background(bg_img, placements)
        img_bgr = apply_post_augs(img_bgr)

        if annotations:
            labeled_count += 1
            total_labels += len(annotations)

        save_example(i, img_bgr, annotations)
        if i < args.debug_n:
            draw_debug(i, img_bgr, annotations, is_hard_neg=False)

    print(f"[DONE] Generated {args.num} images")
    print(f"[SANITY] Images with >=1 label: {labeled_count}/{args.num}")
    if labeled_count > 0:
        print(f"[SANITY] Total corner labels: {total_labels} (avg {total_labels/labeled_count:.2f} per labeled image)")
    print(f"[SANITY] Debug images saved to: {DEBUG_DIR.resolve()}")
    print(f"[SANITY] ROI debug raw assets: {ROI_DEBUG_ASSETS_DIR.resolve()}")
    print(f"[SANITY] ROI debug generated cards: {ROI_DEBUG_GEN_CARDS_DIR.resolve()}")
    print(f"[SANITY] Mask cache saved to: {MASK_CACHE_DIR.resolve()} (version={CACHE_VERSION})")
    print(f"[SANITY] Hard negatives: {hard_neg_count}/{args.num} ({hard_neg_count/args.num:.2%})")


if __name__ == "__main__":
    main()
