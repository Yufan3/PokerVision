# src/dataset_gen/generate_dataset.py
import argparse
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

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
CACHE_VERSION = "v5b_bordercut_all_sides_hardneg_symbols_20251213"


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

for d in [CORNER_IMAGES_DIR, CORNER_LABELS_DIR, DEBUG_DIR, MASK_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# -----------------------
# Corner ROI fractions (raw card coords)
# Keep ROI big enough to contain BOTH rank+small suit for all ranks.
# -----------------------
TL_X0_F, TL_Y0_F = 0.01, 0.01
TL_X1_F, TL_Y1_F = 0.22, 0.28


# -----------------------
# Segmentation thresholds
# -----------------------
ALPHA_THR = 10  # ignore transparent pixels in assets and in composed alpha

# strict -> lenient
INK_DIST_THR_CANDIDATES = [14.0, 12.0, 10.0, 8.0]

# Small noise reject
MIN_COMP_AREA = 10
MIN_INK_PIXELS_ROI = 40

# Morphology
OPEN_ITERS = 0
CLOSE_ITERS_PRE = 1      # before subset selection
CLOSE_ITERS_POST = 1     # after selection (then re-select)
DILATE_ITERS = 0         # keep 0; dilation is the #1 reason it merges to pips/pattern


# -----------------------
# Subset-selection constraints (this is the critical part)
# -----------------------
TL_CX_MAX_FRAC = 0.92
TL_CY_MAX_FRAC = 0.94

# IMPORTANT CHANGE:
# Reject components that touch ANY ROI border within this many pixels.
# This removes “middle patterns” when the ROI border cuts through them.
TL_BORDER_PX = 2

UNION_MAX_AREA_FRAC = 0.94
UNION_MIN_H_FRAC = 0.20
UNION_MIN_W_FRAC = 0.12
UNION_MAX_X2_FRAC = 0.95
UNION_MAX_Y2_FRAC = 0.97


# -----------------------
# Label visibility thresholds (on composed image)
# -----------------------
MIN_INK_PIXELS_GLOBAL = 35
MIN_VISIBLE_FRACTION = 0.55


# -----------------------
# Placement distribution
# -----------------------
HARD_NEG_PROB_DEFAULT = 0.1
NUM_CARDS_MIN = 2
NUM_CARDS_MAX = 7


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


def _estimate_bg_lab(zone_bgra: np.ndarray) -> np.ndarray:
    """
    Estimate background paper LAB color from border pixels of ROI.
    Uses median of border pixels with alpha > ALPHA_THR.
    """
    h, w = zone_bgra.shape[:2]
    a = zone_bgra[:, :, 3]
    opaque = a > ALPHA_THR

    border = np.zeros((h, w), np.uint8)
    border[0:2, :] = 1
    border[h - 2:h, :] = 1
    border[:, 0:2] = 1
    border[:, w - 2:w] = 1
    border_mask = (border.astype(bool) & opaque)

    if border_mask.sum() < 30:
        border_mask = opaque

    bgr = zone_bgra[:, :, :3]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    vals = lab[border_mask]
    if vals.size == 0:
        return np.array([255, 128, 128], dtype=np.float32)
    return np.median(vals.reshape(-1, 3), axis=0).astype(np.float32)


def _union_bbox(comps: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    x1 = min(c[0] for c in comps)
    y1 = min(c[1] for c in comps)
    x2 = max(c[2] for c in comps)
    y2 = max(c[3] for c in comps)
    return x1, y1, x2, y2


def _bbox_area(b: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def _select_components_subset_tl(ink_u8: np.ndarray) -> np.ndarray:
    """
    Candidate components + greedy subset selection.
    Key rule: reject components that touch ANY ROI border (TL_BORDER_PX),
    so if ROI cuts through middle pattern, that component gets removed.
    """
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

        # ---- IMPORTANT CHANGE: reject if touching ANY ROI border ----
        if (x <= TL_BORDER_PX) or (y <= TL_BORDER_PX) or (x2 >= (W - TL_BORDER_PX)) or (y2 >= (H - TL_BORDER_PX)):
            continue

        # still keep near TL (loose)
        if cx >= TL_CX_MAX_FRAC * W or cy >= TL_CY_MAX_FRAC * H:
            continue

        cand.append((comp_id, area, x, y, x2, y2, float(cx), float(cy)))

    if not cand:
        return np.zeros_like(ink_u8)

    max_area = max(c[1] for c in cand)
    rel_keep = []
    for c in cand:
        if c[1] >= max(MIN_COMP_AREA, int(0.01 * max_area)):
            rel_keep.append(c)
    cand = rel_keep if rel_keep else cand

    roi_area = float(H * W)

    def score_candidate(c):
        _, area, _, _, _, _, cx, cy = c
        tl_pen = 0.65 * (cx / W) + 0.85 * (cy / H)
        return float(area) / (1.0 + tl_pen)

    cand_sorted = sorted(cand, key=score_candidate, reverse=True)

    selected: List[Tuple[int, int, int, int, int]] = []
    keep_mask = np.zeros_like(bin_mask)

    def subset_ok(b):
        x1, y1, x2, y2 = b
        bw = (x2 - x1)
        bh = (y2 - y1)
        if bw <= 0 or bh <= 0:
            return False
        area_frac = _bbox_area(b) / roi_area
        if area_frac > UNION_MAX_AREA_FRAC:
            return False
        if (x2 / W) > UNION_MAX_X2_FRAC:
            return False
        if (y2 / H) > UNION_MAX_Y2_FRAC:
            return False
        return True

    # start with best comp
    first = cand_sorted[0]
    comp_id, _, x, y, x2, y2, _, _ = first
    selected.append((comp_id, x, y, x2, y2))
    keep_mask[labels == comp_id] = 1

    # add more (suit often separate)
    for c in cand_sorted[1:]:
        comp_id, _, x, y, x2, y2, _, _ = c
        cur_boxes = [(s[1], s[2], s[3], s[4]) for s in selected]
        new_union = _union_bbox(cur_boxes + [(x, y, x2, y2)])

        if not subset_ok(new_union):
            continue

        cur_union = _union_bbox(cur_boxes)
        cur_area = _bbox_area(cur_union)
        new_area = _bbox_area(new_union)
        if new_area > cur_area * 2.3:
            continue

        selected.append((comp_id, x, y, x2, y2))
        keep_mask[labels == comp_id] = 1

        if len(selected) >= 4:
            break

    # validate union shape (avoid rank-only)
    sel_boxes = [(s[1], s[2], s[3], s[4]) for s in selected]
    union = _union_bbox(sel_boxes)
    bw = union[2] - union[0]
    bh = union[3] - union[1]

    if (bh / float(H)) < UNION_MIN_H_FRAC or (bw / float(W)) < UNION_MIN_W_FRAC:
        cur_union = union
        cur_boxes = sel_boxes
        best_rescue = None
        best_rescue_score = -1e9

        for c in cand_sorted[1:]:
            comp_id, area, x, y, x2, y2, cx, cy = c
            if cy <= (cur_union[1] + 0.10 * H):
                continue
            new_union = _union_bbox(cur_boxes + [(x, y, x2, y2)])
            if not subset_ok(new_union):
                continue
            grow = max(1.0, _bbox_area(new_union) - _bbox_area(cur_union))
            s = float(area) / grow
            if s > best_rescue_score:
                best_rescue_score = s
                best_rescue = (comp_id, x, y, x2, y2)

        if best_rescue is not None:
            comp_id, x, y, x2, y2 = best_rescue
            selected.append((comp_id, x, y, x2, y2))
            keep_mask[labels == comp_id] = 1
            sel_boxes = [(s[1], s[2], s[3], s[4]) for s in selected]
            union = _union_bbox(sel_boxes)
            bw = union[2] - union[0]
            bh = union[3] - union[1]

    if (bh / float(H)) < (UNION_MIN_H_FRAC * 0.85):
        return np.zeros_like(ink_u8)

    return (keep_mask.astype(np.uint8) * 255)


def _largest_reasonable_ink_mask(zone_bgra: np.ndarray) -> Optional[np.ndarray]:
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
        ink = (dist > thr) & (a > ALPHA_THR)
        ink_u8 = (ink.astype(np.uint8) * 255)

        if OPEN_ITERS > 0:
            ink_u8 = cv2.morphologyEx(ink_u8, cv2.MORPH_OPEN, kernel, iterations=OPEN_ITERS)
        if CLOSE_ITERS_PRE > 0:
            ink_u8 = cv2.morphologyEx(ink_u8, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERS_PRE)

        ink_u8 = _select_components_subset_tl(ink_u8)

        if CLOSE_ITERS_POST > 0:
            ink_u8 = cv2.morphologyEx(ink_u8, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERS_POST)
            ink_u8 = _select_components_subset_tl(ink_u8)

        if int((ink_u8 > 0).sum()) < MIN_INK_PIXELS_ROI:
            continue

        if DILATE_ITERS > 0:
            ink_u8 = cv2.dilate(ink_u8, kernel, iterations=DILATE_ITERS)

        ys, xs = np.where(ink_u8 > 0)
        if xs.size == 0 or ys.size == 0:
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        ink_area = int((ink_u8 > 0).sum())
        density = ink_area / float(bbox_area)
        obj = density + 0.00015 * ink_area

        if obj > best_obj:
            best_obj = obj
            best = ink_u8.copy()

    if best is not None:
        return best

    # Fallback: grayscale Otsu (rare)
    gray = cv2.cvtColor(zone_bgra[:, :, :3], cv2.COLOR_BGR2GRAY)
    opaque = (zone_bgra[:, :, 3] > ALPHA_THR).astype(np.uint8)
    if opaque.sum() < 30:
        return None

    g = gray.copy()
    g[opaque == 0] = int(np.median(g[opaque == 1]))
    _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu = (otsu & (opaque * 255)).astype(np.uint8)

    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
    otsu = _select_components_subset_tl(otsu)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
    otsu = _select_components_subset_tl(otsu)

    if int((otsu > 0).sum()) < MIN_INK_PIXELS_ROI:
        return None
    return otsu


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
    """
    Prefer fonts that exist on Windows and include suit glyphs.
    Fall back to DejaVu if available, then PIL default.
    """
    candidates = [
        # Windows (most reliable)
        r"C:\Windows\Fonts\seguisym.ttf",   # Segoe UI Symbol (♣♦♥♠)
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        # Common on linux/conda
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

def _render_text_patch(text: str, fs: int, rgb: Tuple[int, int, int]) -> Image.Image:
    """
    Render a tight RGBA patch for text. If glyph missing, alpha will be ~0.
    """
    font = _try_load_font(fs)

    tmp = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    d0 = ImageDraw.Draw(tmp)
    try:
        bbox = d0.textbbox((0, 0), text, font=font)
    except Exception:
        bbox = (0, 0, fs, fs)

    tw = max(1, bbox[2] - bbox[0])
    th = max(1, bbox[3] - bbox[1])
    pad = int(0.45 * fs)

    patch = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(patch)

    # draw a tiny shadow + main text to increase visibility
    ox = pad - bbox[0]
    oy = pad - bbox[1]
    d.text((ox + 2, oy + 2), text, font=font, fill=(0, 0, 0, 110))
    d.text((ox, oy), text, font=font, fill=(*rgb, 255))

    return patch

def _render_suit_vector(suit: str, fs: int, rgb: Tuple[int, int, int]) -> Image.Image:
    """
    Vector fallback for suits (works even if font can't render ♣♦♥♠).
    """
    w = int(1.2 * fs)
    h = int(1.35 * fs)
    patch = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(patch)

    cx = w // 2
    cy = int(h * 0.48)
    s = fs

    fill = (*rgb, 255)
    shadow = (0, 0, 0, 110)

    def ellipse(x0, y0, x1, y1, f):
        d.ellipse([x0, y0, x1, y1], fill=f)

    def poly(pts, f):
        d.polygon(pts, fill=f)

    # simple shapes; not perfect but very readable
    if suit == "♦":
        pts = [(cx, int(cy - 0.50*s)),
               (int(cx + 0.38*s), cy),
               (cx, int(cy + 0.50*s)),
               (int(cx - 0.38*s), cy)]
        poly([(x+2, y+2) for x, y in pts], shadow)
        poly(pts, fill)

    elif suit == "♥":
        r = int(0.22*s)
        ellipse(cx - r - 2, int(cy - 0.20*s) - r, cx - 2 + r, int(cy - 0.20*s) + r, shadow)
        ellipse(cx + 2 - r, int(cy - 0.20*s) - r, cx + 2 + r, int(cy - 0.20*s) + r, shadow)
        poly([(cx, int(cy + 0.55*s) + 2),
              (int(cx - 0.45*s) + 2, int(cy + 0.00*s) + 2),
              (int(cx + 0.45*s) + 2, int(cy + 0.00*s) + 2)], shadow)

        ellipse(cx - r - 2, int(cy - 0.20*s) - r, cx - 2 + r, int(cy - 0.20*s) + r, fill)
        ellipse(cx + 2 - r, int(cy - 0.20*s) - r, cx + 2 + r, int(cy - 0.20*s) + r, fill)
        poly([(cx, int(cy + 0.55*s)),
              (int(cx - 0.45*s), int(cy + 0.00*s)),
              (int(cx + 0.45*s), int(cy + 0.00*s))], fill)

    elif suit == "♠":
        r = int(0.22*s)
        ellipse(cx - r - 2, int(cy + 0.10*s) - r, cx - 2 + r, int(cy + 0.10*s) + r, shadow)
        ellipse(cx + 2 - r, int(cy + 0.10*s) - r, cx + 2 + r, int(cy + 0.10*s) + r, shadow)
        poly([(cx, int(cy - 0.60*s) + 2),
              (int(cx - 0.45*s) + 2, int(cy + 0.15*s) + 2),
              (int(cx + 0.45*s) + 2, int(cy + 0.15*s) + 2)], shadow)
        # stem
        poly([(int(cx - 0.10*s) + 2, int(cy + 0.15*s) + 2),
              (int(cx + 0.10*s) + 2, int(cy + 0.15*s) + 2),
              (cx + 2, int(cy + 0.55*s) + 2)], shadow)

        ellipse(cx - r - 2, int(cy + 0.10*s) - r, cx - 2 + r, int(cy + 0.10*s) + r, fill)
        ellipse(cx + 2 - r, int(cy + 0.10*s) - r, cx + 2 + r, int(cy + 0.10*s) + r, fill)
        poly([(cx, int(cy - 0.60*s)),
              (int(cx - 0.45*s), int(cy + 0.15*s)),
              (int(cx + 0.45*s), int(cy + 0.15*s))], fill)
        poly([(int(cx - 0.10*s), int(cy + 0.15*s)),
              (int(cx + 0.10*s), int(cy + 0.15*s)),
              (cx, int(cy + 0.55*s))], fill)

    else:  # ♣
        r = int(0.20*s)
        ellipse(cx - r - 2, int(cy - 0.10*s) - r, cx - 2 + r, int(cy - 0.10*s) + r, shadow)
        ellipse(cx + 2 - r, int(cy - 0.10*s) - r, cx + 2 + r, int(cy - 0.10*s) + r, shadow)
        ellipse(cx - r, int(cy - 0.38*s) - r, cx + r, int(cy - 0.38*s) + r, shadow)
        poly([(int(cx - 0.10*s) + 2, int(cy - 0.05*s) + 2),
              (int(cx + 0.10*s) + 2, int(cy - 0.05*s) + 2),
              (cx + 2, int(cy + 0.55*s) + 2)], shadow)

        ellipse(cx - r - 2, int(cy - 0.10*s) - r, cx - 2 + r, int(cy - 0.10*s) + r, fill)
        ellipse(cx + 2 - r, int(cy - 0.10*s) - r, cx + 2 + r, int(cy - 0.10*s) + r, fill)
        ellipse(cx - r, int(cy - 0.38*s) - r, cx + r, int(cy - 0.38*s) + r, fill)
        poly([(int(cx - 0.10*s), int(cy - 0.05*s)),
              (int(cx + 0.10*s), int(cy - 0.05*s)),
              (cx, int(cy + 0.55*s))], fill)

    return patch

def draw_fake_occluders(bg_img: Image.Image, num_shapes=4) -> Image.Image:
    """
    Place random SINGLE rank OR SINGLE suit symbol, no outline.
    Render via mask -> solid fill (clean edges).
    """
    bg = bg_img.convert("RGB").copy()
    W, H = bg.size

    for _ in range(num_shapes):
        color_choice = random.choice([
            (0, 0, 0),         # black
            (200, 0, 0),       # red
            (205, 170, 60),    # gold-ish
        ])

        # choose ONE symbol: rank OR suit
        text = random.choice(_RANKS) if random.random() < 0.5 else random.choice(_SUITS)

        # ---- bigger + more variable sizes (see next section too) ----
        fs = int(random.uniform(0.045, 0.16) * min(W, H))
        fs = max(18, min(fs, 170))
        font = _try_load_font(fs)

        # measure text bbox
        tmp = Image.new("L", (10, 10), 0)
        d0 = ImageDraw.Draw(tmp)
        bbox = d0.textbbox((0, 0), text, font=font)
        tw = max(1, bbox[2] - bbox[0])
        th = max(1, bbox[3] - bbox[1])

        pad = int(0.35 * fs)
        mw, mh = tw + 2 * pad, th + 2 * pad

        # ---- render text to mask (no outline) ----
        mask = Image.new("L", (mw, mh), 0)
        dm = ImageDraw.Draw(mask)
        dm.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=255)

        # ---- colorize using mask ----
        patch = Image.new("RGBA", (mw, mh), (0, 0, 0, 0))
        color_layer = Image.new("RGBA", (mw, mh), (*color_choice, 255))
        patch = Image.composite(color_layer, patch, mask)

        # rotate + paste
        angle = random.uniform(-40, 40)
        patch = patch.rotate(angle, expand=True, resample=Image.BICUBIC)

        px = random.randint(0, max(0, W - patch.size[0]))
        py = random.randint(0, max(0, H - patch.size[1]))
        bg.paste(patch, (px, py), patch)

    return bg

from typing import Union

def make_hard_negative_example(backgrounds: Union[List[Image.Image], Image.Image]) -> np.ndarray:
    if isinstance(backgrounds, Image.Image):
        bg = backgrounds
    else:
        bg = random.choice(backgrounds)

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
# Placements (rotate masks together with card)
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

        angle = random.uniform(-90, 90)

        card_rot = card_resized.rotate(angle, expand=True, resample=Image.BICUBIC)
        rot_w, rot_h = card_rot.size
        if rot_w >= W or rot_h >= H or rot_w < 5 or rot_h < 5:
            continue

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
            if cmask is None:
                return
            if cmask.shape != (rot_h, rot_w):
                return

            corner_crop = cmask[ly0:ly1, lx0:lx1]
            if corner_crop.size == 0:
                return

            ink_full = corner_crop & card_crop
            full_area = int(ink_full.sum())
            if full_area < MIN_INK_PIXELS_GLOBAL:
                return

            occ_crop = occupancy[gy0:gy1, gx0:gx1]
            visible = ink_full & (~occ_crop)
            vis_area = int(visible.sum())
            if vis_area < MIN_INK_PIXELS_GLOBAL:
                return
            if (vis_area / float(full_area)) < MIN_VISIBLE_FRACTION:
                return

            ys, xs = np.where(ink_full)
            if xs.size == 0 or ys.size == 0:
                return
            x1 = int(xs.min()) + gx0
            y1 = int(ys.min()) + gy0
            x2 = int(xs.max()) + 1 + gx0
            y2 = int(ys.max()) + 1 + gy0
            if x2 <= x1 or y2 <= y1:
                return

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
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.imwrite(str(DEBUG_DIR / f"debug_{idx:06d}.jpg"), out)
        return

    if not annotations:
        cv2.putText(out, "NO_LABEL", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.imwrite(str(DEBUG_DIR / f"debug_{idx:06d}.jpg"), out)
        return

    for class_id, x1, y1, x2, y2 in annotations:
        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), (0, 255, 255), 2)
        cv2.putText(out, LABEL_ORDER[class_id], (x1i, max(0, y1i - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(DEBUG_DIR / f"debug_{idx:06d}.jpg"), out)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=2500)
    ap.add_argument("--debug_n", type=int, default=60)
    ap.add_argument("--hard_neg_prob", type=float, default=HARD_NEG_PROB_DEFAULT)
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


    for i in tqdm(range(args.num), desc="Generating corner-mask dataset"):
        bg_img = random.choice(backgrounds)
        W, H = bg_img.size

        # --- hard negative branch ---
        if random.random() < args.hard_neg_prob:
            hard_neg_count += 1
            img_bgr = make_hard_negative_example(bg_img)
            img_bgr = apply_post_augs(img_bgr)
            save_example(i, img_bgr, [])  # no labels
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
    print(f"[SANITY] Mask cache saved to: {MASK_CACHE_DIR.resolve()} (version={CACHE_VERSION})")
    print(f"[SANITY] Hard negatives: {hard_neg_count}/{args.num} ({hard_neg_count/args.num:.2%})")


if __name__ == "__main__":
    main()
