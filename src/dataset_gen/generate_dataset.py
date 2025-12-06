import random
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
import math

from src.utils.config import (
    ensure_dirs,
    STYLE_GROUP_DIRS,
    STYLE_WEIGHTS,
    BACKGROUNDS_DIR,
    IMAGES_DIR,   # unused here but kept for compatibility
    LABELS_DIR,   # unused here but kept for compatibility
    CARD_CLASSES,  # should be the 52 logical cards: AS, 2S, ..., KH
)

from src.dataset_gen.yolo_label_utils import bbox_to_yolo

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
# Dataset paths (NEW)
# -----------------------

CORNER_BASE_DIR = Path("data/yolo_cards_corners")
CORNER_IMAGES_DIR = CORNER_BASE_DIR / "images"
CORNER_LABELS_DIR = CORNER_BASE_DIR / "labels"

CORNER_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
CORNER_LABELS_DIR.mkdir(parents=True, exist_ok=True)

# (seg paths kept but unused – safe to leave)
SEG_BASE_DIR = Path("data/yolo_cards_seg")
SEG_IMAGES_DIR = SEG_BASE_DIR / "images"
SEG_LABELS_DIR = SEG_BASE_DIR / "labels"
SEG_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
SEG_LABELS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------
# Geometry helpers
# -----------------------

def _rotate_full_card(new_w, new_h, angle_deg):
    """
    Return the 4 card corners after rotation with expand=True,
    in the rotated-image coordinate system (top-left at 0,0).
    Order is the same as input:
      [ [0,0], [new_w,0], [new_w,new_h], [0,new_h] ] -> rotated
    """
    full_corners = np.array([
        [0.0,   0.0   ],
        [new_w, 0.0   ],
        [new_w, new_h ],
        [0.0,   new_h ],
    ], dtype=np.float32)

    cx = new_w / 2.0
    cy = new_h / 2.0

    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)

    shifted_full = full_corners - np.array([[cx, cy]], dtype=np.float32)
    rotated_full = shifted_full @ R.T + np.array([[cx, cy]], dtype=np.float32)

    min_xy = rotated_full.min(axis=0, keepdims=True)
    rotated_full -= min_xy  # emulate expand=True placing top-left at (0,0)

    return rotated_full  # (4, 2)


def shift_placements_inside(placements, W, H, margin=2):
    """
    Shift a list of placements so that all bboxes lie fully inside [0, W] x [0, H].
    Each placement has:
        "bbox": (xmin, ymin, xmax, ymax)
        "tl_quad"/"br_quad": [(x, y), ...]  in global coordinates (if present).
    """
    if not placements:
        return placements

    min_x = min(p["bbox"][0] for p in placements)
    min_y = min(p["bbox"][1] for p in placements)
    max_x = max(p["bbox"][2] for p in placements)
    max_y = max(p["bbox"][3] for p in placements)

    if max_x - min_x > W - 2 * margin or max_y - min_y > H - 2 * margin:
        return placements

    dx = 0.0
    dy = 0.0

    if min_x < margin:
        dx = margin - min_x
    if max_x + dx > W - margin:
        dx += (W - margin) - (max_x + dx)

    if min_y < margin:
        dy = margin - min_y
    if max_y + dy > H - margin:
        dy += (H - margin) - (max_y + dy)

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return placements
    
    for p in placements:
        x1, y1, x2, y2 = p["bbox"]
        p["bbox"] = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        if "tl_box" in p and p["tl_box"] is not None:
            px1, py1, px2, py2 = p["tl_box"]
            p["tl_box"] = (px1 + dx, py1 + dy, px2 + dx, py2 + dy)
        if "br_box" in p and p["br_box"] is not None:
            px1, py1, px2, py2 = p["br_box"]
            p["br_box"] = (px1 + dx, py1 + dy, px2 + dx, py2 + dy)


    return placements


# -----------------------
# Helpers to load assets
# -----------------------

def load_card_variants():
    style_names = list(STYLE_GROUP_DIRS.keys())
    variants = {
        cls: {style: [] for style in style_names}
        for cls in CARD_CLASSES
    }

    for cls in CARD_CLASSES:
        for style, dir_list in STYLE_GROUP_DIRS.items():
            for d in dir_list:
                candidate = d / f"{cls}.png"
                if candidate.exists():
                    img = Image.open(candidate).convert("RGBA")
                    variants[cls][style].append(img)

    for cls, style_dict in variants.items():
        if all(len(imgs) == 0 for imgs in style_dict.values()):
            print(f"[WARN] No variants found for {cls}")

    return variants


def choose_card_image(card_variants_for_cls):
    available_styles = [
        s for s, imgs in card_variants_for_cls.items()
        if len(imgs) > 0
    ]
    if not available_styles:
        return None

    weights = [STYLE_WEIGHTS.get(s, 0.0) for s in available_styles]
    if sum(weights) <= 0:
        weights = [1.0] * len(available_styles)

    style_choice = random.choices(available_styles, weights=weights, k=1)[0]
    img = random.choice(card_variants_for_cls[style_choice]).copy()
    return img


def load_backgrounds():
    backgrounds = []
    for bg_path in BACKGROUNDS_DIR.iterdir():
        if bg_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            try:
                img = Image.open(bg_path).convert("RGB")
                backgrounds.append(img)
            except Exception as e:
                print(f"[WARN] Failed to load background {bg_path}: {e}")
    if len(backgrounds) == 0:
        print("[ERROR] No backgrounds found in data/backgrounds/")
    return backgrounds


def draw_fake_occluders(bg_img, num_shapes=3):
    bg = bg_img.copy()
    W, H = bg.size
    img = np.array(bg)

    for _ in range(num_shapes):
        w = random.randint(int(0.05 * W), int(0.25 * W))
        h = random.randint(int(0.05 * H), int(0.25 * H))
        x1 = random.randint(0, max(0, W - w))
        y1 = random.randint(0, max(0, H - h))
        x2 = x1 + w
        y2 = y1 + h

        if random.random() < 0.5:
            color = (
                random.randint(10, 50),
                random.randint(10, 50),
                random.randint(10, 50),
            )
        else:
            color = (
                random.randint(150, 220),
                random.randint(130, 200),
                random.randint(100, 180),
            )

        shape_type = random.choice(["rect"])
        if shape_type == "rect":
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=-1)
        else:
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = (w // 2, h // 2)
            cv2.ellipse(img, center, axes, angle=random.randint(0, 180),
                        startAngle=0, endAngle=360, color=color, thickness=-1)

    out = Image.fromarray(img)
    return out


def make_hard_negative_example(backgrounds):
    bg_img = random.choice(backgrounds)
    img_with_shapes = draw_fake_occluders(bg_img, num_shapes=random.randint(2, 6))
    img_bgr = np.array(img_with_shapes)[:, :, ::-1]  # RGB->BGR
    return img_bgr


# -----------------------
# Card placement routines
# -----------------------

def _available_classes(card_variants):
    return [
        cls for cls, style_dict in card_variants.items()
        if any(len(imgs) > 0 for imgs in style_dict.values())
    ]


# -----------------------
# Corner bbox via mask-rotation  (robust, matches PIL exactly)
# -----------------------

def _compute_corner_bboxes(new_w, new_h, angle_deg, xmin, ymin):
    """
    Compute TL/BR corner bounding boxes in GLOBAL coords.

    Steps:
      1) Create binary masks for TL/BR corner rectangles in the *unrotated* card.
      2) Rotate each mask with the same PIL.rotate(angle, expand=True).
      3) From each rotated mask, read the bbox (in rotated-card coords).
      4) Shift by (xmin, ymin) to get global coords.
    Returns:
      tl_box, br_box where each is (x1, y1, x2, y2) or None.
    """

    # ---- define corner rectangles in unrotated card coords ----
    # Tighter fractions so boxes hug the rank+suit cluster more closely.
    tl_margin_x_frac = 0.030   # a bit more margin from edge
    tl_margin_y_frac = 0.032
    tl_w_frac        = 0.14   # narrower
    tl_h_frac        = 0.24   # shorter

    br_margin_x_frac = 0.030
    br_margin_y_frac = 0.032
    br_w_frac        = 0.14
    br_h_frac        = 0.24

    tl_x0 = int(round(tl_margin_x_frac * new_w))
    tl_y0 = int(round(tl_margin_y_frac * new_h))
    tl_x1 = int(round(tl_x0 + tl_w_frac * new_w))
    tl_y1 = int(round(tl_y0 + tl_h_frac * new_h))

    br_x1 = int(round(new_w - br_margin_x_frac * new_w))
    br_y1 = int(round(new_h - br_margin_y_frac * new_h))
    br_x0 = int(round(br_x1 - br_w_frac * new_w))
    br_y0 = int(round(br_y1 - br_h_frac * new_h))

    # clamp inside card
    tl_x0, tl_y0 = max(tl_x0, 0), max(tl_y0, 0)
    tl_x1, tl_y1 = min(tl_x1, new_w), min(tl_y1, new_h)
    br_x0, br_y0 = max(br_x0, 0), max(br_y0, 0)
    br_x1, br_y1 = min(br_x1, new_w), min(br_y1, new_h)

    tl_mask = np.zeros((new_h, new_w), dtype=np.uint8)
    br_mask = np.zeros((new_h, new_w), dtype=np.uint8)

    if tl_x1 > tl_x0 and tl_y1 > tl_y0:
        tl_mask[tl_y0:tl_y1, tl_x0:tl_x1] = 255
    if br_x1 > br_x0 and br_y1 > br_y0:
        br_mask[br_y0:br_y1, br_x0:br_x1] = 255

    angle = float(angle_deg)
    tl_mask_rot = Image.fromarray(tl_mask).rotate(angle, expand=True)
    br_mask_rot = Image.fromarray(br_mask).rotate(angle, expand=True)

    tl_mask_rot_np = np.array(tl_mask_rot)
    br_mask_rot_np = np.array(br_mask_rot)

    def mask_to_box(mask_np):
        ys, xs = np.where(mask_np > 0)
        if xs.size == 0 or ys.size == 0:
            return None
        x1 = xs.min()
        y1 = ys.min()
        x2 = xs.max()
        y2 = ys.max()
        return x1, y1, x2, y2

    tl_local_box = mask_to_box(tl_mask_rot_np)
    br_local_box = mask_to_box(br_mask_rot_np)

    def to_global(local_box):
        if local_box is None:
            return None
        x1, y1, x2, y2 = local_box
        return float(x1 + xmin), float(y1 + ymin), float(x2 + xmin), float(y2 + ymin)

    tl_box_global = to_global(tl_local_box)
    br_box_global = to_global(br_local_box)

    return tl_box_global, br_box_global



def hard_fan_instances(card_variants, W, H):
    placements = []

    available_classes = _available_classes(card_variants)
    if len(available_classes) == 0:
        return placements

    k = random.randint(3, 5)
    chosen_classes = random.sample(available_classes, k=min(k, len(available_classes)))

    base_images = []
    for c in chosen_classes:
        img = choose_card_image(card_variants[c])
        base_images.append(img)

    if any(img is None for img in base_images):
        return placements

    fan_frac_min = 0.18
    fan_frac_max = 0.22
    fan_width_fraction = random.uniform(fan_frac_min, fan_frac_max)
    target_card_w = int(W * fan_width_fraction)

    # fan geometry
    start_angle = random.uniform(-30, 30)
    step_deg = random.uniform(9, 12)

    anchor_x = random.randint(int(0.35 * W), int(0.65 * W))
    anchor_y = random.randint(int(0.45 * H), int(0.80 * H))

    for i, (cls, img_raw) in enumerate(zip(chosen_classes, base_images)):
        orig_w, orig_h = img_raw.size
        if orig_w == 0 or orig_h == 0:
            continue

        # resize card
        scale = target_card_w / float(orig_w)
        new_w = target_card_w
        new_h = int(orig_h * scale)
        card_img = img_raw.resize((new_w, new_h), Image.BICUBIC)

        angle = start_angle - i * step_deg

        # ---- build rotation transform matching PIL.rotate(expand=True) ----
        full_corners = np.array([
            [0.0,   0.0],
            [new_w, 0.0],
            [new_w, new_h],
            [0.0,   new_h],
        ], dtype=np.float32)

        cx = new_w / 2.0
        cy = new_h / 2.0

        theta = math.radians(-angle)
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s],
                      [s,  c]], dtype=np.float32)

        # rotate full card to get min_xy (for expand=True)
        shifted_full = full_corners - np.array([[cx, cy]], dtype=np.float32)
        rotated_full = shifted_full @ R.T + np.array([[cx, cy]], dtype=np.float32)
        min_xy = rotated_full.min(axis=0, keepdims=True)

        # size of rotated card
        rotated_full -= min_xy
        rot_w = float(rotated_full[:, 0].max())
        rot_h = float(rotated_full[:, 1].max())

        # rotate the *grip point* at (0.6, 0.75) in local card coords
        grip_local = np.array([[0.65 * new_w, 0.8 * new_h]], dtype=np.float32)
        shifted_grip = grip_local - np.array([[cx, cy]], dtype=np.float32)
        rotated_grip = shifted_grip @ R.T + np.array([[cx, cy]], dtype=np.float32)
        rotated_grip -= min_xy
        grip_rot_x, grip_rot_y = float(rotated_grip[0, 0]), float(rotated_grip[0, 1])

        # actually rotate the image with PIL
        rotated = card_img.rotate(angle, expand=True)
        # (rot_w, rot_h) should match rotated.size; enforce that
        rot_w, rot_h = rotated.size

        # back cards slightly shifted from the front around grip point
        dist_from_front = (k - 1) - i
        fan_offset_x = -dist_from_front * (0.04 * new_w)
        fan_offset_y = -dist_from_front * (0.03 * new_h)

        # place so that rotated grip sits at anchor + offset
        xmin = int(anchor_x + fan_offset_x - grip_rot_x)
        ymin = int(anchor_y + fan_offset_y - grip_rot_y)
        xmax = xmin + rot_w
        ymax = ymin + rot_h

        # compute TL/BR corner boxes in global coords
        tl_box, br_box = _compute_corner_bboxes(new_w, new_h, angle, xmin, ymin)

        placements.append({
            "cls": cls,
            "img": rotated,
            "bbox": (xmin, ymin, xmax, ymax),
            "tl_box": tl_box,
            "br_box": br_box,
        })

    # require fully inside
    for p in placements:
        x1, y1, x2, y2 = p["bbox"]
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            return []

    placements = shift_placements_inside(placements, W, H)
    return placements


def random_single_card_instances(card_variants, W, H, num_cards):
    placements = []

    available_classes = _available_classes(card_variants)
    if len(available_classes) == 0:
        return placements

    chosen_classes = random.sample(
        available_classes,
        k=min(num_cards, len(available_classes))
    )

    for cls in chosen_classes:
        base_img = choose_card_image(card_variants[cls])
        if base_img is None:
            continue

        orig_w, orig_h = base_img.size
        if orig_w == 0 or orig_h == 0:
            continue

        r = random.random()
        if r < 0.35:
            frac_min, frac_max = 0.18, 0.30
        else:
            frac_min, frac_max = 0.12, 0.18

        target_frac = random.uniform(frac_min, frac_max)
        target_w = int(W * target_frac)

        scale = target_w / float(orig_w)
        new_w = target_w
        new_h = int(orig_h * scale)

        card_img = base_img.resize((new_w, new_h), Image.BICUBIC)
        angle = random.uniform(-90, 90)

        rotated = card_img.rotate(angle, expand=True)
        rot_w, rot_h = rotated.size

        if rot_w >= W or rot_h >= H:
            continue

        xmin = random.randint(0, W - rot_w)
        ymin = random.randint(0, H - rot_h)
        xmax = xmin + rot_w
        ymax = ymin + rot_h

        tl_box, br_box = _compute_corner_bboxes(new_w, new_h, angle, xmin, ymin)

        placements.append({
            "cls": cls,
            "img": rotated,
            "bbox": (xmin, ymin, xmax, ymax),
            "tl_box": tl_box,
            "br_box": br_box,
        })


    return placements


def fan_cluster_instances(card_variants, W, H):
    """
    Create a realistic fan of cards where all cards share a common grip point
    around (0.6, 0.75) in card-local coordinates (TL=(0,0), BR=(1,1)).

    The grip point is rotated with the same transform as the card image
    (PIL.rotate(..., expand=True)), then aligned to (anchor_x + fan_offset_x,
    anchor_y + fan_offset_y) in the global image.
    """
    placements = []

    available_classes = _available_classes(card_variants)
    if len(available_classes) == 0:
        return placements

    k = random.randint(3, 5)
    chosen_classes = random.sample(available_classes, k=min(k, len(available_classes)))

    base_images = []
    for c in chosen_classes:
        img = choose_card_image(card_variants[c])
        base_images.append(img)

    if any(img is None for img in base_images):
        return placements

    fan_frac_min = 0.19
    fan_frac_max = 0.21
    fan_width_fraction = random.uniform(fan_frac_min, fan_frac_max)
    target_card_w = int(W * fan_width_fraction)

    # fan geometry
    start_angle = random.uniform(-30, 30)
    step_deg = random.uniform(12, 14)

    anchor_x = random.randint(int(0.35 * W), int(0.65 * W))
    anchor_y = random.randint(int(0.45 * H), int(0.80 * H))

    for i, (cls, img_raw) in enumerate(zip(chosen_classes, base_images)):
        orig_w, orig_h = img_raw.size
        if orig_w == 0 or orig_h == 0:
            continue

        # resize card
        scale = target_card_w / float(orig_w)
        new_w = target_card_w
        new_h = int(orig_h * scale)
        card_img = img_raw.resize((new_w, new_h), Image.BICUBIC)

        angle = start_angle - i * step_deg

        # ---- build rotation transform matching PIL.rotate(expand=True) ----
        full_corners = np.array([
            [0.0,   0.0],
            [new_w, 0.0],
            [new_w, new_h],
            [0.0,   new_h],
        ], dtype=np.float32)

        cx = new_w / 2.0
        cy = new_h / 2.0

        theta = math.radians(-angle)
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s],
                      [s,  c]], dtype=np.float32)

        # rotate full card to get min_xy (for expand=True)
        shifted_full = full_corners - np.array([[cx, cy]], dtype=np.float32)
        rotated_full = shifted_full @ R.T + np.array([[cx, cy]], dtype=np.float32)
        min_xy = rotated_full.min(axis=0, keepdims=True)

        # size of rotated card
        rotated_full -= min_xy
        rot_w = float(rotated_full[:, 0].max())
        rot_h = float(rotated_full[:, 1].max())

        # rotate the *grip point* at (0.6, 0.75) in local card coords
        grip_local = np.array([[0.65 * new_w, 0.8 * new_h]], dtype=np.float32)
        shifted_grip = grip_local - np.array([[cx, cy]], dtype=np.float32)
        rotated_grip = shifted_grip @ R.T + np.array([[cx, cy]], dtype=np.float32)
        rotated_grip -= min_xy
        grip_rot_x, grip_rot_y = float(rotated_grip[0, 0]), float(rotated_grip[0, 1])

        # actually rotate the image with PIL
        rotated = card_img.rotate(angle, expand=True)
        # (rot_w, rot_h) should match rotated.size; enforce that
        rot_w, rot_h = rotated.size

        # back cards slightly shifted from the front around grip point
        dist_from_front = (k - 1) - i
        fan_offset_x = -dist_from_front * (0.04 * new_w)
        fan_offset_y = -dist_from_front * (0.03 * new_h)

        # place so that rotated grip sits at anchor + offset
        xmin = int(anchor_x + fan_offset_x - grip_rot_x)
        ymin = int(anchor_y + fan_offset_y - grip_rot_y)
        xmax = xmin + rot_w
        ymax = ymin + rot_h

        # compute TL/BR corner boxes in global coords
        tl_box, br_box = _compute_corner_bboxes(new_w, new_h, angle, xmin, ymin)

        placements.append({
            "cls": cls,
            "img": rotated,
            "bbox": (xmin, ymin, xmax, ymax),
            "tl_box": tl_box,
            "br_box": br_box,
        })

    # require fully inside
    for p in placements:
        x1, y1, x2, y2 = p["bbox"]
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            return []

    placements = shift_placements_inside(placements, W, H)
    return placements

# -----------------------
# Compose & labels
# -----------------------

def _poly_to_mask(H, W, poly):
    """Rasterize a polygon (list of (x,y)) into a boolean mask."""
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = np.round(np.array(poly, dtype=np.float32)).astype(np.int32)
    if pts.ndim == 2:
        pts = pts.reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def compose_on_background(bg_img, placements):
    """
    Produce:
      composite_bgr: HxWx3 uint8
      annotations:   list of (class_id, xmin, ymin, xmax, ymax)

    Strategy:
      1) Paste all cards back->front with RGBA alpha.
      2) Iterate front->back, computing card alpha masks.
         For each TL/BR box, check how much of it is actually visible
         (not occluded by cards in front). Only keep boxes where
         visible_fraction >= MIN_CORNER_FRACTION.
    """
    bg = bg_img.copy()
    W, H = bg.size

    # 1) Paste cards back->front
    for p in placements:
        card_img = p["img"]  # PIL RGBA
        xmin, ymin, xmax, ymax = p["bbox"]

        vis_xmin = max(int(xmin), 0)
        vis_ymin = max(int(ymin), 0)
        vis_xmax = min(int(xmax), W)
        vis_ymax = min(int(ymax), H)

        if vis_xmin >= vis_xmax or vis_ymin >= vis_ymax:
            continue

        crop_left   = vis_xmin - xmin
        crop_top    = vis_ymin - ymin
        crop_right  = crop_left + (vis_xmax - vis_xmin)
        crop_bottom = crop_top  + (vis_ymax - vis_ymin)

        card_visible = card_img.crop((crop_left, crop_top, crop_right, crop_bottom))
        bg.paste(card_visible, (vis_xmin, vis_ymin), card_visible)

    # 2) Visibility-based corner selection (front->back)
    occupancy = np.zeros((H, W), dtype=bool)  # pixels already covered by front cards
    annotations = []

    MIN_CORNER_PIXELS   = 30     # absolute minimum area in pixels
    MIN_CORNER_FRACTION = 0.45   # at least 45% of corner box must be visible

    for p in reversed(placements):  # front card first
        cls      = p["cls"]
        card_img = p["img"]
        xmin, ymin, xmax, ymax = p["bbox"]

        vis_xmin = max(int(xmin), 0)
        vis_ymin = max(int(ymin), 0)
        vis_xmax = min(int(xmax), W)
        vis_ymax = min(int(ymax), H)

        if vis_xmin >= vis_xmax or vis_ymin >= vis_ymax:
            continue

        crop_left   = vis_xmin - xmin
        crop_top    = vis_ymin - ymin
        crop_right  = crop_left + (vis_xmax - vis_xmin)
        crop_bottom = crop_top  + (vis_ymax - vis_ymin)

        card_visible = card_img.crop((crop_left, crop_top, crop_right, crop_bottom))
        if card_visible.mode != "RGBA":
            continue

        alpha = np.array(card_visible.split()[-1])
        mask_local = alpha > 10
        if not mask_local.any():
            continue

        h_vis, w_vis = mask_local.shape
        card_mask_full = np.zeros((H, W), dtype=bool)
        card_mask_full[vis_ymin:vis_ymin + h_vis,
                       vis_xmin:vis_xmin + w_vis] = mask_local

        if not card_mask_full.any():
            continue

        class_id = LABEL_ORDER.index(cls)
        tl_box = p.get("tl_box")
        br_box = p.get("br_box")

        def handle_corner(box):
            if box is None:
                return
            x1, y1, x2, y2 = box

            # clamp to image & int
            x1 = max(0, min(W - 1, int(round(x1))))
            y1 = max(0, min(H - 1, int(round(y1))))
            x2 = max(0, min(W - 1, int(round(x2))))
            y2 = max(0, min(H - 1, int(round(y2))))

            if x2 <= x1 or y2 <= y1:
                return

            # corner area within this card
            corner_mask = card_mask_full[y1:y2, x1:x2]
            full_area = int(corner_mask.sum())
            if full_area < MIN_CORNER_PIXELS:
                return

            # visible = not occluded by cards in front
            visible_mask = corner_mask & (~occupancy[y1:y2, x1:x2])
            vis_area = int(visible_mask.sum())
            if vis_area < MIN_CORNER_PIXELS:
                return

            if vis_area / full_area < MIN_CORNER_FRACTION:
                return

            # passed visibility checks → keep bbox
            annotations.append((class_id, float(x1), float(y1),
                                float(x2), float(y2)))

        # TL & BR
        handle_corner(tl_box)
        handle_corner(br_box)

        # mark this whole card as occupied for deeper cards
        occupancy |= card_mask_full

    composite = np.array(bg)[:, :, ::-1]  # RGB -> BGR
    return composite, annotations


# -----------------------
# realism augs
# -----------------------

def apply_post_augs(img_bgr):
    out = img_bgr.astype(np.float32)

    if random.random() < 0.9:
        alpha = random.uniform(0.6, 1.3)
        beta = random.uniform(-45, 35)
        out = out * alpha + beta

    if random.random() < 0.7:
        rb = random.uniform(0.9, 1.1)
        gb = random.uniform(0.9, 1.1)
        bb = random.uniform(0.9, 1.1)
        color_gain = np.array([bb, gb, rb], dtype=np.float32).reshape(1, 1, 3)
        out = out * color_gain

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
        noise = np.random.normal(0, sigma, out.shape).astype(np.float32)
        out = out + noise

    if random.random() < 0.6:
        quality = random.randint(25, 70)
        enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ok, enc = cv2.imencode(".jpg", np.clip(out, 0, 255).astype(np.uint8), enc_param)
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32)

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def save_example(idx, img_bgr, annotations):
    """
    Save detection-style YOLO labels for 52 card classes.
    """
    img_h, img_w = img_bgr.shape[:2]

    img_path = CORNER_IMAGES_DIR / f"img_{idx:06d}.jpg"
    lbl_path = CORNER_LABELS_DIR / f"img_{idx:06d}.txt"

    cv2.imwrite(str(img_path), img_bgr)

    if not annotations:
        with open(lbl_path, "w") as f:
            pass
        return

    lines = []
    for class_id, xmin, ymin, xmax, ymax in annotations:
        cx, cy, w, h = bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(lbl_path, "w") as f:
        f.write("\n".join(lines))


# -----------------------
# main loop
# -----------------------

def main():
    ensure_dirs()

    card_variants = load_card_variants()
    backgrounds = load_backgrounds()

    if len(backgrounds) == 0:
        print("[FATAL] You have no backgrounds in data/backgrounds/. Add images first.")
        return

    if all(all(len(imgs) == 0 for imgs in style_dict.values())
           for style_dict in card_variants.values()):
        print("[FATAL] You have no cards in data/raw_cards/(normal|inverted|real).")
        return

    num_samples = 25  # scale up for real training
    HARD_NEG_PROB = 0.08   # 8% pure clutter/background
    HARD_POS_PROB = 0.25   # 25% hard fans

    for i in tqdm(range(num_samples), desc="Generating corner-based dataset"):
        bg_img = random.choice(backgrounds)
        W, H = bg_img.size

        mode_r = random.random()

        # 1) hard negative: background + occluders, NO labels
        if mode_r < HARD_NEG_PROB:
            img_bgr = make_hard_negative_example(backgrounds)
            img_bgr = apply_post_augs(img_bgr)
            save_example(i, img_bgr, [])
            continue

        # 2) hard fans near edges
        elif mode_r < HARD_NEG_PROB + HARD_POS_PROB:
            placements = hard_fan_instances(card_variants, W, H)
            extra = random_single_card_instances(card_variants, W, H,
                                                 num_cards=random.randint(0, 1))
            placements.extend(extra)

        # 3) normal mixture
        else:
            if random.random() < 0.75:
                placements = fan_cluster_instances(card_variants, W, H)
                extra = random_single_card_instances(card_variants, W, H,
                                                     num_cards=random.randint(0, 2))
                placements.extend(extra)
            else:
                placements = random_single_card_instances(card_variants, W, H,
                                                          num_cards=random.randint(2, 7))

        if not placements:
            img_bgr = make_hard_negative_example(backgrounds)
            img_bgr = apply_post_augs(img_bgr)
            save_example(i, img_bgr, [])
            continue

        img_bgr, annotations = compose_on_background(bg_img, placements)
        img_bgr = apply_post_augs(img_bgr)
        save_example(i, img_bgr, annotations)

    print("Done generating 52-class corner-based dataset.")


if __name__ == "__main__":
    main()