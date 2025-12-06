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
        if "tl_quad" in p and p["tl_quad"] is not None:
            p["tl_quad"] = [(px + dx, py + dy) for (px, py) in p["tl_quad"]]
        if "br_quad" in p and p["br_quad"] is not None:
            p["br_quad"] = [(px + dx, py + dy) for (px, py) in p["br_quad"]]

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
                random.randint(10, 40),
                random.randint(10, 40),
                random.randint(10, 40),
            )
        else:
            color = (
                random.randint(150, 220),
                random.randint(130, 200),
                random.randint(100, 180),
            )

        shape_type = random.choice(["rect", "ellipse"])
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


def _compute_corner_quads(new_w, new_h, angle_deg, xmin, ymin):
    """
    Given card size, rotation (degrees), and top-left placement (xmin, ymin)
    of the *rotated* card image, compute global TL/BR corner quads.

    Corner regions are defined in card-local coords, then rotated using
    the same transform as PIL.rotate(..., expand=True), then shifted by (xmin, ymin).
    """

    # --- 1) define corner rectangles in card-local coords ---
    # You can tweak these fractions if you want slightly bigger / smaller patches.
    tl_w_frac, tl_h_frac = 0.20, 0.286
    br_w_frac, br_h_frac = 0.20, 0.286

    tl_w = tl_w_frac * new_w
    tl_h = tl_h_frac * new_h

    br_w = br_w_frac * new_w
    br_h = br_h_frac * new_h

    tl_local = np.array([
        [0.0,    0.0   ],
        [tl_w,   0.0   ],
        [tl_w,   tl_h  ],
        [0.0,    tl_h  ],
    ], dtype=np.float32)

    br_local = np.array([
        [new_w - br_w, new_h - br_h],
        [new_w,        new_h - br_h],
        [new_w,        new_h       ],
        [new_w - br_w, new_h       ],
    ], dtype=np.float32)

    # --- 2) build the same rotation/expand transform as PIL ---
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

    # rotate full card to find min_xy (for expand=True)
    shifted_full = full_corners - np.array([[cx, cy]], dtype=np.float32)
    rotated_full = shifted_full @ R.T + np.array([[cx, cy]], dtype=np.float32)
    min_xy = rotated_full.min(axis=0, keepdims=True)

    def rot_local_pts(pts_local):
        shifted = pts_local - np.array([[cx, cy]], dtype=np.float32)
        rot = shifted @ R.T + np.array([[cx, cy]], dtype=np.float32)
        rot -= min_xy  # top-left of rotated card at (0,0)
        return rot

    tl_rot = rot_local_pts(tl_local)
    br_rot = rot_local_pts(br_local)

    # --- 3) shift into global image coords ---
    tl_global = [(float(px + xmin), float(py + ymin)) for (px, py) in tl_rot]
    br_global = [(float(px + xmin), float(py + ymin)) for (px, py) in br_rot]

    return tl_global, br_global


def hard_fan_instances(card_variants, W, H):
    placements = []

    available_classes = _available_classes(card_variants)
    if not available_classes:
        return placements

    k = random.randint(3, 6)
    chosen_classes = random.sample(available_classes, k=min(k, len(available_classes)))

    base_images = []
    for c in chosen_classes:
        img = choose_card_image(card_variants[c])
        base_images.append(img)

    if any(img is None for img in base_images):
        return placements

    base_card_width = base_images[0].size[0]
    scale_base = (W / base_card_width) * 0.22
    scale_factor = random.uniform(1.0, 1.4)
    scale = scale_base * scale_factor

    start_angle = random.uniform(-40, 40)
    step_deg = random.uniform(4, 8)

    anchor_x = random.randint(int(0.50 * W), int(0.90 * W))
    anchor_y = random.randint(int(0.76 * H), int(0.90 * H))

    for i, (cls, img_raw) in enumerate(zip(chosen_classes, base_images)):
        orig_w, orig_h = img_raw.size
        if orig_w == 0 or orig_h == 0:
            continue

        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        card_img = img_raw.resize((new_w, new_h), Image.BICUBIC)

        angle = start_angle - i * step_deg

        # compute rotated full-card corners to get "grip" in rotated coords
        whole_rot = _rotate_full_card(new_w, new_h, angle)
        rot_w = whole_rot[:, 0].max() - whole_rot[:, 0].min()
        rot_h = whole_rot[:, 1].max() - whole_rot[:, 1].min()

        rotated = card_img.rotate(angle, expand=True)
        rot_w, rot_h = rotated.size  # ensures consistency with PIL

        # grip = bottom-right vertex in rotated coords
        xs = whole_rot[:, 0]
        ys = whole_rot[:, 1]
        br_idx = np.argmax(xs + ys)
        grip_rot_x, grip_rot_y = whole_rot[br_idx]

        dist_from_front = (k - 1) - i
        fan_offset_x = -dist_from_front * (0.10 * new_w)
        fan_offset_y = -dist_from_front * (0.04 * new_h)

        xmin = int(anchor_x + fan_offset_x - grip_rot_x)
        ymin = int(anchor_y + fan_offset_y - grip_rot_y)
        xmax = xmin + rot_w
        ymax = ymin + rot_h

        tl_quad, br_quad = _compute_corner_quads(new_w, new_h, angle, xmin, ymin)

        placements.append({
            "cls": cls,
            "img": rotated,
            "bbox": (xmin, ymin, xmax, ymax),
            "tl_quad": tl_quad,
            "br_quad": br_quad,
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
            frac_min, frac_max = 0.20, 0.32
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

        tl_quad, br_quad = _compute_corner_quads(new_w, new_h, angle, xmin, ymin)

        placements.append({
            "cls": cls,
            "img": rotated,
            "bbox": (xmin, ymin, xmax, ymax),
            "tl_quad": tl_quad,
            "br_quad": br_quad,
        })

    return placements


def fan_cluster_instances(card_variants, W, H):
    placements = []

    available_classes = _available_classes(card_variants)
    if len(available_classes) == 0:
        return placements

    k = random.randint(2, 5)
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

    start_angle = random.uniform(-20, 30)
    step_deg = random.uniform(9, 16)

    anchor_x = random.randint(int(0.45 * W), int(0.85 * W))
    anchor_y = random.randint(int(0.55 * H), int(0.95 * H))

    for i, (cls, img_raw) in enumerate(zip(chosen_classes, base_images)):
        orig_w, orig_h = img_raw.size
        if orig_w == 0 or orig_h == 0:
            continue

        scale = target_card_w / float(orig_w)
        new_w = target_card_w
        new_h = int(orig_h * scale)
        card_img = img_raw.resize((new_w, new_h), Image.BICUBIC)

        angle = start_angle - i * step_deg

        whole_rot = _rotate_full_card(new_w, new_h, angle)
        rotated = card_img.rotate(angle, expand=True)
        rot_w, rot_h = rotated.size

        # grip = bottom-right in rotated coords
        xs = whole_rot[:, 0]
        ys = whole_rot[:, 1]
        br_idx = np.argmax(xs + ys)
        grip_rot_x, grip_rot_y = whole_rot[br_idx]

        dist_from_front = (k - 1) - i
        fan_offset_x = -dist_from_front * (0.08 * new_w)
        fan_offset_y = -dist_from_front * (0.03 * new_h)

        xmin = int(anchor_x + fan_offset_x - grip_rot_x)
        ymin = int(anchor_y + fan_offset_y - grip_rot_y)
        xmax = xmin + rot_w
        ymax = ymin + rot_h

        tl_quad, br_quad = _compute_corner_quads(new_w, new_h, angle, xmin, ymin)

        placements.append({
            "cls": cls,
            "img": rotated,
            "bbox": (xmin, ymin, xmax, ymax),
            "tl_quad": tl_quad,
            "br_quad": br_quad,
        })

    for p in placements:
        x1, y1, x2, y2 = p["bbox"]
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            return []

    placements = shift_placements_inside(placements, W, H)
    return placements


# -----------------------
# Compose & labels
# -----------------------

def quad_to_bbox_clamped(quad, W, H, min_size=4):
    """
    Given a list of 4 (x, y) points in GLOBAL coords, return an axis-aligned
    bbox (xmin, ymin, xmax, ymax) clamped inside the image [0,W) x [0,H).
    Returns None if the box is degenerate or too small.
    """
    xs = np.array([p[0] for p in quad], dtype=np.float32)
    ys = np.array([p[1] for p in quad], dtype=np.float32)

    xmin = float(xs.min())
    xmax = float(xs.max())
    ymin = float(ys.min())
    ymax = float(ys.max())

    # clamp to image
    xmin = max(0.0, min(float(W - 1), xmin))
    ymin = max(0.0, min(float(H - 1), ymin))
    xmax = max(0.0, min(float(W - 1), xmax))
    ymax = max(0.0, min(float(H - 1), ymax))

    if xmax <= xmin or ymax <= ymin:
        return None

    if (xmax - xmin) < min_size or (ymax - ymin) < min_size:
        return None

    return xmin, ymin, xmax, ymax


def compose_on_background(bg_img, placements):
    """
    Produce:
      composite_bgr: HxWx3 uint8
      annotations:   list of (class_id, xmin, ymin, xmax, ymax)

    - We paste all cards (back->front) using RGBA alpha.
    - For each card, we use its precomputed corner quads (tl_quad / br_quad)
      to build tight axis-aligned boxes for TL / BR corner patches.

    NO alpha-based occupancy tricks here: just pure geometry.
    This guarantees the boxes are glued to the corners for any rotation.
    """
    bg = bg_img.copy()
    W, H = bg.size

    # 1) draw cards back->front in the order they appear
    for p in placements:
        card_img = p["img"]  # PIL RGBA
        xmin, ymin, xmax, ymax = p["bbox"]

        # clip bbox to image
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

    # 2) build corner bboxes from corner quads
    annotations = []

    for p in placements:
        cls = p["cls"]
        tl_quad = p.get("tl_quad")
        br_quad = p.get("br_quad")

        class_id = LABEL_ORDER.index(cls)  # LABEL_ORDER must match YAML 'names' order

        if tl_quad is not None:
            bbox = quad_to_bbox_clamped(tl_quad, W, H)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                annotations.append((class_id, x1, y1, x2, y2))

        if br_quad is not None:
            bbox = quad_to_bbox_clamped(br_quad, W, H)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                annotations.append((class_id, x1, y1, x2, y2))

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

    num_samples = 20  # scale up for real training
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
            if random.random() < 0.7:
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