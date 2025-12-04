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
    IMAGES_DIR,
    LABELS_DIR,
    CARD_CLASSES,
)

from src.dataset_gen.yolo_label_utils import bbox_to_yolo

SEG_BASE_DIR = Path("data/yolo_cards_seg")
SEG_IMAGES_DIR = SEG_BASE_DIR / "images"
SEG_LABELS_DIR = SEG_BASE_DIR / "labels"
SEG_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
SEG_LABELS_DIR.mkdir(parents=True, exist_ok=True)


def compute_rotated_quad(new_w, new_h, angle_deg):
    """
    Compute the 4 corner coordinates of a rectangle of size (new_w, new_h)
    after rotation by angle_deg (in degrees) around its center, assuming
    PIL's rotate(expand=True) semantics: the rotated rectangle is placed
    in a new canvas whose top-left is at (0, 0).

    Returns a (4, 2) array of (x, y) in the rotated-image coordinate system.
    Order: [top-left, top-right, bottom-right, bottom-left] (approximately).
    """
    corners = np.array([
        [0,         0        ],  # TL
        [new_w,     0        ],  # TR
        [new_w,     new_h    ],  # BR
        [0,         new_h    ],  # BL
    ], dtype=np.float32)

    cx = new_w / 2.0
    cy = new_h / 2.0

    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)

    shifted = corners - np.array([[cx, cy]], dtype=np.float32)
    rotated = shifted @ R.T + np.array([[cx, cy]], dtype=np.float32)

    # simulate expand=True canvas: shift so that min coords are at (0,0)
    min_xy = rotated.min(axis=0, keepdims=True)
    rotated -= min_xy

    # reorder approximately TL, TR, BR, BL by y then x
    ys = rotated[:, 1]
    xs = rotated[:, 0]
    idx_top = np.argsort(ys)[:2]
    idx_bottom = np.argsort(ys)[2:]

    top = rotated[idx_top]
    bottom = rotated[idx_bottom]

    top = top[np.argsort(top[:, 0])]       # left, right
    bottom = bottom[np.argsort(bottom[:, 0])]

    ordered = np.vstack([top, bottom[::-1]])  # TL, TR, BR, BL
    return ordered

def shift_placements_inside(placements, W, H, margin=2):
    """
    Shift a list of placements so that all bboxes lie fully inside [0, W] x [0, H].
    Each placement has:
        "bbox": (xmin, ymin, xmax, ymax)
        "poly": [(x, y), ...]  in global coordinates (if present).
    """
    if not placements:
        return placements

    min_x = min(p["bbox"][0] for p in placements)
    min_y = min(p["bbox"][1] for p in placements)
    max_x = max(p["bbox"][2] for p in placements)
    max_y = max(p["bbox"][3] for p in placements)

    # If the fan is literally larger than the image, we just skip shifting.
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
        if "poly" in p and p["poly"] is not None:
            p["poly"] = [(px + dx, py + dy) for (px, py) in p["poly"]]

    return placements

# -----------------------
# Helpers to load assets
# -----------------------

def load_card_variants():
    """
    Build a dict of the form:

        {
          "AS": {
             "normal":   [PIL.Image, ...],
             "inverted": [PIL.Image, ...],
             "real":     [PIL.Image, ...],
          },
          "2D": {...},
          ...
        }

    using STYLE_GROUP_DIRS from config.py
    """
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

    # Warn if any card is completely missing
    for cls, style_dict in variants.items():
        if all(len(imgs) == 0 for imgs in style_dict.values()):
            print(f"[WARN] No variants found for {cls}")

    return variants

def choose_card_image(card_variants_for_cls):
    """
    card_variants_for_cls is a dict like:
        {"normal": [imgs...], "inverted": [...], "real": [...]}

    We sample a style according to STYLE_WEIGHTS (normalized over
    available styles that actually have images), then pick a random
    image from that style.
    """
    # Keep only styles that actually have at least 1 image
    available_styles = [
        s for s, imgs in card_variants_for_cls.items()
        if len(imgs) > 0
    ]
    if not available_styles:
        return None  # caller should handle this

    # Build weights for the available styles
    weights = [STYLE_WEIGHTS.get(s, 0.0) for s in available_styles]
    if sum(weights) <= 0:
        # fallback: uniform if all weights are zero
        weights = [1.0] * len(available_styles)

    # Sample style according to the desired probability
    style_choice = random.choices(available_styles, weights=weights, k=1)[0]

    # Random image within that style
    img = random.choice(card_variants_for_cls[style_choice]).copy()
    return img

def load_backgrounds():
    """
    Load possible table / desk / felt / etc. backgrounds from data/backgrounds/.
    """
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
    """
    Draw random rectangles/ellipses that vaguely resemble hands/phones etc.
    This teaches YOLO 'not-card' patterns.
    """
    bg = bg_img.copy()
    W, H = bg.size
    img = np.array(bg)

    for _ in range(num_shapes):
        # random position and size
        w = random.randint(int(0.05 * W), int(0.25 * W))
        h = random.randint(int(0.05 * H), int(0.25 * H))
        x1 = random.randint(0, max(0, W - w))
        y1 = random.randint(0, max(0, H - h))
        x2 = x1 + w
        y2 = y1 + h

        # choose a 'phone-like' or 'hand-like' color
        if random.random() < 0.5:
            # dark gray / black-ish (phone, wallet, etc.)
            color = (
                random.randint(10, 40),
                random.randint(10, 40),
                random.randint(10, 40),
            )
        else:
            # skin-ish tone
            color = (
                random.randint(150, 220),
                random.randint(130, 200),
                random.randint(100, 180),
            )

        # rectangle or ellipse
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
    """
    Build a 'no cards' image that still looks card-ish:
    - background + fake occluder shapes
    - returns img_bgr, empty_annotations
    """
    bg_img = random.choice(backgrounds)
    img_with_shapes = draw_fake_occluders(bg_img, num_shapes=random.randint(2, 6))
    img_bgr = np.array(img_with_shapes)[:, :, ::-1]  # RGB->BGR
    annotations = []  # NO labels => hard negative
    return img_bgr, annotations

def hard_fan_instances(card_variants, W, H):
    """
    A harder version of fan_cluster_instances:
    - tighter angles (cards almost parallel)
    - more cards
    - fans near edges / partially off-frame
    """
    placements = []

    # only keep card classes that actually have at least one image
    available_classes = [
        cls for cls, style_dict in card_variants.items()
        if any(len(imgs) > 0 for imgs in style_dict.values())
    ]
    if not available_classes:
        return placements

    # 3–6 cards in a tight fan
    k = random.randint(3, 6)
    chosen_classes = random.sample(available_classes, k=min(k, len(available_classes)))

    # pick one image for each class using style weights
    base_images = []
    for c in chosen_classes:
        img = choose_card_image(card_variants[c])
        base_images.append(img)

    # if any class had no available image, bail out
    if any(img is None for img in base_images):
        return placements

    # base scale
    base_card_width = base_images[0].size[0]
    scale_base = (W / base_card_width) * 0.22  # slightly larger than normal fan
    scale_factor = random.uniform(1.0, 1.4)
    scale = scale_base * scale_factor

    start_angle = random.uniform(-10, 10)   # cards almost vertical
    step_deg    = random.uniform(4, 8)      # very tight spread

    # anchor somewhere where part of the fan may fall outside frame
    anchor_x = random.randint(int(0.4 * W), int(0.95 * W))
    anchor_y = random.randint(int(0.4 * H), int(0.95 * H))

    for i, (cls, img_raw) in enumerate(zip(chosen_classes, base_images)):
        orig_w, orig_h = img_raw.size
        if orig_w == 0 or orig_h == 0:
            continue

        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        card_img = img_raw.resize((new_w, new_h), Image.BICUBIC)

        angle = start_angle - i * step_deg

        # 1) get rotated quad in the SAME coords as the rotated image (expand=True)
        quad_local = compute_rotated_quad(new_w, new_h, angle)

        # 2) rotate the image with expand=True (same transform as in quad_local)
        rotated = card_img.rotate(angle, expand=True)
        rot_w, rot_h = rotated.size

        # 3) choose the "grip" point as the bottom-right vertex of the quad
        #    (compute_rotated_quad returns approx [TL, TR, BR, BL])
        grip_rot_x, grip_rot_y = quad_local[2]

        # 4) overlap offsets so earlier cards are pushed left/up
        dist_from_front = (k - 1) - i
        fan_offset_x = -dist_from_front * (0.10 * new_w)
        fan_offset_y = -dist_from_front * (0.04 * new_h)

        xmin = int(anchor_x + fan_offset_x - grip_rot_x)
        ymin = int(anchor_y + fan_offset_y - grip_rot_y)

        xmax = xmin + rot_w
        ymax = ymin + rot_h

        poly_global = [(float(px + xmin), float(py + ymin)) for (px, py) in quad_local]

        placements.append({
            "cls": cls,
            "img": rotated,
            "bbox": (xmin, ymin, xmax, ymax),
            "poly": poly_global,
        })

    # keep this fan only if all its cards are fully inside the background
    for p in placements:
        x1, y1, x2, y2 = p["bbox"]
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            # discard this entire fan – we'll just not use it
            return []

    # shift whole fan so all cards are fully inside the image
    placements = shift_placements_inside(placements, W, H)

    return placements

# -----------------------
# Card placement routines
# -----------------------

def random_single_card_instances(card_variants, W, H, num_cards):
    """
    Return a list of card placements, where each placement is:
    {
       "cls": "AS",
       "img": PIL.Image (already rotated+scaled),
       "bbox": (xmin, ymin, xmax, ymax) BEFORE clipping,
    }
    For normal table scattering mode.
    """
    placements = []

    available_classes = [
    cls for cls, style_dict in card_variants.items()
    if any(len(imgs) > 0 for imgs in style_dict.values())
    ]

    if len(available_classes) == 0:
        return placements

    chosen_classes = random.sample(
        available_classes,
        k=min(num_cards, len(available_classes))
    )

    for cls in chosen_classes:
        # pick style with desired probabilities
        base_img = choose_card_image(card_variants[cls])
        if base_img is None:
            continue

        orig_w, orig_h = base_img.size
        if orig_w == 0 or orig_h == 0:
            continue

        # Decide how big this card should be RELATIVE to the background.
        # Keep two modes:
        #   - normal cards: smaller
        #   - zoomed-in cards: a bit larger, but never crazy big
        r = random.random()
        if r < 0.35:
            # "zoomed-in" single card
            frac_min, frac_max = 0.20, 0.32   # % of background width
        else:
            # normal single card
            frac_min, frac_max = 0.12, 0.18   # % of background width

        target_frac = random.uniform(frac_min, frac_max)
        target_w = int(W * target_frac)

        # Resize to this target width, preserving aspect ratio
        scale = target_w / float(orig_w)
        new_w = target_w
        new_h = int(orig_h * scale)

        card_img = base_img.resize((new_w, new_h), Image.BICUBIC)

        # rotation
        angle = random.uniform(-90, 90)

        # compute rotated quad in local (rotated image) coords
        quad_local = compute_rotated_quad(new_w, new_h, angle)

        # actual image rotation
        card_img = card_img.rotate(angle, expand=True)
        rot_w, rot_h = card_img.size

        # 🔒 Require this card to fit completely inside the background
        if rot_w >= W or rot_h >= H:
            # card would be larger than the whole image – skip this one
            continue

        # sample a position such that the rotated card is fully inside
        xmin = random.randint(0, W - rot_w)
        ymin = random.randint(0, H - rot_h)
        xmax = xmin + rot_w
        ymax = ymin + rot_h

        # convert local quad to global image coords
        poly_global = [(float(px + xmin), float(py + ymin)) for (px, py) in quad_local]

        placements.append({
            "cls": cls,
            "img": card_img,
            "bbox": (xmin, ymin, xmax, ymax),
            "poly": poly_global,
        })


    return placements

def fan_cluster_instances(card_variants, W, H):
    """
    Simulate a right-hand poker fan:
    - LAST card is most clockwise / in front.
    - Earlier cards are tucked behind to the left.
    """

    placements = []

    # only keep card classes that actually have at least one image
    available_classes = [
        cls for cls, style_dict in card_variants.items()
        if any(len(imgs) > 0 for imgs in style_dict.values())
    ]
    if len(available_classes) == 0:
        return placements

    # 2–5 cards in the fan
    k = random.randint(2, 5)
    chosen_classes = random.sample(available_classes, k=min(k, len(available_classes)))

    base_images = []
    for c in chosen_classes:
        img = choose_card_image(card_variants[c])
        base_images.append(img)

    if any(img is None for img in base_images):
        return placements

    # "in hand" card size
    fan_frac_min = 0.18
    fan_frac_max = 0.22
    fan_width_fraction = random.uniform(fan_frac_min, fan_frac_max)
    target_card_w = int(W * fan_width_fraction)

    start_angle = random.uniform(-20, 30)
    step_deg    = random.uniform(9, 16)

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

        # 1) quad in SAME coord frame as rotated(expand=True)
        quad_local = compute_rotated_quad(new_w, new_h, angle)

        # 2) rotate image
        rotated = card_img.rotate(angle, expand=True)
        rot_w, rot_h = rotated.size

        # 3) “grip point” = bottom-right vertex in that same frame
        grip_rot_x, grip_rot_y = quad_local[2]

        # 4) offsets so earlier cards are pushed left/up
        dist_from_front = (k - 1) - i
        fan_offset_x = -dist_from_front * (0.08 * new_w)
        fan_offset_y = -dist_from_front * (0.03 * new_h)

        xmin = int(anchor_x + fan_offset_x - grip_rot_x)
        ymin = int(anchor_y + fan_offset_y - grip_rot_y)
        xmax = xmin + rot_w
        ymax = ymin + rot_h

        poly_global = [(float(px + xmin), float(py + ymin)) for (px, py) in quad_local]

        placements.append({
            "cls": cls,
            "img": rotated,
            "bbox": (xmin, ymin, xmax, ymax),
            "poly": poly_global,
        })

        # keep this fan only if all its cards are fully inside the background
    for p in placements:
        x1, y1, x2, y2 = p["bbox"]
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            # discard this entire fan – we'll just not use it
            return []

    # shift whole fan so all cards are fully inside the image
    placements = shift_placements_inside(placements, W, H)

    return placements

def compose_on_background(bg_img, placements):
    """
    Paste all given card placements into bg_img and return:
        composite    : final BGR image (H x W x 3, uint8)
        annotations  : [(class_id, xmin, ymin, xmax, ymax), ...]
        seg_polys    : [(class_id, [(x1,y1), ...]), ...]

    IMPORTANT:
    - We first render the cards in their natural order (back -> front)
      so the image looks correct.
    - Then we compute VISIBLE masks in the opposite order (front -> back),
      using the alpha channel and an occupancy map.
    - We drop instances that are too heavily occluded so they don't
      produce crazy skinny triangles/strips.
    """
    bg = bg_img.copy()
    W, H = bg.size

    # --- 1) Draw all cards once (back -> front) ---
    # placements are already ordered from back to front by our generators
    for p in placements:
        card_img = p["img"]           # PIL RGBA
        xmin, ymin, xmax, ymax = p["bbox"]

        # clip bbox to image bounds
        vis_xmin = max(int(xmin), 0)
        vis_ymin = max(int(ymin), 0)
        vis_xmax = min(int(xmax), W)
        vis_ymax = min(int(ymax), H)

        if vis_xmin >= vis_xmax or vis_ymin >= vis_ymax:
            continue

        # crop visible portion of the card
        crop_left   = vis_xmin - xmin
        crop_top    = vis_ymin - ymin
        crop_right  = crop_left + (vis_xmax - vis_xmin)
        crop_bottom = crop_top  + (vis_ymax - vis_ymin)

        card_visible = card_img.crop((crop_left, crop_top, crop_right, crop_bottom))

        # paste with alpha
        bg.paste(card_visible, (vis_xmin, vis_ymin), card_visible)

    # --- 2) Build segmentation labels (front -> back) ---
    occupancy = np.zeros((H, W), dtype=bool)   # where foreground is already taken by FRONT cards
    annotations = []
    seg_polys = []

    # thresholds to avoid crazy tiny/skinny masks
    MIN_VIS_PIXELS   = 50        # absolute minimum visible pixels
    MIN_VIS_FRACTION = 0.10       # require at least 25% of the card to be visible

    # process from front card to back card
    for p in reversed(placements):
        cls = p["cls"]
        card_img = p["img"]
        xmin, ymin, xmax, ymax = p["bbox"]

        # clip bbox again
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
            # fallback: label as plain bbox if alpha missing
            class_id = CARD_CLASSES.index(cls)
            annotations.append((class_id, vis_xmin, vis_ymin, vis_xmax, vis_ymax))
            continue

        # alpha mask in local coords
        alpha = np.array(card_visible.split()[-1])   # H_vis x W_vis
        mask_local = alpha > 10
        if not mask_local.any():
            continue

        h_vis, w_vis = mask_local.shape

        # full-frame mask of this card region
        card_mask_full = np.zeros((H, W), dtype=bool)
        card_mask_full[vis_ymin:vis_ymin + h_vis,
                       vis_xmin:vis_xmin + w_vis] = mask_local

        full_area = int(card_mask_full.sum())
        if full_area == 0:
            continue

        # visible area = card - regions already occupied by closer cards
        visible_mask = card_mask_full & (~occupancy)
        vis_area = int(visible_mask.sum())

        # --- filter out heavily occluded cards ---
        if vis_area < MIN_VIS_PIXELS or (vis_area / full_area) < MIN_VIS_FRACTION:
            # still update occupancy so deeper cards don't shine through
            occupancy |= card_mask_full
            continue

        ys, xs = np.where(visible_mask)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)  # (N,2) [x,y]

        # oriented minimal rectangle over visible pixels
        rect = cv2.minAreaRect(pts)      # ((cx,cy),(w,h),angle)
        box = cv2.boxPoints(rect)        # 4x2 float
        poly = [(float(x), float(y)) for (x, y) in box]

        # clamp polygon to image bounds
        clamped = []
        for px, py in poly:
            px = min(max(px, 0.0), float(W - 1))
            py = min(max(py, 0.0), float(H - 1))
            clamped.append((px, py))

        xs_poly = [pt[0] for pt in clamped]
        ys_poly = [pt[1] for pt in clamped]
        bb_xmin, bb_xmax = min(xs_poly), max(xs_poly)
        bb_ymin, bb_ymax = min(ys_poly), max(ys_poly)

        class_id = CARD_CLASSES.index(cls)
        annotations.append(
            (class_id, int(bb_xmin), int(bb_ymin), int(bb_xmax), int(bb_ymax))
        )
        seg_polys.append((class_id, clamped))

        # mark this card as occupying its whole area for deeper cards
        occupancy |= card_mask_full

    # we iterated placements in reversed order; if you care about order
    # (not necessary for YOLO) you can reverse lists, but it's not required.
    composite = np.array(bg)[:, :, ::-1]  # RGB -> BGR
    return composite, annotations, seg_polys

# -----------------------
# realism augs
# -----------------------

def apply_post_augs(img_bgr):
    """
    Heavier 'camera / phone' style augs:
    - stronger brightness/contrast jitter
    - random color cast
    - Gaussian / motion blur
    - Gaussian noise
    - JPEG compression artifacts
    """
    out = img_bgr.astype(np.float32)

    # --- brightness & contrast ---
    if random.random() < 0.9:
        alpha = random.uniform(0.6, 1.3)   # contrast
        beta  = random.uniform(-45, 35)    # brightness
        out = out * alpha + beta

    # --- subtle color cast / white balance shift ---
    if random.random() < 0.7:
        # RGB scaling
        rb = random.uniform(0.9, 1.1)
        gb = random.uniform(0.9, 1.1)
        bb = random.uniform(0.9, 1.1)
        color_gain = np.array([bb, gb, rb], dtype=np.float32).reshape(1, 1, 3)
        out = out * color_gain

    # --- blur (Gaussian or cheap 'motion' blur) ---
    if random.random() < 0.6:
        if random.random() < 0.5:
            k = random.choice([3, 5, 7])
            out = cv2.GaussianBlur(out, (k, k), sigmaX=random.uniform(0.8, 2.0))
        else:
            # simple motion blur: average along horizontal or vertical
            k = random.choice([3, 5, 7])
            kernel = np.zeros((k, k), np.float32)
            if random.random() < 0.5:
                kernel[int((k - 1) / 2), :] = 1.0 / k
            else:
                kernel[:, int((k - 1) / 2)] = 1.0 / k
            out = cv2.filter2D(out, -1, kernel)

    # --- Gaussian noise ---
    if random.random() < 0.7:
        sigma = random.uniform(5, 18)
        noise = np.random.normal(0, sigma, out.shape).astype(np.float32)
        out = out + noise

    # --- JPEG compression artifacts ---
    if random.random() < 0.6:
        quality = random.randint(25, 70)  # lower = more artifacts
        enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ok, enc = cv2.imencode(".jpg", np.clip(out, 0, 255).astype(np.uint8), enc_param)
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32)

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def save_example(idx, img_bgr, annotations):
    """
    Write final .jpg and .txt
    """
    img_h, img_w = img_bgr.shape[:2]

    img_path = IMAGES_DIR / f"img_{idx:06d}.jpg"
    lbl_path = LABELS_DIR / f"img_{idx:06d}.txt"

    cv2.imwrite(str(img_path), img_bgr)

    lines = []
    for class_id, xmin, ymin, xmax, ymax in annotations:
        cx, cy, w, h = bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(lbl_path, "w") as f:
        f.write("\n".join(lines))

def save_example_seg(idx, img_bgr, seg_polys):
    """
    Save image and YOLO-segmentation labels in *polygon-only* format:

        class x1 y1 x2 y2 ... xN yN

    All coords normalized to [0,1].
    We compress all cards into a single class: 0 = 'card'.
    """
    img_h, img_w = img_bgr.shape[:2]

    img_path = SEG_IMAGES_DIR / f"img_{idx:06d}.jpg"
    lbl_path = SEG_LABELS_DIR / f"img_{idx:06d}.txt"

    cv2.imwrite(str(img_path), img_bgr)

    lines = []
    for _, poly in seg_polys:  # ignore original class_id, treat everything as 'card'
        if len(poly) < 3:
            continue

        # normalize polygon coords
        norm_coords = []
        for px, py in poly:
            nx = px / float(img_w)
            ny = py / float(img_h)
            norm_coords.append(nx)
            norm_coords.append(ny)

        seg_str = " ".join(f"{v:.6f}" for v in norm_coords)
        # single-class: 0 = 'card'
        line = f"0 {seg_str}"
        lines.append(line)

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

    if all(len(v) == 0 for v in card_variants.values()):
        print("[FATAL] You have no cards in data/raw_cards/(normal|inverted|real).")
        return

    num_samples = 2500  # scale up later
    HARD_NEG_PROB = 0.08   # % images with no cards but card-like clutter
    HARD_POS_PROB = 0.22   # % images as 'hard' fans


    for i in tqdm(range(num_samples), desc="Generating synthetic dataset"):
        bg_img = random.choice(backgrounds)
        W, H = bg_img.size

        mode_r = random.random()

        # --- 1) HARD NEGATIVE: no cards, just background + fake occluders ---
        if mode_r < HARD_NEG_PROB:
            img_bgr, ann = make_hard_negative_example(backgrounds)
            img_bgr = apply_post_augs(img_bgr)
            # no cards => no polygons
            save_example_seg(i, img_bgr, [])
            continue

        # --- 2) HARD POSITIVE: nasty overlapping fan near edges ---
        elif mode_r < HARD_NEG_PROB + HARD_POS_PROB:
            placements = hard_fan_instances(card_variants, W, H)
            # optionally add 0–1 extra stray card somewhere else
            extra = random_single_card_instances(card_variants, W, H, num_cards=random.randint(0, 1))
            placements.extend(extra)

        # --- 3) NORMAL MIXTURE (your existing behaviour) ---
        else:
            if random.random() < 0.7:
                placements = fan_cluster_instances(card_variants, W, H)
                extra = random_single_card_instances(card_variants, W, H, num_cards=random.randint(0, 2))
                placements.extend(extra)
            else:
                placements = random_single_card_instances(card_variants, W, H, num_cards=random.randint(2, 7))

        # NEW: if no cards placed at all (unintentional empty), regenerate this index
        if not placements:
            continue

        img_bgr, ann, seg_polys = compose_on_background(bg_img, placements)
        img_bgr = apply_post_augs(img_bgr)

        # if compose produced no polygons, treat as intentional negative only
        if not seg_polys:
            # if you actually *want* some real hard negatives, you can
            # either call save_example_seg(i, img_bgr, []) here
            # or just skip this sample:
            #   i -= 1; continue
            save_example_seg(i, img_bgr, [])
        else:
            save_example_seg(i, img_bgr, seg_polys)

    print("Done generating synthetic dataset.")

if __name__ == "__main__":
    main()
