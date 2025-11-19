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

        grip_x_local = new_w * 0.9
        grip_y_local = new_h * 0.96

        cx = new_w / 2.0
        cy = new_h / 2.0
        theta = math.radians(angle)

        dx = grip_x_local - cx
        dy = grip_y_local - cy

        grip_rot_x = cx + (dx * math.cos(theta) - dy * math.sin(theta))
        grip_rot_y = cy + (dx * math.sin(theta) + dy * math.cos(theta))

        rotated = card_img.rotate(angle, expand=True)
        rot_w, rot_h = rotated.size

        dist_from_front = (k - 1) - i
        fan_offset_x = -dist_from_front * (0.10 * new_w)
        fan_offset_y = -dist_from_front * (0.04 * new_h)

        xmin = int(anchor_x + fan_offset_x - grip_rot_x)
        ymin = int(anchor_y + fan_offset_y - grip_rot_y)
        xmax = xmin + rot_w
        ymax = ymin + rot_h

        placements.append({
            "cls": cls,
            "img": rotated,
            "bbox": (xmin, ymin, xmax, ymax),
        })

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

    available_classes = [cls for cls, imgs in card_variants.items() if len(imgs) > 0]
    if len(available_classes) == 0:
        return placements

    chosen_classes = random.sample(
        available_classes,
        k=min(num_cards, len(available_classes))
    )

    for cls in chosen_classes:
        # pick style with desired probabilities (70% real, 20% normal, 10% inverted)
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
            frac_min, frac_max = 0.20, 0.30   # % of background width
        else:
            # normal single card
            frac_min, frac_max = 0.10, 0.18   # % of background width

        target_frac = random.uniform(frac_min, frac_max)
        target_w = int(W * target_frac)

        # Resize to this target width, preserving aspect ratio
        scale = target_w / float(orig_w)
        new_w = target_w
        new_h = int(orig_h * scale)

        card_img = base_img.resize((new_w, new_h), Image.BICUBIC)


        # rotation
        angle = random.uniform(-90, 90)
        card_img = card_img.rotate(angle, expand=True)

        rot_w, rot_h = card_img.size

        # We allow off-screen placement by ~15% of bg size.
        # But we must clamp so low <= high even if card is huge.
        raw_xmin_low  = int(-0.15 * W)
        raw_xmin_high = int(W - rot_w + 0.15 * W)
        raw_ymin_low  = int(-0.15 * H)
        raw_ymin_high = int(H - rot_h + 0.15 * H)

        # Fix swapped ranges if card is bigger than background
        xmin_low  = min(raw_xmin_low,  raw_xmin_high)
        xmin_high = max(raw_xmin_low,  raw_xmin_high)
        ymin_low  = min(raw_ymin_low,  raw_ymin_high)
        ymin_high = max(raw_ymin_low,  raw_ymin_high)

        xmin = random.randint(xmin_low, xmin_high)
        ymin = random.randint(ymin_low, ymin_high)
        xmax = xmin + rot_w
        ymax = ymin + rot_h

        placements.append({
            "cls": cls,
            "img": card_img,
            "bbox": (xmin, ymin, xmax, ymax),
        })

    return placements



def fan_cluster_instances(card_variants, W, H):
    """
    Simulate a right-hand poker fan:
    - The LAST card in the list is the most clockwise and closest to the thumb,
      like the card you're really looking at.
    - Earlier cards are tucked slightly behind and to the left.
    """

    placements = []

    available_classes = [cls for cls, imgs in card_variants.items() if len(imgs) > 0]
    if len(available_classes) == 0:
        return placements

    # number of cards in the fan
    k = random.randint(2, 5)

    # pick k unique logical cards
    chosen_classes = random.sample(
        available_classes,
        k=min(k, len(available_classes))
    )
    # we'll treat index 0 = far left/back, index k-1 = far right/front
    # so we keep the order we sampled, that's fine

    base_images = []
    for c in chosen_classes:
        img = choose_card_image(card_variants[c])
        if img is not None:
            base_images.append(img)
        else:
            base_images.append(None)

    # If any card has no available image, just skip fan for safety
    if any(img is None for img in base_images):
        return []


    # scale: "in hand" cards should be decently large
    fan_frac_min = 0.16  # 16% of bg width
    fan_frac_max = 0.22  # 22% of bg width
    fan_width_fraction = random.uniform(fan_frac_min, fan_frac_max)
    target_card_w = int(W * fan_width_fraction)

    # rotation plan:
    # - card 0 (back/left)  : least clockwise
    # - card k-1 (front/right): most clockwise
    #
    # start_angle can be fairly vertical (-5° to +10° ish)
    # step_deg is clockwise "fan amount"
    start_angle = random.uniform(-20, 30)
    step_deg    = random.uniform(9, 16)

    # thumb grip location in the frame:
    # near lower-right-ish, like your reference photo (hand at lower right)
    anchor_x = random.randint(int(0.45 * W), int(0.85 * W))
    anchor_y = random.randint(int(0.55 * H), int(0.95 * H))

    for i, (cls, img_raw) in enumerate(zip(chosen_classes, base_images)):
        # Resize this card so its width is EXACTLY target_card_w
        orig_w, orig_h = img_raw.size
        if orig_w == 0 or orig_h == 0:
            continue

        scale = target_card_w / float(orig_w)
        new_w = target_card_w
        new_h = int(orig_h * scale)
        card_img = img_raw.resize((new_w, new_h), Image.BICUBIC)


        # Angle grows with i so that last card is the most clockwise.
        angle = start_angle - i * step_deg

        # "grip point" (thumb hold) is slightly inside the bottom-right corner.
        # Move it a touch more inward than before so it really looks like thumb pressure.
        grip_x_local = new_w * 0.92
        grip_y_local = new_h * 0.92

        # Rotate around center to figure out where that grip point lands.
        cx = new_w / 2.0
        cy = new_h / 2.0
        theta = math.radians(angle)

        dx = grip_x_local - cx
        dy = grip_y_local - cy

        grip_rot_x = cx + (dx * math.cos(theta) - dy * math.sin(theta))
        grip_rot_y = cy + (dx * math.sin(theta) + dy * math.cos(theta))

        # rotate actual image with expand=True
        rotated = card_img.rotate(angle, expand=True)
        rot_w, rot_h = rotated.size

        # Now we offset cards so that:
        #  - earlier cards (i=0) are pushed LEFT and UP a little
        #  - later cards (i close to k-1) sit closer to the "true" thumb position
        # This matches the real fan: you see edges of earlier cards to the left,
        # and the "main" card is most right/forward.
        #
        # We'll define offset relative to distance from the FRONT card (k-1).
        dist_from_front = (k - 1) - i  # so last card => 0, first card => biggest dist
        fan_offset_x = -dist_from_front * (0.08 * new_w)  # push back cards left
        fan_offset_y = -dist_from_front * (0.03 * new_h)  # push back cards slightly up

        xmin = int(anchor_x + fan_offset_x - grip_rot_x)
        ymin = int(anchor_y + fan_offset_y - grip_rot_y)
        xmax = xmin + rot_w
        ymax = ymin + rot_h

        placements.append({
            "cls": cls,
            "img": rotated,
            "bbox": (xmin, ymin, xmax, ymax),
        })

    return placements




def compose_on_background(bg_img, placements):
    """
    Paste all given card placements into bg_img with proper clipping,
    and return:
        final_bgr (np.uint8 HxWx3)
        annotations [(class_id, xmin, ymin, xmax, ymax), ...] in VISIBLE coords
    """
    bg = bg_img.copy()
    W, H = bg.size
    annotations = []

    for p in placements:
        cls = p["cls"]
        card_img = p["img"]
        xmin, ymin, xmax, ymax = p["bbox"]

        # clip bbox to visible region
        vis_xmin = max(xmin, 0)
        vis_ymin = max(ymin, 0)
        vis_xmax = min(xmax, W)
        vis_ymax = min(ymax, H)

        if vis_xmin >= vis_xmax or vis_ymin >= vis_ymax:
            continue

        # crop visible region of card
        crop_left = vis_xmin - xmin
        crop_top = vis_ymin - ymin
        crop_right = crop_left + (vis_xmax - vis_xmin)
        crop_bottom = crop_top + (vis_ymax - vis_ymin)

        card_visible = card_img.crop((crop_left, crop_top, crop_right, crop_bottom))

        # alpha composite
        bg.paste(card_visible, (vis_xmin, vis_ymin), card_visible)

        # record bbox (only visible part)
        class_id = CARD_CLASSES.index(cls)
        annotations.append((class_id, vis_xmin, vis_ymin, vis_xmax, vis_ymax))

    composite = np.array(bg)[:, :, ::-1]  # RGB -> BGR
    return composite, annotations


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
        alpha = random.uniform(0.7, 1.4)   # contrast
        beta  = random.uniform(-40, 40)    # brightness
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

    num_samples = 1200  # scale up later
    HARD_NEG_PROB = 0.10   # % images with no cards but card-like clutter
    HARD_POS_PROB = 0.15   # % images as 'hard' fans


    for i in tqdm(range(num_samples), desc="Generating synthetic dataset"):
        bg_img = random.choice(backgrounds)
        W, H = bg_img.size

        mode_r = random.random()

        # --- 1) HARD NEGATIVE: no cards, just background + fake occluders ---
        if mode_r < HARD_NEG_PROB:
            img_bgr, ann = make_hard_negative_example(backgrounds)
            # still apply camera-style augs
            img_bgr = apply_post_augs(img_bgr)
            # write an empty label file so YOLO sees this as a negative
            save_example(i, img_bgr, [])
            continue

        # --- 2) HARD POSITIVE: nasty overlapping fan near edges ---
        elif mode_r < HARD_NEG_PROB + HARD_POS_PROB:
            placements = hard_fan_instances(card_variants, W, H)
            # optionally add 0–1 extra stray card somewhere else
            extra = random_single_card_instances(card_variants, W, H, num_cards=random.randint(0, 1))
            placements.extend(extra)

        # --- 3) NORMAL MIXTURE (your existing behaviour) ---
        else:
            if random.random() < 0.4:
                placements = fan_cluster_instances(card_variants, W, H)
                extra = random_single_card_instances(card_variants, W, H, num_cards=random.randint(0, 2))
                placements.extend(extra)
            else:
                placements = random_single_card_instances(card_variants, W, H, num_cards=random.randint(2, 7))

        img_bgr, ann = compose_on_background(bg_img, placements)
        img_bgr = apply_post_augs(img_bgr)
        save_example(i, img_bgr, ann)

    print("Done generating synthetic dataset.")


if __name__ == "__main__":
    main()
