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
        # pick style with desired probabilities
        base_img = choose_card_image(card_variants[cls])
        if base_img is None:
            continue

        # scale:
        # 65% normal, 35% zoomed-in large
        # make scale adaptive to background width
        base_card_width = base_img.size[0]
        scale_base = (W / base_card_width) * 0.15  # 15% of bg width looks like a realistic card

        if random.random() < 0.35:
            scale_factor = random.uniform(1.1, 2.1)
        else:
            scale_factor = random.uniform(0.5, 1.1)

        scale = scale_base * scale_factor


        new_w = int(base_img.size[0] * scale)
        new_h = int(base_img.size[1] * scale)
        card_img = base_img.resize((new_w, new_h), Image.BICUBIC)

        # rotation
        angle = random.uniform(-30, 30)
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
    base_card_width = base_images[0].size[0]
    scale_base = (W / base_card_width) * 0.17  # ~17% of frame width baseline
    scale_factor = random.uniform(1.0, 1.3)
    scale = scale_base * scale_factor

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
        # Resize this card
        new_w = int(img_raw.size[0] * scale)
        new_h = int(img_raw.size[1] * scale)
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
    Simulate webcam: brightness, blur, noise.
    """
    out = img_bgr.astype(np.float32)

    # brightness / contrast
    if random.random() < 0.7:
        alpha = random.uniform(0.8, 1.2)  # contrast
        beta = random.uniform(-20, 20)    # brightness
        out = out * alpha + beta

    # blur
    if random.random() < 0.3:
        k = random.choice([3,5])
        out = cv2.GaussianBlur(out, (k, k), sigmaX=1.0)

    # noise
    if random.random() < 0.4:
        noise = np.random.normal(0, 8, out.shape).astype(np.float32)
        out = out + noise

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

    num_samples = 10  # scale up later

    for i in tqdm(range(num_samples), desc="Generating synthetic dataset"):
        bg_img = random.choice(backgrounds)
        W, H = bg_img.size

        # With 40% probability, generate a fanned "hand"
        if random.random() < 0.4:
            placements = fan_cluster_instances(card_variants, W, H)
            # plus maybe add 0-2 scattered single cards on table for realism
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
