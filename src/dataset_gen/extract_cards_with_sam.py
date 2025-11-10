from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

# --- Make sure we can import src.* ---
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.segmentation.sam_utils import load_sam_automatic, generate_masks


# ---- Config ----
RAW_DIR = ROOT / "data" / "raw_cards" / "realjpg"   # input photos
OUT_DIR = ROOT / "data" / "raw_cards" / "real"      # output PNG cards

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Your real card ratio ~ 1.53
# We keep any mask whose bounding box has:
#   1.43 < (height / width) < 1.63
RATIO_MIN = 1.23
RATIO_MAX = 1.83

# Target normalized card size (width x height).
# Use ~1.53 aspect: H = 1.53 * W
TARGET_W = 700
TARGET_H = int(TARGET_W * 1.53)


def is_card_mask(mask_dict: dict, img_area: int) -> bool:
    """
    Decide if a SAM mask looks like a card, based on:
      - area fraction
      - bounding box aspect ratio
    """
    area = mask_dict.get("area", 0)
    if area < 0.03 * img_area:
        return False

    x, y, w, h = mask_dict.get("bbox", [0, 0, 0, 0])
    if w <= 0 or h <= 0:
        return False

    ratio = h / float(w)
    if ratio < RATIO_MIN or ratio > RATIO_MAX:
        return False

    return True


def order_points_quad(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as TL, TR, BR, BL.
    """
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_card_from_mask(bgr_img: np.ndarray, mask_bool: np.ndarray) -> np.ndarray | None:
    """
    Given a BGR image and a boolean mask for one card,
    find a quadrilateral around it and warp to TARGET_W x TARGET_H.
    Returns an RGBA (H x W x 4) array or None if it fails.
    """
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 50:
        return None

    # Min-area rectangle around the card
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)  # 4x2 float32
    box = order_points_quad(box)  # TL, TR, BR, BL

    # Target quad (portrait-ish: width < height)
    dst = np.array([
        [0,          0],
        [TARGET_W-1, 0],
        [TARGET_W-1, TARGET_H-1],
        [0,          TARGET_H-1]
    ], dtype=np.float32)

    # Perspective warp
    M = cv2.getPerspectiveTransform(box, dst)
    warped_bgr = cv2.warpPerspective(bgr_img, M, (TARGET_W, TARGET_H), flags=cv2.INTER_CUBIC)
    warped_mask = cv2.warpPerspective(mask_u8, M, (TARGET_W, TARGET_H), flags=cv2.INTER_NEAREST)

    # Clean mask a bit
    warped_mask = cv2.medianBlur(warped_mask, 5)

    # ✅ Ensure final card is taller than wide (portrait)
    h_out, w_out = warped_bgr.shape[:2]
    if h_out < w_out:
        # rotate both image and mask 90° so height > width
        warped_bgr = cv2.rotate(warped_bgr, cv2.ROTATE_90_CLOCKWISE)
        warped_mask = cv2.rotate(warped_mask, cv2.ROTATE_90_CLOCKWISE)
        h_out, w_out = warped_bgr.shape[:2]

    # Build RGBA output
    rgba = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = warped_mask  # alpha from mask

    return rgba



def main():
    if not RAW_DIR.exists():
        print(f"[ERROR] Input folder not found: {RAW_DIR}")
        return

    # Load SAM mask generator
    mask_gen, device = load_sam_automatic(
        model_type="vit_b",
        checkpoint=str(ROOT / "weights" / "sam_vit_b.pth")
    )
    # Load SAM mask generator on CPU to avoid GPU OOM
    # mask_gen, device = load_sam_automatic(
    #     model_type="vit_b",
    #     checkpoint=str(ROOT / "weights" / "sam_vit_b.pth"),
    #     device="cpu"
    # )

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Reading from: {RAW_DIR}")
    print(f"[INFO] Writing PNG cards to: {OUT_DIR}")
    print(f"[INFO] Accepting masks with ratio in ({RATIO_MIN}, {RATIO_MAX})")

    img_paths = [p for p in RAW_DIR.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    if not img_paths:
        print("[WARN] No input images found in realjpg/. Add photos and run again.")
        return

    total_cards = 0

    for img_path in img_paths:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[WARN] Could not read {img_path.name}, skipping.")
            continue

        # 🔽 Downscale very large images to avoid GPU OOM
        H, W = bgr.shape[:2]
        max_side = max(H, W)
        if max_side > 1200:
            scale = 1200 / max_side
            new_w = int(W * scale)
            new_h = int(H * scale)
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            H, W = bgr.shape[:2]
            H, W = bgr.shape[:2]

        img_area = H * W

        masks = generate_masks(mask_gen, bgr)
        print(f"[INFO] {img_path.name}: SAM returned {len(masks)} masks before filtering.")

        keep_count = 0
        for i, m in enumerate(masks):
            if not is_card_mask(m, img_area):
                continue

            seg = m.get("segmentation", None)
            if seg is None:
                continue

            card_rgba = warp_card_from_mask(bgr, seg)
            if card_rgba is None:
                continue

            # Save PNG (user can rename later to 2C, KD, etc.)
            out_name = f"{img_path.stem}_card{i}.png"
            out_path = OUT_DIR / out_name
            Image.fromarray(card_rgba).save(out_path)
            keep_count += 1
            total_cards += 1

        print(f"[INFO] {img_path.name}: kept {keep_count} card(s) after ratio filter.")

    print(f"[DONE] Extracted {total_cards} card PNG(s) total into {OUT_DIR}")


if __name__ == "__main__":
    main()
