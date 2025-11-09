from pathlib import Path
import sys
import cv2

# --- Add project root to path ---
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.segmentation.sam_utils import load_sam_automatic, generate_masks


# 1) Load SAM
try:
    mask_gen, device = load_sam_automatic(
        model_type="vit_b",
        checkpoint="weights/sam_vit_b.pth"
    )
    print("✅ SAM loaded on device:", device)
except Exception as e:
    print("❌ Failed to load SAM:", e)
    raise SystemExit

# 2) Try running SAM on one real photo if it exists
realjpg_dir = Path("data/raw_cards/realjpg")
imgs = list(realjpg_dir.glob("*.jpg")) + list(realjpg_dir.glob("*.png"))

if not imgs:
    print("ℹ️ No images found in data/raw_cards/realjpg yet.")
    print("   (This is OK for now — we just wanted to test loading.)")
else:
    img_path = imgs[0]
    print("Using test image:", img_path)
    img = cv2.imread(str(img_path))
    if img is None:
        print("❌ OpenCV failed to read the image.")
    else:
        masks = generate_masks(mask_gen, img)
        print(f"✅ SAM returned {len(masks)} masks for {img_path.name}")
