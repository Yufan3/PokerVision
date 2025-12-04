import cv2
import glob
import os
import numpy as np

# where your seg dataset lives
BASE = "data/yolo_cards_seg"
IMG_DIR = os.path.join(BASE, "images")
LBL_DIR = os.path.join(BASE, "labels")

# max size of displayed window (adapt if your screen is smaller)
MAX_W, MAX_H = 1400, 900

def draw_polys(img_path, lbl_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] cannot read {img_path}")
        return None

    h, w = img.shape[:2]

    if not os.path.exists(lbl_path):
        return img

    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= 3:
                continue
            floats = list(map(float, parts))
            poly = floats[1:]  # <-- everything after class id is polygon
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            pts = pts.astype(int)
            cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)



    return img

def main():
    img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    if not img_paths:
        print("[ERROR] No images found in", IMG_DIR)
        return

    cv2.namedWindow("seg_check", cv2.WINDOW_NORMAL)

    for img_path in img_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(LBL_DIR, stem + ".txt")

        img = draw_polys(img_path, lbl_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        scale = min(MAX_W / w, MAX_H / h, 1.0)
        if scale < 1.0:
            disp = cv2.resize(img, (int(w * scale), int(h * scale)))
        else:
            disp = img

        cv2.imshow("seg_check", disp)
        print("Showing", os.path.basename(img_path), "- press any key for next, or q/ESC to quit")

        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
