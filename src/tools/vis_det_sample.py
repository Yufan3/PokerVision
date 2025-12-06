import cv2
import glob
import os
import numpy as np

# where your corner dataset lives
BASE = "data/yolo_cards_corners"
IMG_DIR = os.path.join(BASE, "images")
LBL_DIR = os.path.join(BASE, "labels")

MAX_W, MAX_H = 1400, 900

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


def draw_boxes(img_path, lbl_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] cannot read {img_path}")
        return None

    h, w = img.shape[:2]

    if not os.path.exists(lbl_path):
        return img

    with open(lbl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                # YOLO det format: cls cx cy w h
                continue

            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])

            # convert normalized cx,cy,w,h -> pixel x1,y1,x2,y2
            x_center = cx * w
            y_center = cy * h
            box_w = bw * w
            box_h = bh * h

            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if 0 <= cls_id < len(LABEL_ORDER):
                label = LABEL_ORDER[cls_id]
            else:
                label = str(cls_id)

            cv2.putText(
                img,
                label,
                (x1, max(18, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    return img


def main():
    img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    if not img_paths:
        print("[ERROR] No images found in", IMG_DIR)
        return

    cv2.namedWindow("det_check", cv2.WINDOW_NORMAL)

    for img_path in img_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(LBL_DIR, stem + ".txt")

        img = draw_boxes(img_path, lbl_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        scale = min(MAX_W / w, MAX_H / h, 1.0)
        disp = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img

        cv2.imshow("det_check", disp)
        print("Showing", os.path.basename(img_path), "- press any key for next, or q/ESC to quit")

        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
