from pathlib import Path
import random
import shutil

def main():
    base = Path("data/yolo_cards_corners")
    images_dir = base / "images"
    labels_dir = base / "labels"

    # target dirs
    train_img_dir = images_dir / "train"
    val_img_dir   = images_dir / "val"
    train_lbl_dir = labels_dir / "train"
    val_lbl_dir   = labels_dir / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # all jpg images at top level (not inside train/val yet)
    all_images = [p for p in images_dir.glob("*.jpg")]
    all_images.sort()
    if not all_images:
        print("[ERROR] No images found in data/yolo_cards_seg/images/*.jpg")
        return

    random.seed(42)
    random.shuffle(all_images)

    n_total = len(all_images)
    n_train = int(0.85 * n_total)
    train_imgs = set(all_images[:n_train])
    val_imgs   = set(all_images[n_train:])

    print(f"Total images: {n_total}")
    print(f" Train: {len(train_imgs)}")
    print(f" Val  : {len(val_imgs)}")

    def move_pair(img_path, dst_img_dir, dst_lbl_dir):
        lbl_name = img_path.with_suffix(".txt").name
        lbl_path = labels_dir / lbl_name
        if not lbl_path.exists():
            print(f"[WARN] No label for {img_path.name}, skipping.")
            return
        shutil.move(str(img_path), str(dst_img_dir / img_path.name))
        shutil.move(str(lbl_path), str(dst_lbl_dir / lbl_name))

    for img_path in train_imgs:
        move_pair(img_path, train_img_dir, train_lbl_dir)
    for img_path in val_imgs:
        move_pair(img_path, val_img_dir, val_lbl_dir)

    print("Done splitting segmentation dataset.")

if __name__ == "__main__":
    main()