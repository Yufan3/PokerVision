from pathlib import Path
import shutil
import random

ROOT = Path(__file__).resolve().parents[2]
GEN_IMG = ROOT / "data" / "generated" / "images"
GEN_LBL = ROOT / "data" / "generated" / "labels"

OUT = ROOT / "data" / "yolo_cards_single"
OUT_IMG_TRAIN = OUT / "images" / "train"
OUT_IMG_VAL   = OUT / "images" / "val"
OUT_LBL_TRAIN = OUT / "labels" / "train"
OUT_LBL_VAL   = OUT / "labels" / "val"

VAL_FRACTION = 0.15  # 15% val

def write_singleclass(lbl_src: Path, lbl_dst: Path) -> bool:
    if not lbl_src.exists():
        return False
    lines_out = []
    for line in lbl_src.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        parts[0] = "0"  # force single class
        lines_out.append(" ".join(parts))
    if not lines_out:
        return False
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
    lbl_dst.write_text("\n".join(lines_out))
    return True

def main():
    # Create folders
    for d in [OUT_IMG_TRAIN, OUT_IMG_VAL, OUT_LBL_TRAIN, OUT_LBL_VAL]:
        d.mkdir(parents=True, exist_ok=True)

    imgs = sorted(GEN_IMG.glob("*.jpg"))
    random.seed(42)
    random.shuffle(imgs)
    n = len(imgs)
    k = int(n * (1 - VAL_FRACTION))
    train_imgs = imgs[:k]
    val_imgs   = imgs[k:]

    def process(split_imgs, img_dst, lbl_dst):
        kept = 0
        for img in split_imgs:
            lbl = GEN_LBL / f"{img.stem}.txt"
            out_img = img_dst / img.name
            out_lbl = lbl_dst / f"{img.stem}.txt"
            ok = write_singleclass(lbl, out_lbl)
            if not ok:
                continue
            out_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out_img)
            kept += 1
        return kept

    kt = process(train_imgs, OUT_IMG_TRAIN, OUT_LBL_TRAIN)
    kv = process(val_imgs,   OUT_IMG_VAL,   OUT_LBL_VAL)

    print(f"[DONE] single-class dataset at: {OUT}")
    print(f"      train: {kt}  |  val: {kv}")

if __name__ == "__main__":
    main()
