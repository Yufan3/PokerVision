from pathlib import Path
from PIL import Image, ImageOps

# where your normal extracted cards live (52 PNGs like AS.png, 2D.png, etc.)
ORIG_DIR = Path("data/raw_cards")
# where we'll save inverted versions
INV_DIR = ORIG_DIR / "inverted"
INV_DIR.mkdir(parents=True, exist_ok=True)

def is_card_filename(name: str) -> bool:
    # valid names look like "2D.png", "TH.png", "AS.png"
    # (rank in 2-9,T,J,Q,K,A) + (suit C,D,H,S)
    if not name.lower().endswith(".png"):
        return False
    stem = name[:-4]  # remove .png
    if len(stem) != 2:
        return False
    rank, suit = stem[0], stem[1]
    valid_ranks = "23456789TJQKA"
    valid_suits = "CDHS"
    return (rank in valid_ranks) and (suit in valid_suits)

def invert_card(img: Image.Image) -> Image.Image:
    """
    Invert colors but keep transparency.
    - Split alpha
    - Invert only RGB
    """
    img = img.convert("RGBA")
    rgb, alpha = img.split()[:3], img.split()[3]

    # merge rgb to RGB image
    rgb_img = Image.merge("RGB", rgb)
    # invert RGB channels
    inv_rgb = ImageOps.invert(rgb_img)
    # put alpha back
    inv_rgba = Image.merge("RGBA", (*inv_rgb.split(), alpha))
    return inv_rgba

def main():
    for file in ORIG_DIR.iterdir():
        if not file.is_file():
            continue
        if not is_card_filename(file.name):
            continue

        img = Image.open(file).convert("RGBA")
        inv = invert_card(img)

        out_path = INV_DIR / file.name  # same name, different folder
        inv.save(out_path)
        print(f"Inverted {file.name} -> {out_path}")

    print("Done. Check data/raw_cards/inverted/")

if __name__ == "__main__":
    main()
