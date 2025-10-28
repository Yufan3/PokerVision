from pathlib import Path

# Root of the project: POKERVISION/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
BACKGROUNDS_DIR = DATA_DIR / "backgrounds"

GENERATED_DIR = DATA_DIR / "generated"
IMAGES_DIR = GENERATED_DIR / "images"
LABELS_DIR = GENERATED_DIR / "labels"

RAW_CARDS_DIR = DATA_DIR / "raw_cards"
NORMAL_DIR = RAW_CARDS_DIR / "normal"
INVERTED_DIR = RAW_CARDS_DIR / "inverted"
REAL_DIR = RAW_CARDS_DIR / "real"  # optional future folder for actual photos

# Where weights (trained YOLO checkpoints) will go
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Card label space in fixed order
RANKS = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]
SUITS = ["C","D","H","S"]  # Clubs, Diamonds, Hearts, Spades
CARD_CLASSES = [r + s for r in RANKS for s in SUITS]  # e.g. 2C, 2D, ..., AS

def ensure_dirs():
    """
    Create required directories if they don't exist.
    Safe to call at the start of dataset generation.
    """
    for d in [
        DATA_DIR,
        BACKGROUNDS_DIR,
        GENERATED_DIR,
        IMAGES_DIR,
        LABELS_DIR,
        RAW_CARDS_DIR,
        NORMAL_DIR,
        INVERTED_DIR,
        REAL_DIR,
        WEIGHTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
