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

# Existing styles
NORMAL_DIR       = RAW_CARDS_DIR / "normal"
INVERTED_DIR     = RAW_CARDS_DIR / "inverted"

# Real card styles (you can comment out any you don’t like)
REAL1_DIR        = RAW_CARDS_DIR / "real1"
REAL2_DIR        = RAW_CARDS_DIR / "real2"
REAL_INVERTED_DIR = RAW_CARDS_DIR / "realinverted"

# (Optional: if your old `real` folder is still good, you can keep it)
# REAL_DIR        = RAW_CARDS_DIR / "real"

# A single list of all style directories the generator should use
CARD_STYLE_DIRS = [
    NORMAL_DIR,
    INVERTED_DIR,
    REAL1_DIR,
    REAL2_DIR,
    REAL_INVERTED_DIR,
    # REAL_DIR,   # uncomment if you also want to include data/raw_cards/real
]

# Group styles into 3 logical types and set sampling weights
STYLE_GROUP_DIRS = {
    "normal":   [NORMAL_DIR],
    "inverted": [INVERTED_DIR],
    "real":     [REAL1_DIR, REAL2_DIR, REAL_INVERTED_DIR],
}

# Desired probabilities when sampling a card style
STYLE_WEIGHTS = {
    "normal":   0.20,  # 20%
    "inverted": 0.10,  # 10%
    "real":     0.70,  # 70% (across all real* dirs)
}

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
        CARD_STYLE_DIRS,
        WEIGHTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
