# src/segmentation/sam_utils.py
"""
Helper utilities for loading and running the Segment Anything Model (SAM)
inside the PokerVision project.
"""

from pathlib import Path
import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def load_sam_automatic(
    model_type: str = "vit_b",
    checkpoint: str = "weights/sam_vit_b.pth",
    device: str | None = None,
):
    """
    Load SAM and return an automatic mask generator.

    Parameters
    ----------
    model_type : str
        Backbone type: 'vit_b', 'vit_l', or 'vit_h'.
        'vit_b' (Base) is recommended for PokerVision — best speed/accuracy tradeoff.
    checkpoint : str
        Path to the SAM checkpoint file.
    device : str | None
        'cuda' (GPU) or 'cpu'. Defaults to auto-detect.

    Returns
    -------
    mask_generator : SamAutomaticMaskGenerator
        SAM mask generator ready to use.
    device : str
        The device SAM was loaded on.
    """
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SAM checkpoint not found at {ckpt_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Loading SAM ({model_type}) on {device} ...")

    sam = sam_model_registry[model_type](checkpoint=str(ckpt_path))
    sam.to(device)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,            # grid density
        pred_iou_thresh=0.86,          # quality threshold for mask prediction
        stability_score_thresh=0.92,   # filter unstable masks
        crop_n_layers=0,               # full-image run, no pyramid
        min_mask_region_area=500,      # ignore tiny blobs
    )

    print("[INFO] SAM ready.")
    return mask_generator, device


def generate_masks(mask_generator, bgr_image: np.ndarray):
    """
    Run SAM on a BGR image and return list of mask dictionaries.

    Each mask dictionary has keys:
      'segmentation', 'area', 'bbox', 'predicted_iou', etc.

    Parameters
    ----------
    mask_generator : SamAutomaticMaskGenerator
        Returned by load_sam_automatic().
    bgr_image : np.ndarray
        OpenCV image (BGR).

    Returns
    -------
    list[dict]
        SAM mask outputs.
    """
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(rgb)
    return masks
