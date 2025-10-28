def bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """Convert pixel bbox corners to YOLO format: class_id cx cy w h (normalized 0–1)."""
    bbox_w = xmax - xmin
    bbox_h = ymax - ymin
    cx = xmin + bbox_w / 2.0
    cy = ymin + bbox_h / 2.0
    return (
        cx / img_w,
        cy / img_h,
        bbox_w / img_w,
        bbox_h / img_h,
    )
