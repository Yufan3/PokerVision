import numpy as np
from collections import deque


def bbox_iou_xyxy(box1, box2):
    """
    Compute IoU between two [x1, y1, x2, y2] boxes.
    box1, box2: np.ndarray shape (4,)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    if inter <= 0:
        return 0.0

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0

    return inter / union


class Track:
    """
    Single tracked card.
    Stores:
      - track_id: persistent ID
      - bbox_xyxy: np.ndarray(4,)
      - polygon: np.ndarray(N, 2) in image coordinates
      - age: how many frames it has been successfully matched
      - lost_frames: frames since last match
      - conf_ema: exponentially smoothed confidence
    """

    def __init__(self, track_id, box, polygon, conf, smooth_factor=0.25):
        self.track_id = track_id
        self.bbox_xyxy = np.asarray(box, dtype=np.float32)
        self.polygon = np.asarray(polygon, dtype=np.float32)
        self.age = 1
        self.lost_frames = 0
        self.smooth_factor = float(smooth_factor)
        self.conf_ema = float(conf)  # start from first conf

    def update(self, box, polygon, conf):
        self.bbox_xyxy = np.asarray(box, dtype=np.float32)
        self.polygon = np.asarray(polygon, dtype=np.float32)
        self.age += 1
        self.lost_frames = 0

        c = float(conf)
        self.conf_ema = (
            (1.0 - self.smooth_factor) * self.conf_ema
            + self.smooth_factor * c
        )

    @property
    def stable_score(self):
        """
        Stable "tag confidence":
        - EMA of raw confidences
        - boosted gradually as age increases (up to 10 frames).
        """
        warmup = min(1.0, self.age / 10.0)
        return self.conf_ema * warmup


class CardTracker:
    """
    Very lightweight tracker for cards using box IoU.

    update(polys, boxes, confs) expects:
      polys: list of np.ndarray [Ni, 2] in image coords
      boxes: np.ndarray [N, 4] xyxy
      confs: np.ndarray [N]
    """

    def __init__(self, iou_thresh=0.5, max_lost=15, smooth_factor=0.25):
        self.iou_thresh = float(iou_thresh)
        self.max_lost = int(max_lost)
        self.smooth_factor = float(smooth_factor)

        self.tracks = {}      # track_id -> Track
        self.next_id = 0

    def update(self, polys, boxes, confs):
        """
        Returns a list of Track objects that were successfully
        matched/created in THIS frame (no stale tracks).
        """
        if boxes is None or len(boxes) == 0:
            # No detections this frame: increase lost_frames
            dead_ids = []
            for tid, trk in self.tracks.items():
                trk.lost_frames += 1
                if trk.lost_frames > self.max_lost:
                    dead_ids.append(tid)
            for tid in dead_ids:
                del self.tracks[tid]
            return []

        boxes = np.asarray(boxes, dtype=np.float32)
        confs = np.asarray(confs, dtype=np.float32)
        assert len(polys) == boxes.shape[0] == confs.shape[0]

        n = boxes.shape[0]
        used = [False] * n
        updated_ids = set()

        # --- 1) Match existing tracks by IoU ---
        for tid, trk in list(self.tracks.items()):
            best_iou = 0.0
            best_j = -1

            for j in range(n):
                if used[j]:
                    continue
                iou = bbox_iou_xyxy(trk.bbox_xyxy, boxes[j])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j >= 0 and best_iou >= self.iou_thresh:
                trk.update(
                    box=boxes[best_j],
                    polygon=polys[best_j],
                    conf=confs[best_j],
                )
                updated_ids.add(tid)
                used[best_j] = True
            else:
                trk.lost_frames += 1

        # --- 2) Kill tracks that have been lost too long ---
        dead_ids = [
            tid for tid, trk in self.tracks.items()
            if trk.lost_frames > self.max_lost
        ]
        for tid in dead_ids:
            del self.tracks[tid]

        # --- 3) Create new tracks for unmatched detections ---
        for j in range(n):
            if not used[j]:
                new_id = self.next_id
                self.next_id += 1

                self.tracks[new_id] = Track(
                    track_id=new_id,
                    box=boxes[j],
                    polygon=polys[j],
                    conf=confs[j],
                    smooth_factor=self.smooth_factor,
                )
                updated_ids.add(new_id)

        # --- 4) Return only tracks that were updated this frame ---
        active_tracks = [self.tracks[tid] for tid in updated_ids]
        return active_tracks
