# bubble_detector.py
"""Pixel-based speech-bubble detection (backup signal for the "never cut
through a bubble" rule).

Why it exists: the AI's bubble_boxes may be missing or wrong. This OpenCV
detector provides an independent, offline signal. A speech bubble is
characterised as a near-white region that:
  - is a closed, compact contour of sufficient area,
  - has a dark border (outline), and
  - has a "busy" interior (the dark text/glyphs inside create variance).
A plain white GUTTER fails all three tests (open, no dark border, flat) and
is therefore ignored — that is exactly what separates bubbles from gutters.

Heuristic thresholds (area, white level, border brightness, busyness) are
configurable but ship with conservative defaults; tune on real strips.
"""
from __future__ import annotations

import cv2
import numpy as np

from adapters.schemas import BBox


def detect_bubbles(
    gray: np.ndarray,
    *,
    min_area: int = 1500,
    white_threshold: int = 230,
    dark_border_max: int = 200,
    min_busyness: float = 5.0,
    min_bbox_side: int = 10,
) -> list[BBox]:
    """Return pixel bounding boxes of candidate speech bubbles in `gray`."""
    h, w = gray.shape
    bright = (gray > white_threshold).astype(np.uint8) * 255
    closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                              np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[BBox] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < min_bbox_side or bh < min_bbox_side or bw * bh <= 0:
            continue
        if area / (bw * bh) < 0.3:  # not compact / closed
            continue
        if bw > w * 0.6 or bh > h * 0.6:  # panel-scale, not bubble-scale
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 5.0:  # extremely elongated -> not a bubble
            continue

        offset = c - np.array([x, y])
        filled = np.zeros((bh, bw), dtype=np.uint8)
        cv2.drawContours(filled, [offset], -1, 255, thickness=cv2.FILLED)
        eroded = cv2.erode(filled, np.ones((3, 3), np.uint8))
        ring = (filled > 0) & (eroded == 0)
        region = gray[y:y + bh, x:x + bw]
        ring_mean = float(region[ring].mean()) if ring.any() else 255.0
        if ring_mean > dark_border_max:
            continue  # no dark outline -> likely just a white area/gutter

        interior = region[filled > 0]
        busyness = float(interior.std()) if interior.size else 0.0
        if busyness < min_busyness:
            continue  # flat white interior -> no text

        boxes.append(BBox(x=x, y=y, w=bw, h=bh))
    return boxes
