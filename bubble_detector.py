# bubble_detector.py
"""Pixel-based speech-bubble detection (backup signal for the "never cut
through a bubble" rule).

Why it exists: the AI's bubble_boxes may be missing or wrong. This OpenCV
detector provides an independent, offline signal. A speech bubble is
characterised as a near-white region that:
  - is a closed, compact contour of sufficient area,
  - has a dark border (outline) close OUTSIDE the white region, and
  - has a "busy" interior (the dark text/glyphs inside create variance).
A plain white GUTTER fails all three tests (panel-scale, no dark outline,
flat) and is therefore ignored — that is exactly what separates bubbles from
gutters.

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
    ring_pad: int = 3,
) -> list[BBox]:
    """Return pixel bounding boxes of candidate speech bubbles in `gray`.

    The outline test samples the ring just OUTSIDE the bright blob. The
    contour traces the edge of the WHITE INTERIOR, so the bubble's dark
    outline lies outside it: an earlier version measured the ring *inside*
    the blob (`filled & ~eroded`), which is white by construction, so
    `ring_mean` was always ~255 and EVERY candidate was rejected — the
    "never cut through a bubble" backup signal was silently inert.
    """
    h, w = gray.shape
    bright = (gray > white_threshold).astype(np.uint8) * 255
    closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                              np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[BBox] = []
    kernel = np.ones((2 * ring_pad + 1, 2 * ring_pad + 1), np.uint8)
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

        # ROI padded so the outward ring around the blob stays in-bounds.
        x0, y0 = max(0, x - ring_pad), max(0, y - ring_pad)
        x1, y1 = min(w, x + bw + ring_pad), min(h, y + bh + ring_pad)
        roi = gray[y0:y1, x0:x1]
        filled = np.zeros(roi.shape, dtype=np.uint8)
        cv2.drawContours(filled, [c - np.array([x0, y0])], -1, 255,
                         thickness=cv2.FILLED)
        dilated = cv2.dilate(filled, kernel)
        ring = (dilated > 0) & (filled == 0)   # outline band, OUTSIDE the blob
        ring_mean = float(roi[ring].mean()) if ring.any() else 255.0
        if ring_mean > dark_border_max:
            continue  # no dark outline -> likely just a white area/gutter

        interior = roi[filled > 0]
        busyness = float(interior.std()) if interior.size else 0.0
        if busyness < min_busyness:
            continue  # flat white interior -> no text

        boxes.append(BBox(x=x, y=y, w=bw, h=bh))
    return boxes
