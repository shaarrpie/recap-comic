# adapters/panels_opencv.py
"""OpenCV panel detection adapter (BUILD decision, offline, CPU-only).

Why not plain Otsu thresholding: on a white-gutter page the gutter is the
bright background, so "bright = background" works. On a dark/full-bleed page
the gutter is the DARKEST thing in the image, so Otsu inverts the mask and
merges all panels into one blob. This adapter therefore estimates the gutter
colour from the image border (median of the outer 8px frame) and builds the
content mask as "distance from border colour", which works for both gutter
polarities and never crops panel content.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .schemas import (
    SCHEMA_VERSION,
    BBox,
    Meta,
    Panel,
    PanelsArtifact,
)


class PanelsError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_hash(cfg: dict) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def detect_panels(gray: np.ndarray, *, tol: int = 28,
                  min_area_frac: float = 0.02) -> list[tuple[int, int, int, int]]:
    """Return (x, y, w, h) boxes sorted top-to-bottom, left-to-right."""
    h, w = gray.shape
    frame = np.concatenate([
        gray[:8].ravel(), gray[-8:].ravel(),
        gray[:, :8].ravel(), gray[:, -8:].ravel(),
    ])
    border = int(np.median(frame))
    mask = (np.abs(gray.astype(np.int16) - border) > tol).astype(np.uint8) * 255
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE,
                              np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh < min_area_frac * w * h:
            continue
        boxes.append((x, y, bw, bh))
    boxes.sort(key=lambda b: (b[1], b[0]))  # reading order: top_to_bottom
    return boxes


def run_panels(pages_dir: Path, out_path: Path, *,
               reading_order: str = "top_to_bottom",
               tol: int = 28, min_area_frac: float = 0.02,
               force: bool = False) -> PanelsArtifact:
    """Produce panels.json. Reads only pages/, writes only out_path."""
    cfg = {"reading_order": reading_order, "tol": tol,
           "min_area_frac": min_area_frac}
    images = sorted(p for p in pages_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise PanelsError(f"no supported images in {pages_dir}")
    if not force and out_path.exists():
        existing = PanelsArtifact.model_validate_json(out_path.read_text("utf-8"))
        new_hashes = {p.name: _sha256(p) for p in images}
        if (existing.meta.schema_version == SCHEMA_VERSION
                and existing.meta.config_hash == _config_hash(cfg)
                and existing.meta.input_hashes == new_hashes):
            return existing  # cache hit

    panels: list[Panel] = []
    for page_no, img_path in enumerate(images, start=1):
        data = np.frombuffer(img_path.read_bytes(), dtype=np.uint8)
        gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise PanelsError(f"unreadable image: {img_path}")
        boxes = detect_panels(gray, tol=tol, min_area_frac=min_area_frac)
        if not boxes:
            raise PanelsError(
                f"no panels detected on page {page_no} ({img_path.name}); "
                "the page may be an unsegmentable full-bleed spread - handle "
                "it explicitly (e.g. emit one full-page panel) rather than "
                "silently dropping it")
        for idx, (x, y, bw, bh) in enumerate(boxes, start=1):
            panels.append(Panel(
                id=f"{page_no:03d}.{idx:02d}", page=page_no, index=idx,
                bbox=BBox(x=x, y=y, w=bw, h=bh),
                source_image=f"pages/{img_path.name}"))

    artifact = PanelsArtifact(
        meta=Meta(schema_version=SCHEMA_VERSION,
                  generator="panels_opencv",
                  config_hash=_config_hash(cfg),
                  input_hashes={p.name: _sha256(p) for p in images}),
        reading_order=reading_order,  # type: ignore[arg-type]
        pages=[p.name for p in images],
        panels=panels)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(artifact.model_dump_json(indent=2), "utf-8")
    tmp.replace(out_path)  # atomic write
    return artifact
