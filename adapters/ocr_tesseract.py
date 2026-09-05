# adapters/ocr_tesseract.py
"""Tesseract adapter (offline OCR fallback for the ocr stage).

pytesseract.image_to_data(..., output_type=pytesseract.Output.DICT) usage is
verified against the pytesseract 0.3.13 README (fetched 2026-09-05); the
returned dict keys used here ('text', 'conf', 'left', 'top', 'width',
'height', 'level', 'block_num', 'par_num', 'line_num') are the standard TSV
columns. Runtime requires the Tesseract BINARY on PATH (apt/winget/UB-Mannheim
installer); pytesseract raises TesseractNotFoundError otherwise.

Preprocessing (one reason each):
- upscale 2x cubic: Tesseract is trained ~300 dpi; webtoon lettering is small.
- grayscale: drops colour noise, Tesseract works on grey.
NOT used (cargo-cult for antialiased comic lettering): adaptive threshold and
dilate-to-merge-letters, which destroy antialiasing and frequently reduce
accuracy on comic fonts. (Claim is qualitative, not a benchmark.)

SFX policy: a line whose text is alphabetic, ALL-CAPS, >= 2 chars and whose
glyph height is >= sfx_min_height px (post-upscale) is marked kind="sfx" so
narration can ignore it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from .schemas import SCHEMA_VERSION, BBox, Meta, OcrArtifact, OcrRegion


def _hash_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _hash_cfg(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def extract_regions(image_path: Path, *, upscale: int = 2,
                    sfx_min_height: int = 60) -> list[OcrRegion]:
    data = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"unreadable image: {image_path}")
    big = cv2.resize(gray, None, fx=upscale, fy=upscale,
                     interpolation=cv2.INTER_CUBIC)
    d = pytesseract.image_to_data(big, output_type=pytesseract.Output.DICT)
    # group words into lines by (block_num, par_num, line_num)
    lines: dict[tuple, list[int]] = {}
    for i, txt in enumerate(d["text"]):
        if txt.strip() and int(d["conf"][i]) > 30:
            key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
            lines.setdefault(key, []).append(i)
    regions: list[OcrRegion] = []
    for n, (_, idxs) in enumerate(sorted(lines.items()), start=1):
        x = min(d["left"][i] for i in idxs)
        y = min(d["top"][i] for i in idxs)
        w = max(d["left"][i] + d["width"][i] for i in idxs) - x
        h = max(d["top"][i] + d["height"][i] for i in idxs) - y
        text = " ".join(d["text"][i] for i in idxs).strip()
        conf = float(np.mean([float(d["conf"][i]) for i in idxs]))
        letters_only = text.isalpha()
        is_sfx = letters_only and text.isupper() and len(text) >= 2 \
            and h >= sfx_min_height * upscale
        regions.append(OcrRegion(
            id=f"{image_path.stem}.r{n:02d}", panel_id=None,
            page=int(image_path.stem), bbox=BBox(x=x // upscale, y=y // upscale,
                                                 w=w // upscale, h=h // upscale),
            text=text, confidence=conf, kind="sfx" if is_sfx else "unknown"))
    return regions


def run_ocr(pages_dir: Path, panels_path: Path, out_path: Path, *,
            force: bool = False) -> OcrArtifact:
    cfg = {"backend": "tesseract", "upscale": 2, "sfx_min_height": 60}
    pages = sorted(p for p in pages_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    inputs = {"panels.json": _hash_file(panels_path), **{
        p.name: _hash_file(p) for p in pages}}
    if not force and out_path.exists():
        existing = OcrArtifact.model_validate_json(out_path.read_text("utf-8"))
        if existing.meta.schema_version == SCHEMA_VERSION and \
                existing.meta.config_hash == _hash_cfg(cfg) and \
                existing.meta.input_hashes == inputs:
            return existing  # cache hit
    regions: list[OcrRegion] = []
    for p in pages:
        regions.extend(extract_regions(p))
    artifact = OcrArtifact(
        meta=Meta(schema_version=SCHEMA_VERSION, generator="ocr_tesseract",
                  config_hash=_hash_cfg(cfg), input_hashes=inputs),
        backend="tesseract", regions=regions)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(artifact.model_dump_json(indent=2), "utf-8")
    tmp.replace(out_path)
    return artifact
