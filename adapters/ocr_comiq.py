# adapters/ocr_comiq.py
"""ComiQ adapter (BORROW decision for OCR-stage quality tier).

ComiQ (StoneSteel27/ComiQ, MIT, Python, 30 stars) wraps PP-OCRv6/EasyOCR with
an MLLM pass that groups word boxes into bubbles and classifies
dialogue/thought/narration/sound_effect. API below is quoted from the ComiQ
README (opened 2026-09-05): ComiQ(api_key=..., model_name=..., base_url=...)
and .extract(image, ocr="paddleocr"|"easyocr"|custom) returning dicts with
"text_box": [ymin, xmin, ymax, xmax] and "type": "dialogue"|"thought"|
"narration"|"sound_effect"|"background".

# UNVERIFIED AGAINST comiq==<latest> — not pip-installed in this session;
# run pilots/pilot_ocr_comiq.py to confirm before relying on it.
Note: ComiQ's MLLM grouping calls a paid API by default (MLLM_API_KEY env
var, Gemini-compatible endpoint); the offline fallback is ocr_tesseract.py.
"""
from __future__ import annotations

from pathlib import Path

from .schemas import BBox, OcrRegion

_KIND = {"dialogue": "dialogue", "thought": "narration",
         "narration": "narration", "sound_effect": "sfx",
         "background": "unknown"}


def extract_regions(image_path: Path, *, api_key: str | None = None,
                    page: int | None = None) -> list[OcrRegion]:
    import comiq  # imported lazily: heavy optional dependency

    cq = comiq.ComiQ(api_key=api_key)
    results = cq.extract(str(image_path))
    page_no = page if page is not None else int(image_path.stem)
    regions: list[OcrRegion] = []
    for n, r in enumerate(results, start=1):
        ymin, xmin, ymax, xmax = r["text_box"]
        regions.append(OcrRegion(
            id=f"{image_path.stem}.c{n:02d}", panel_id=None, page=page_no,
            bbox=BBox(x=int(xmin), y=int(ymin), w=int(xmax - xmin),
                      h=int(ymax - ymin)),
            text=r["text"], confidence=None,
            kind=_KIND.get(r.get("type", "unknown"), "unknown")))
    return regions
