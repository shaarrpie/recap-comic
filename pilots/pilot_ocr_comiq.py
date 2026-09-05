# pilots/pilot_ocr_comiq.py
"""Pilot for the ComiQ OCR adapter. NOT run in this session (needs the comiq
package + PP-OCR model weights + an MLLM key for bubble grouping).
Run after: pip install comiq paddleocr  (see its README for exact extras)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.ocr_comiq import extract_regions

if __name__ == "__main__":
    image = Path(sys.argv[1])
    for r in extract_regions(image):
        print(r.id, r.kind, r.bbox, repr(r.text))
