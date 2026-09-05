# pilots/pilot_panels.py
"""Offline pilot for the panels adapter on synthetic pages.
Exercises adapters/panels_opencv.py against (a) white gutters and
(b) black gutters. Success = correct panel count and top-to-bottom order.
NOTE: synthetic fixtures do not prove real-world accuracy on actual manhwa
pages (bleed, screen-tone, split panels).
Run:  python pilots/pilot_panels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.panels_opencv import detect_panels


def make_page(path: Path, bg: int, gutters: list[int]) -> None:
    """Vertical strip of noisy panels separated by uniform-colour gutters."""
    w, ph = 800, 340
    img = np.full((len(gutters) * (ph + 40) + 40, w), bg, dtype=np.uint8)
    y = 40
    for _ in gutters:
        img[y:y + ph, 40:w - 40] = np.random.default_rng(0).integers(
            30, 220, (ph, w - 80), dtype=np.uint8)
        y += ph + 40
    Image.fromarray(img).save(path)


def main() -> None:
    tmp = Path(__file__).parent / "_fixtures"
    tmp.mkdir(exist_ok=True)
    failures = 0
    for name, bg in (("white_gutter", 255), ("black_gutter", 10)):
        page = tmp / f"{name}.png"
        make_page(page, bg, [0, 1, 2])
        boxes = detect_panels(np.array(Image.open(page).convert("L")))
        ok = len(boxes) == 3 and [b[1] for b in boxes] == sorted(
            b[1] for b in boxes) and all(b[2] > 600 for b in boxes)
        print(f"{name}: {len(boxes)} panels, ys={[b[1] for b in boxes]} "
              f"-> {'PASS' if ok else 'FAIL'}")
        failures += 0 if ok else 1
    raise SystemExit(failures)


if __name__ == "__main__":
    main()
