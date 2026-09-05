# samples/make_sample_strip.py
"""Generates a synthetic tall manhwa-style strip for offline demos/tests.

800 px wide x 4400 px tall: 4 panels, clean white gutters, one tall panel
(1916..3700) with an internal gutter so --max-panel-height splitting is
exercisable, and one white speech bubble in panel 1.
Run: python samples/make_sample_strip.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

WIDTH = 800
HEIGHT = 4400


def make_sample_strip(path: str | Path) -> Path:
    rng = np.random.default_rng(7)
    arr = np.full((HEIGHT, WIDTH), 240, dtype=np.uint8)

    def art(y0: int, y1: int) -> None:
        arr[y0:y1] = rng.integers(40, 215, (y1 - y0, WIDTH), dtype=np.uint8)

    def gutter(y0: int, y1: int) -> None:
        arr[y0:y1] = 255  # clean white gutter -> zero row variance

    # layout: panels, gutters, internal gutter, speech bubble
    art(40, 1100)                       # panel 1
    bubble = (60, 300, 320, 420)
    x0, y0, x1, y1 = bubble
    arr[y0:y1, x0:x1] = 255             # white bubble interior
    arr[y0, x0:x1] = 0
    arr[y1 - 1, x0:x1] = 0
    arr[y0:y1, x0] = 0
    arr[y0:y1, x1 - 1] = 0              # black outline

    gutter(1100, 1116)
    art(1116, 1900)                     # panel 2
    gutter(1900, 1916)
    art(1916, 3700)                     # panel 3 (tall: 1784 px > 1600 default)
    gutter(2800, 2816)                  # internal gutter near midpoint
    gutter(3700, 3716)
    art(3716, 4400)                     # panel 4

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out, "PNG")
    return out


if __name__ == "__main__":
    dest = Path(__file__).resolve().parent / "sample_strip.png"
    make_sample_strip(dest)
    print("wrote", dest)