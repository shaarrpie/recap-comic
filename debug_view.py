# debug_view.py
"""Visual debugging + review surfaces for the guided manhwa pipeline.

- draw_overlay(): renders the full strip with AI-proposed boundaries (red),
  the gutter-snapped final boundaries (green), panel IDs + confidence labels,
  and bubble boxes (blue). Used by `guided plan --debug-overlay` and by
  scripts/smoke_test_live.py.
- write_panel_report(): a single self-contained HTML page (images embedded as
  base64 data URIs) listing each output panel with its narration and dialogue
  in reading order. Used by `guided cut --report`.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from guided_cutter import CutPanel
from strip_analyzer import PanelPlan


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow without the size argument
        return ImageFont.load_default()


def draw_overlay(strip_path: str | Path, plan: PanelPlan, out_path: str | Path,
                 cuts: list[CutPanel] | None = None) -> Path:
    """Draw AI-proposed boundaries (red), snapped finals (green), bubbles.

    - red lines: every AI-proposed panel boundary (clip to strip);
    - green lines: the final gutter-snapped boundaries from build_cuts
      (omitted when cuts is None, e.g. in Phase 1 --dry-run);
    - labels: panel index + confidence + type (red), final cut id (green);
    - blue rectangles: every bubble box from the AI plan.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(strip_path) as img:
        img.load()
        rgb = img.convert("RGB")
        w, h = rgb.size
        draw = ImageDraw.Draw(rgb)
        small = _load_font(max(12, w // 50))

        # AI-proposed boundaries (red)
        for e in plan.entries:
            for y in (max(0, e.y_start), min(h, e.y_end)):
                draw.line([(0, y), (w, y)], fill=(255, 0, 0), width=4)
            label = f"#{e.panel_index} conf={e.confidence:.2f} {e.panel_type}"
            draw.text((6, max(0, e.y_start + 4)), label, fill=(200, 0, 0),
                      font=small)

        # final snapped boundaries (green) — omitted in Phase 1 (no cuts yet)
        if cuts is not None:
            for c in cuts:
                for y in (max(0, c.y_start), min(h, c.y_end)):
                    draw.line([(0, y), (w, y)], fill=(0, 190, 0), width=4)
                draw.text((max(0, w - 200), max(0, c.y_start + 4)), c.id,
                          fill=(0, 140, 0), font=small)

        # bubble boxes (blue)
        for e in plan.entries:
            for b in e.bubble_boxes:
                x0, y0 = max(0, b.x), max(0, b.y)
                x1, y1 = min(w, b.x + b.w), min(h, b.y + b.h)
                draw.rectangle((x0, y0, x1, y1), outline=(0, 0, 255), width=3)

        rgb.save(out, "PNG")
    return out


