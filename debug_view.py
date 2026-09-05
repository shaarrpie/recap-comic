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

import base64
import html
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from guided_cutter import CutArtifact, CutPanel
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
                draw.text((w - 200, max(0, c.y_start + 4)), c.id,
                          fill=(0, 140, 0), font=small)

        # bubble boxes (blue)
        for e in plan.entries:
            for b in e.bubble_boxes:
                x0, y0 = max(0, b.x), max(0, b.y)
                x1, y1 = min(w, b.x + b.w), min(h, b.y + b.h)
                draw.rectangle((x0, y0, x1, y1), outline=(0, 0, 255), width=3)

        rgb.save(out, "PNG")
    return out


def _panel_block(p: CutPanel, out_dir: Path) -> str:
    img_path = out_dir / p.image_file
    if img_path.exists():
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        img_html = (f'<img class="panel-img" '
                    f'src="data:image/png;base64,{b64}" '
                    f'alt="{html.escape(p.id)}">')
    else:
        img_html = f'<div class="missing">{html.escape(p.image_file)} missing</div>'
    notes: list[str] = []
    if p.merged_with:
        notes.append("merged: " + ", ".join(str(i) for i in p.merged_with))
    if p.split_of:
        notes.append(f"split of {p.split_of}")
    notes.append(f"y {p.y_start}..{p.y_end} ({p.y_end - p.y_start}px)")
    notes.append(f"type={p.panel_type} conf={p.confidence:.2f}")
    if p.snap_distances:
        notes.append("snap(px)=" + ",".join(str(d) for d in p.snap_distances))
    return f"""<div class="panel">
  {img_html}
  <div class="info">
    <h3>{html.escape(p.id)} — {html.escape(p.panel_type)}</h3>
    <p class="narration">{html.escape(p.narration)}</p>
    <p class="dialogue">{html.escape(p.dialogue)}</p>
    <p class="meta">{", ".join(html.escape(n) for n in notes)}</p>
  </div>
</div>"""


def write_panel_report(artifact: CutArtifact, out_dir: str | Path,
                       report_path: str | Path) -> Path:
    """Self-contained HTML review page: panels in reading order + their text."""
    out_dir = Path(out_dir)
    panels = sorted(artifact.panels, key=lambda c: (c.y_start, c.id))
    body = "\n".join(_panel_block(p, out_dir) for p in panels)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Guided cut report - {html.escape(artifact.source)}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #111; color: #eee; }}
 h1 {{ font-size: 1.15rem; }} .meta {{ color: #888; font-size: .85rem; }}
 .panel {{ display: flex; gap: 1rem; padding: 1rem 0; border-bottom: 1px solid #333; }}
 .panel-img {{ max-height: 520px; max-width: 300px; object-fit: contain; border: 1px solid #444; background:#000; }}
 .missing {{ color:#f88; }}
 .info {{ flex: 1; }} .narration {{ font-size: 1.05rem; }}
 .dialogue {{ color: #9cf; }} .meta {{ color: #888; font-size: .8rem; }}
</style></head><body>
<h1>Guided cut report - {html.escape(artifact.source)}
  <span class="meta">({len(panels)} panels, {artifact.width}x{artifact.height})</span></h1>
{body}
</body></html>"""
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(page, encoding="utf-8")
    return report