# report.py
"""Self-contained HTML report for reviewing a guided cut.

Emits a single HTML page showing each output panel image (base64-embedded so
the file is portable) next to its narration + dialogue, in reading order.
This is the review surface for `guided cut --report report.html`.
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

from guided_cutter import CutArtifact

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Guided Cut Report &mdash; {source}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#fafafa;color:#222;}}
h1{{font-size:20px;}}
.meta{{color:#666;margin-bottom:16px;}}
.panel{{display:flex;gap:18px;margin:0 0 28px;padding:14px;border:1px solid #ddd;
        border-radius:8px;background:#fff;}}
.panel img{{max-width:320px;max-height:570px;border:1px solid #ccc;}}
.panel .body{{flex:1;}}
.panel h2{{margin:0 0 6px;font-size:16px;}}
.panel .type{{color:#888;font-size:12px;}}
.panel .conf{{color:#888;font-size:12px;}}
.panel .y-range{{color:#888;font-size:12px;}}
.panel .narration{{margin:8px 0;white-space:pre-wrap;}}
.panel .dialogue{{color:#384;}}
.empty{{color:#aaa;font-style:italic;}}
</style>
</head>
<body>
<h1>Guided Cut Report &mdash; {source}</h1>
<div class="meta">{n_panels} panels &middot; {width}&times;{height} &middot; renderer config: {config}</div>
{panels}
</body>
</html>
"""


def _embed_png(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" alt="{path.name}">'


def render_report(artifact: CutArtifact, out_path: Path,
                  out_dir: Path | None = None) -> Path:
    """Write a self-contained HTML report. Panel images are resolved relative
    to out_dir (the cut output directory), independent of where the report
    file itself is written."""
    out_path = Path(out_path)
    out_dir = Path(out_dir) if out_dir is not None else out_path.parent
    panels_html: list[str] = []
    for p in artifact.panels:
        img_path = out_dir / p.image_file
        img = (_embed_png(img_path) if img_path.exists()
               else '<div class="empty">image missing</div>')
        narration = (html.escape(p.narration)
                     if p.narration.strip()
                     else '<span class="empty">(no narration)</span>')
        dialogue = (f'<div class="dialogue"><b>Dialogue:</b> {html.escape(p.dialogue)}</div>'
                    if p.dialogue.strip() else "")
        panels_html.append(f"""\
<div class="panel">
  {img}
  <div class="body">
    <h2>{html.escape(str(p.id))}</h2>
    <div class="type">type: {html.escape(str(p.panel_type))}</div>
    <div class="conf">confidence: {p.confidence:.2f}</div>
    <div class="y-range">Y: {p.y_start}&ndash;{p.y_end}</div>
    <div class="narration">{narration}</div>
    {dialogue}
  </div>
</div>""")
    html_out = HTML_TEMPLATE.format(
        source=html.escape(artifact.source), n_panels=len(artifact.panels),
        width=artifact.width, height=artifact.height,
        config=html.escape(str(artifact.config)), panels="\n".join(panels_html))
    out_path.write_text(html_out, encoding="utf-8")
    return out_path
