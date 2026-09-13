# webapp/manual_crop_api.py
"""Manual panel cropping (no AI).

Boundaries are stored in `<session>/manual_crop.json`:

    {"mode": "manual", "boundaries": [0, 800, 1600, ...], "panel_ids": [...]}

Generation crops the original strip at those Y coordinates, writes panel PNGs
and a fresh `panels.json`, and backs up the previous auto-detected panels to
`panels_auto_backup.json` so Reset can restore them.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image

from guided_cutter import CutArtifact, CutPanel

from .panel_api import _read_panels_json, _session_dir, _write_edit, get_panels

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.environ.get("RECAP_OUTPUT_DIR") or BASE_DIR / "webapp_output")

log = logging.getLogger(__name__)


def _manual_path(session: str) -> Path:
    return _session_dir(session) / "manual_crop.json"


def _read_manual(session: str) -> dict:
    p = _manual_path(session)
    if p.is_file():
        try:
            data = json.loads(p.read_text("utf-8"))
            if data.get("mode") == "manual":
                return data
        except Exception:
            pass
    return {"mode": "manual", "boundaries": [], "panel_ids": []}


def _write_manual(session: str, data: dict) -> None:
    p = _manual_path(session)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), "utf-8")
    tmp.replace(p)


def get_manual_state(session: str) -> dict:
    return _read_manual(session)


def save_manual_boundaries(session: str, boundaries: list[int]) -> dict:
    # Validate up front (clamping at generate time used to accept
    # negative/float/oversized values silently): ints only, clamped to
    # the strip, and at least two distinct values are needed for a cut.
    if not isinstance(boundaries, list):
        raise HTTPException(400, "boundaries must be a list of y positions")
    clean: list[int] = []
    for b in boundaries:
        if isinstance(b, bool) or not isinstance(b, (int, float)):
            raise HTTPException(400, f"boundary {b!r} is not a number")
        if int(b) != b:
            raise HTTPException(400, f"boundary {b!r} must be an integer")
        clean.append(int(b))
    strip_path = _find_strip(_session_dir(session))
    if strip_path is not None:
        with Image.open(strip_path) as img:
            height = img.size[1]
        clean = [max(0, min(height, b)) for b in clean]
    clean = sorted(set(clean))
    if len(clean) < 2:
        raise HTTPException(400, "need at least two distinct boundaries "
                                 "(top and bottom of a panel)")
    data = _read_manual(session)
    data["boundaries"] = clean
    # ensure panel_ids matches boundary count (minus 1)
    n = max(0, len(clean) - 1)
    old_ids = data.get("panel_ids", [])
    if len(old_ids) != n:
        data["panel_ids"] = [str(uuid.uuid4())[:8] for _ in range(n)]
    _write_manual(session, data)
    return data


def get_auto_guides(session: str) -> list[int]:
    panels = _read_panels_json(session)
    if not panels:
        return []
    max_y = max(p.get("y_end", 0) for p in panels)
    boundaries = sorted(set([0] + [p.get("y_start", 0) for p in panels]
                            + [p.get("y_end", 0) for p in panels]))
    return [b for b in boundaries if 0 <= b <= max_y]


def _find_strip(session_dir: Path) -> Path | None:
    for name in ("strip.webp", "strip.png", "strip.jpg", "strip.jpeg"):
        p = session_dir / name
        if p.is_file():
            return p
    # Fallback must never pick up a PANEL crop (panel_*.png / s<id>_*.png):
    # generating manual panels from an already-cut panel image silently
    # produces nonsense. Only a non-panel-named image qualifies.
    for p in session_dir.iterdir():
        if (p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                and p.is_file()
                and not p.name.startswith("panel_")
                and not p.name.startswith("s")):
            return p
    return None


def generate_manual_panels(session: str) -> dict:
    d = _session_dir(session)
    data = _read_manual(session)
    raw = data.get("boundaries", [])
    strip_path = _find_strip(d)
    if strip_path is None:
        raise FileNotFoundError("no strip image found in session directory")

    # Guard: zero interior boundaries would produce ONE panel covering
    # the whole strip — almost never what the user wants, and it used to
    # happen silently (blank canvas + Generate = giant panel). Point the
    # user at Load Guides instead of guessing.
    if not raw:
        raise HTTPException(
            400, "no manual boundaries set — double-click the strip to "
                 "add cut lines (or press Load Guides to start from the "
                 "AI's detected edges), then Generate Panels")

    with Image.open(strip_path) as img:
        width, height = img.size
        rgb = img.convert("RGB")
        gray_arr = np.asarray(img.convert("L"))

    # clamp boundaries to strip height and deduplicate
    ys = sorted({0, height} | {max(0, min(height, int(b))) for b in raw})
    ranges = list(pairwise(ys))

    panels_json = d / "panels.json"
    backup = d / "panels_auto_backup.json"
    if not backup.exists() and panels_json.exists():
        backup.write_bytes(panels_json.read_bytes())

    panel_ids = data.get("panel_ids", [])
    if len(panel_ids) != len(ranges):
        panel_ids = [str(uuid.uuid4())[:8] for _ in range(len(ranges))]

    saved = []
    for i, (y0, y1) in enumerate(ranges):
        if y1 - y0 < 30:
            # thin range skipped: the id stays reserved for its position
            # so ids map 1:1 to ranges across generate runs (a truncated
            # list would re-shift every id and orphan review/narration
            # state keyed by the old ids).
            continue
        pid = panel_ids[i]
        dest_name = f"panel_{pid}.png"
        dest = d / dest_name
        crop = rgb.crop((0, y0, width, y1))
        crop.save(dest, "PNG")
        saved.append({
            "id": pid,
            "panel_index": i + 1,
            "y_start": y0,
            "y_end": y1,
            "narration": "",
            "dialogue": "",
            "panel_type": "panel",
            "confidence": 1.0,
            "image_file": dest_name,
        })

    artifact = CutArtifact(
        source=strip_path.name,
        width=width,
        height=height,
        plan_hash="manual",
        config={},
        panels=[CutPanel(**p) for p in saved],
    )
    panels_json.write_text(artifact.model_dump_json(indent=2) + "\n", "utf-8")

    # panel_ids keeps ALL range positions (skipped thin ranges keep their
    # reserved slot) so the id<->position mapping is stable.
    data["panel_ids"] = panel_ids
    _write_manual(session, data)
    # Manual cropping redefines the panel set: every downstream step
    # (validation/review/order/narration/render) is now stale.
    try:
        from . import checkpoint as _cp
        _cp.apply_edit_invalidation(session, "panels.json")
    except Exception:
        pass
    result = get_panels(session)
    # Manual cropping defines a NEW panel set; run validation once and
    # surface it to the review overlay.
    try:
        from panel_validator import save_report, validate_panels
        vreport = validate_panels(gray_arr, saved)
        save_report(d, vreport)
        result["validation"] = {
            "detected": vreport.detected,
            "accepted": vreport.accepted,
            "suspicious": vreport.suspicious,
            "rejected": vreport.rejected,
            "duplicates": vreport.duplicates,
            "verdicts": [{"panel_id": v.panel_id, "quality": v.quality, "reasons": v.reasons} for v in vreport.verdicts],
        }
    except Exception as exc:
        log.warning("manual panel validation skipped: %s", exc)
    return result


def reset_manual(session: str) -> dict:
    d = _session_dir(session)
    backup = d / "panels_auto_backup.json"
    panels_json = d / "panels.json"
    if backup.exists():
        panels_json.write_bytes(backup.read_bytes())
        backup.unlink()
    _manual_path(session).unlink(missing_ok=True)
    # The restored automatic panels have different ids than the manual ones;
    # drop any Panel Review state that referenced them, or get_panels()
    # would filter everything through stale order/deleted lists.
    edit_path = d / "panels_edit.json"
    if edit_path.is_file():
        try:
            edit = json.loads(edit_path.read_text("utf-8"))
            restored = _read_panels_json(session) or []
            ids = {p.get("id") for p in restored}
            edit["deleted"] = [i for i in edit.get("deleted", []) if i in ids]
            edit["order"] = [i for i in edit.get("order", []) if i in ids]
            edit["review"] = {k: v for k, v in edit.get("review", {}).items()
                              if k in ids}
            edit["custom"] = [c for c in edit.get("custom", [])
                              if c.get("id") in ids]
            edit["confirmed"] = False
            edit["confirmed_at"] = None
            _write_edit(session, edit)
        except Exception as exc:
            log.warning("reset_manual could not clean panels_edit.json: %s", exc)
    panels = _read_panels_json(session)
    if not panels:
        raise HTTPException(404, "no panels yet")
    return get_panels(session)


from fastapi import HTTPException  # noqa: E402
