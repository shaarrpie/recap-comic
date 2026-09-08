# webapp/panel_api.py
"""Panel Review state for recap-comic.

Read-only layer over `panels.json` (the AI's segmentation — never rewritten
here) plus a recoverable user override layer persisted to
`<session>/panels_edit.json`:

    {version, deleted:[id], order:[id...], review:{id:status},
     confirmed:bool, confirmed_at:ts}

Nothing in this module mutates `panels.json`, so the AI baseline is always
intact and every user edit (reorder / delete / restore / review / confirm) is
recoverable.  Panel ids stay stable; only a *display order* is renumbered.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import HTTPException

import re

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "webapp_output"
_SESSION_RE = re.compile(r"^[0-9a-f]{12}$")

VALID_REVIEW = {"needs_review", "reviewed", "edited"}


def _session_dir(session: str) -> Path:
    if not _SESSION_RE.match(session):
        raise HTTPException(400, "invalid session id")
    d = (OUTPUT_DIR / session).resolve()
    base = OUTPUT_DIR.resolve()
    if base not in d.parents and d != base:
        raise HTTPException(400, "invalid session path")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_panels_json(session: str) -> list[dict] | None:
    p = _session_dir(session) / "panels.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text("utf-8")).get("panels", [])
    except Exception:
        return None


def _edit_path(session: str) -> Path:
    return _session_dir(session) / "panels_edit.json"


def _read_edit(session: str) -> dict:
    p = _edit_path(session)
    if p.is_file():
        try:
            e = json.loads(p.read_text("utf-8"))
            e.setdefault("custom", [])
            return e
        except Exception:
            pass
    return {"version": 1, "deleted": [], "order": [], "review": {},
            "custom": [], "confirmed": False, "confirmed_at": None}


def _write_edit(session: str, edit: dict) -> None:
    p = _edit_path(session)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(edit, indent=2), "utf-8")
    tmp.replace(p)


def _all_panels(session: str, edit: dict) -> dict:
    """AI panels + user-created custom panels, keyed by id."""
    panels = _read_panels_json(session) or []
    by_id = {p["id"]: dict(p) for p in panels}
    for c in edit.get("custom", []):
        by_id[c["id"]] = dict(c)
    return by_id


def _panel_image_exists(panel: dict, session_root: Path) -> bool:
    """True iff `panel['image_file']` resolves to a real file on disk.

    Used to filter out panels whose PNG was dropped by the cutter (e.g. a
    piece that was too thin and skipped during guided_cut). Returning True
    for panels with no image_file (custom panels) keeps them visible.
    """
    name = panel.get("image_file")
    if not name:
        return True
    return (session_root / name).is_file()


def get_panels(session: str) -> dict:
    """AI panels merged with the edit layer -> user-facing review list."""
    panels = _read_panels_json(session)
    if not panels:
        raise HTTPException(404, "no panels yet; run segmentation first")
    edit = _read_edit(session)
    deleted = set(edit.get("deleted", []))
    order = edit.get("order", [])
    review = edit.get("review", {})
    by_id = _all_panels(session, edit)
    session_root = _session_dir(session)

    # Filter out AI panels whose PNG was never produced (too thin, zero-range,
    # or otherwise dropped by the cutter). The front-end would otherwise issue
    # a /files/.../panel_XXX.png request for every entry in panels.json and
    # flood the log with 404s. Custom panels are kept unconditionally — they
    # are user-owned and have no on-disk prerequisite.
    base_ids = [p["id"] for p in sorted(panels, key=lambda p: (p.get("y_start", 0), p.get("panel_index", 0)))
                if _panel_image_exists(p, session_root) or not p.get("image_file")]
    custom_ids = [c["id"] for c in edit.get("custom", [])]
    ordered = ([i for i in order if i in by_id and i in (set(base_ids) | set(custom_ids))]
               + [i for i in base_ids + custom_ids if i not in order])

    active = [i for i in ordered if i not in deleted]
    out = []
    for n, pid in enumerate(active, 1):
        out.append({**by_id[pid], "display_order": n, "deleted": False,
                    "review_status": review.get(pid, "needs_review")})
    for pid in ordered:
        if pid in deleted:
            d = dict(by_id[pid])
            d["display_order"] = None
            d["deleted"] = True
            d["review_status"] = "deleted"
            out.append(d)

    conf = edit.get("confirmed", False)
    low = sum(1 for p in out if not p["deleted"] and (p.get("confidence") or 0) < 0.6)
    return {
        "session": session,
        "total": len(by_id),
        "active": len(active),
        "stripped": len(by_id),
        "confirmed": conf,
        "custom_order": bool(edit.get("order")),
        "deleted_count": len(deleted),
        "needs_review_count": low,
        "panels": out,
    }


def set_order(session: str, ids: list[str]) -> dict:
    """Persist a full active-panel ordering. ids must cover every panel once."""
    panels = _read_panels_json(session)
    if not panels:
        raise HTTPException(404, "no panels yet")
    edit = _read_edit(session)
    have = set(_all_panels(session, edit))
    if set(ids) != have:
        raise HTTPException(400, "order must cover every detected panel exactly once")
    edit["order"] = list(ids)
    edit["confirmed"] = False
    edit["confirmed_at"] = None
    _write_edit(session, edit)
    return get_panels(session)


def delete_panels(session: str, ids: list[str]) -> dict:
    """Soft-delete (recoverable). panels.json untouched."""
    panels = _read_panels_json(session)
    if not panels:
        raise HTTPException(404, "no panels yet")
    have = set(_all_panels(session, _read_edit(session)))
    for i in ids:
        if i not in have:
            raise HTTPException(400, f"unknown panel {i}")
    edit = _read_edit(session)
    deleted = set(edit.get("deleted", []))
    deleted |= set(ids)
    edit["deleted"] = sorted(deleted)
    edit["confirmed"] = False
    edit["confirmed_at"] = None
    _write_edit(session, edit)
    return get_panels(session)


def restore_panels(session: str, ids: list[str]) -> dict:
    edit = _read_edit(session)
    doomed = set(ids)
    edit["deleted"] = [i for i in edit.get("deleted", []) if i not in doomed]
    edit["confirmed"] = False
    edit["confirmed_at"] = None
    _write_edit(session, edit)
    return get_panels(session)


def set_review(session: str, panel_id: str, status: str) -> dict:
    if panel_id not in _all_panels(session, _read_edit(session)):
        raise HTTPException(404, f"unknown panel {panel_id}")
    if status not in VALID_REVIEW:
        raise HTTPException(400, f"invalid review status {status!r}")
    edit = _read_edit(session)
    edit["review"][panel_id] = status
    _write_edit(session, edit)
    return get_panels(session)


def confirm(session: str, *, review_all: bool = False) -> dict:
    """Lock the current panel set/order as the narration source."""
    panels = _read_panels_json(session)
    if not panels:
        raise HTTPException(404, "no panels yet")
    edit = _read_edit(session)
    deleted = set(edit.get("deleted", []))
    active_ids = [i for i in edit.get("order", []) if i not in deleted]
    if not active_ids:
        active_ids = [p["id"] for p in sorted(panels, key=lambda p: p.get("y_start", 0))
                      if p["id"] not in deleted]
    if not active_ids:
        raise HTTPException(400, "no active panels to confirm")
    if review_all:
        review = dict(edit.get("review", {}))
        for pid in active_ids:
            review.setdefault(pid, "reviewed")
        edit["review"] = review
    edit["order"] = active_ids
    edit["confirmed"] = True
    edit["confirmed_at"] = time.time()
    _write_edit(session, edit)
    return get_panels(session)

# --------------------------------------------------------------------------
# Custom panel operations: duplicate / merge / split.
#
# Panels are vertical crops of the strip with a known y-range, so all three
# are real image operations (PIL), not bookkeeping fakes:
#   duplicate  -> byte-copy of the crop under a new stable id
#   merge      -> vertical stack of two adjacent crops, y-range union
#   split      -> horizontal cut of one crop at a fraction, two new y-ranges
#
# The originals are SOFT-DELETED (restore route recovers them); the derived
# panels live in the edit layer's `custom` list, so the AI baseline in
# panels.json is never mutated and every operation is reversible by deleting
# the derived panel and restoring the originals.
# --------------------------------------------------------------------------

def _alloc_custom_id(session: str, edit: dict) -> str:
    existing = set(_all_panels(session, edit))
    n = 1
    while f"u{n:03d}" in existing:
        n += 1
    return f"u{n:03d}"


def _materialize_order(session: str, edit: dict, replacement: dict) -> None:
    """Rewrite edit['order'] so each source id becomes its derived id(s),
    keeping every other active panel in place."""
    deleted = set(edit.get("deleted", []))
    by_id = _all_panels(session, edit)
    base = [p["id"] for p in sorted(_read_panels_json(session) or [],
                                    key=lambda p: (p.get("y_start", 0), p.get("panel_index", 0)))]
    current = [i for i in edit.get("order", []) if i in by_id] \
        or [i for i in base + [c["id"] for c in edit.get("custom", [])] if i in by_id]
    out: list[str] = []
    for pid in current:
        if pid in deleted:
            continue
        out.extend(replacement.get(pid, [pid]))
    for cid, ids in replacement.items():
        if cid not in current:          # source wasn't ordered; append derived
            out.extend(ids)
    edit["order"] = out


def _custom_entry(src: dict, new_id: str, image_file: str, *,
                  y_start: int, y_end: int, panel_type: str,
                  confidence: float, panel_index: int) -> dict:
    return {"id": new_id, "panel_index": panel_index, "panel_type": panel_type,
            "confidence": round(confidence, 4), "y_start": y_start, "y_end": y_end,
            "image_file": image_file, "narration": "", "dialogue": "",
            "custom": True, "derived_from": src.get("id")}


def _save_crop_image(session: str, data_src, dest,
                     box=None, stack=None) -> None:
    from PIL import Image
    if stack is not None:
        imgs = [Image.open(p) for p in stack]
        try:
            w = max(im.width for im in imgs)
            h = sum(im.height for im in imgs)
            canvas = Image.new("RGB", (w, h), (255, 255, 255))
            y = 0
            for im in imgs:
                canvas.paste(im, (0, y))
                y += im.height
            canvas.save(dest, "PNG")
        finally:
            for im in imgs:
                im.close()
        return
    with Image.open(data_src) as im:
        (im.crop(box) if box else im).save(dest, "PNG")

def duplicate_panel(session: str, panel_id: str) -> dict:
    by_id = _all_panels(session, _read_edit(session))
    src = by_id.get(panel_id)
    if src is None:
        raise HTTPException(404, f"unknown panel {panel_id}")
    edit = _read_edit(session)
    new_id = _alloc_custom_id(session, edit)
    dest_name = f"{new_id}.png"
    d = _session_dir(session)
    _save_crop_image(session, d / src["image_file"], d / dest_name)
    entry = _custom_entry(src, new_id, dest_name,
                          y_start=src["y_start"], y_end=src["y_end"],
                          panel_type=src.get("panel_type", "panel"),
                          confidence=src.get("confidence", 1.0),
                          panel_index=src.get("panel_index", 0))
    entry["derived_from"] = panel_id
    edit.setdefault("custom", []).append(entry)
    _materialize_order(session, edit, {panel_id: [panel_id, new_id]})
    _write_edit(session, edit)
    return get_panels(session)


def merge_panels(session: str, ids: list[str]) -> dict:
    if len(ids) != 2:
        raise HTTPException(400, "merge takes exactly two panel ids")
    by_id = _all_panels(session, _read_edit(session))
    a, b = by_id.get(ids[0]), by_id.get(ids[1])
    if a is None or b is None:
        raise HTTPException(404, "unknown panel id in merge")
    if min(a["y_start"], a["y_end"]) > min(b["y_start"], b["y_end"]):
        a, b = b, a                              # a = upper panel
    edit = _read_edit(session)
    new_id = _alloc_custom_id(session, edit)
    dest_name = f"{new_id}.png"
    d = _session_dir(session)
    _save_crop_image(session, None, d / dest_name,
                     stack=[d / a["image_file"], d / b["image_file"]])
    entry = _custom_entry(a, new_id, dest_name,
                          y_start=a["y_start"], y_end=b["y_end"],
                          panel_type=a.get("panel_type", "panel"),
                          confidence=min(a.get("confidence", 1.0),
                                         b.get("confidence", 1.0)),
                          panel_index=a.get("panel_index", 0))
    entry["derived_from"] = [a["id"], b["id"]]
    edit.setdefault("custom", []).append(entry)
    edit["deleted"] = sorted(set(edit.get("deleted", [])) | {a["id"], b["id"]})
    _materialize_order(session, edit, {a["id"]: [new_id], b["id"]: []})
    _write_edit(session, edit)
    return get_panels(session)


def split_panel(session: str, panel_id: str, *, fraction: float = 0.5) -> dict:
    if not 0.15 <= fraction <= 0.85:
        raise HTTPException(400, "fraction must be within 0.15-0.85")
    by_id = _all_panels(session, _read_edit(session))
    src = by_id.get(panel_id)
    if src is None:
        raise HTTPException(404, f"unknown panel {panel_id}")
    d = _session_dir(session)
    from PIL import Image
    with Image.open(d / src["image_file"]) as im:
        w, h = im.size
        cut = max(1, min(h - 1, round(h * fraction)))
    edit = _read_edit(session)
    id1 = _alloc_custom_id(session, edit)
    id2 = f"u{(int(id1[1:]) + 1):03d}"
    if id2 in by_id:
        raise HTTPException(409, "could not allocate two adjacent custom ids")
    name1, name2 = f"{id1}.png", f"{id2}.png"
    _save_crop_image(session, d / src["image_file"], d / name1, box=(0, 0, w, cut))
    _save_crop_image(session, d / src["image_file"], d / name2, box=(0, cut, w, h))
    y0, y1 = src["y_start"], src["y_end"]
    span = max(1, y1 - y0)
    yc = y0 + round(span * cut / h)
    conf = src.get("confidence", 1.0)
    e1 = _custom_entry(src, id1, name1, y_start=y0, y_end=yc,
                       panel_type=src.get("panel_type", "panel"),
                       confidence=conf, panel_index=src.get("panel_index", 0))
    e2 = _custom_entry(src, id2, name2, y_start=yc, y_end=y1,
                       panel_type=src.get("panel_type", "panel"),
                       confidence=conf, panel_index=src.get("panel_index", 0))
    edit.setdefault("custom", []).extend([e1, e2])
    edit["deleted"] = sorted(set(edit.get("deleted", [])) | {panel_id})
    _materialize_order(session, edit, {panel_id: [id1, id2]})
    _write_edit(session, edit)
    return get_panels(session)

