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
import re
import time
from pathlib import Path

from fastapi import HTTPException

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "webapp_output"
_SESSION_RE = re.compile(r"^[0-9a-f]{12}$")

VALID_REVIEW = {"needs_review", "reviewed", "edited"}


def _session_dir(session: str, *, create: bool = False) -> Path:
    """Resolve (and optionally create) the session directory.

    Read paths (get_panels, guides, …) must NOT materialize directories:
    probing /api/panels/<hex-id> would otherwise create a phantom empty
    session that then shows up in /api/projects.
    """
    if not _SESSION_RE.match(session):
        raise HTTPException(400, "invalid session id")
    d = (OUTPUT_DIR / session).resolve()
    base = OUTPUT_DIR.resolve()
    if base not in d.parents and d != base:
        raise HTTPException(400, "invalid session path")
    if create or d.is_dir():
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


def chain_for(session: str) -> list[str]:
    """Continuation chain for a session, oldest first.

    Reads each session's persisted continuation.json (written by the
    merge_continuation pipeline stage), cycle-guarded. The chain EXCLUDES
    the given session (it lists the strips that come BEFORE it).
    """
    chain: list[str] = []
    cur: str | None = session
    seen: set[str] = set()
    while cur:
        if cur in seen:
            break
        seen.add(cur)
        link = _continuation_link_of(cur)
        if not link:
            break
        chain.append(link)
        cur = link
    chain.reverse()
    return chain


def _continuation_link_of(session: str) -> str | None:
    p = _session_dir(session) / "continuation.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text("utf-8")).get("continue_from")
    except Exception:
        return None


def _ns_id(session: str, panel_id: str) -> str:
    """Combined-view id: s<owner8>_<original_id> (stable, collision-free)."""
    return f"s{session[:8]}_{panel_id}"


def _parse_ns_id(ns: str) -> tuple[str, str] | None:
    """Inverse of _ns_id: (owner_session_prefix, original_panel_id) or None.

    Accepts both the full session form and the 8-char prefix form.
    """
    m = re.match(r"^s([0-9a-f]{8,})_(.+)$", ns)
    if not m:
        return None
    return m.group(1), m.group(2)


def _owner_session_of(prefix_or_id: str, chain: list[str]) -> str | None:
    """Resolve an owner prefix (8+ chars) to a full session id in the chain."""
    for s in chain:
        if s.startswith(prefix_or_id):
            return s
    return None


def _edit_path(session: str) -> Path:
    return _session_dir(session, create=True) / "panels_edit.json"


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


def get_panels(session: str, *, combined: bool = False) -> dict:
    """User-facing review list.

    combined=False: this strip's panels only (AI baseline + edit layer) —
    the view used while reviewing THIS strip and by the pipeline stages.

    combined=True: the full continuation sequence — this strip's panels
    plus every previous chained strip's CURRENT review state (their own
    edit layer applies; deleted stay deleted, review status preserved).
    Ids are namespaced s<owner8>_<id> so every operation can be routed
    back to the owning strip. This is what Panel Review shows so a new
    strip's panels appear AFTER the previous strips' panels, never
    replacing them.
    """
    if not combined:
        return _get_panels_one(session)

    chain = chain_for(session)
    all_out: list[dict] = []
    strips = []
    total_deleted = 0
    for strip in [*chain, session]:
        try:
            one = _get_panels_one(strip)
        except Exception:
            continue          # a chain strip without panels yet: skip
        for p in one["panels"]:
            q = dict(p)
            q["id"] = _ns_id(strip, p["id"])
            q["source_session"] = strip
            q["image_file"] = _ns_id(strip, p["image_file"])
            all_out.append(q)
        strips.append({"session": strip, "active": one["active"],
                       "confirmed": one["confirmed"]})
        total_deleted += one["deleted_count"]
    active = [p for p in all_out if not p["deleted"]]
    for n, p in enumerate(active, 1):
        p["display_order"] = n
    low = sum(1 for p in active if (p.get("confidence") or 0) < 0.6)
    return {
        "session": session,
        "strips": strips,
        "total": len(all_out),
        "active": len(active),
        "confirmed": all(s["confirmed"] for s in strips),
        "custom_order": bool(all_out) and _any_order(chain + [session]),
        "deleted_count": total_deleted,
        "needs_review_count": low,
        "panels": all_out,
    }


def _any_order(sessions: list[str]) -> bool:
    for s in sessions:
        edit = _read_edit(s)
        if edit.get("order"):
            return True
    return False


def _get_panels_one(session: str) -> dict:
    """Single-strip panel list (AI baseline merged with the edit layer)."""
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
        "confirmed": conf,
        "custom_order": bool(edit.get("order")),
        "deleted_count": len(deleted),
        "needs_review_count": low,
        "panels": out,
    }


def _resolve_panel(session: str, ns_or_raw_id: str) -> tuple[str, str]:
    """Route a (possibly namespaced) panel id to (owner_session, raw_id).

    Combined-view ids are s<owner8>_<raw>; the owner must be this session
    or part of its continuation chain. Raw ids belong to THIS session.
    """
    chain = chain_for(session)
    parsed = _parse_ns_id(ns_or_raw_id)
    if parsed:
        prefix, raw = parsed
        owner = _owner_session_of(prefix, [*chain, session])
        if owner is None:
            raise HTTPException(404, f"unknown panel {ns_or_raw_id} "
                                      f"(not in this strip's sequence)")
        return owner, raw
    return session, ns_or_raw_id


def _invalidate_pipeline(owner: str, changed: str = "panels_edit.json") -> None:
    """Step-by-Step ledger: mark dependent steps stale after a user edit.

    Best-effort only — a missing/locked pipeline_state.json must never
    block a Panel Review edit.
    """
    try:
        from . import checkpoint as _cp
        _cp.apply_edit_invalidation(owner, changed)
    except Exception:
        pass


def set_order(session: str, ids: list[str]) -> dict:
    """Persist a full active-panel ordering.

    ids must cover every ACTIVE panel exactly once (deleted panels are
    excluded from ordering). Combined-view ids are namespaced; each is
    routed to its owning strip, whose relative order is rewritten.
    """
    by_strip: dict[str, list[str]] = {}
    for i in ids:
        owner, raw = _resolve_panel(session, i)
        by_strip.setdefault(owner, []).append(raw)
    for owner, raw_ids in by_strip.items():
        panels = _read_panels_json(owner)
        if not panels:
            raise HTTPException(404, f"strip {owner} has no panels yet")
        edit = _read_edit(owner)
        have = set(_all_panels(owner, edit))
        deleted = set(edit.get("deleted", []))
        active = have - deleted
        got = set(raw_ids)
        if got != active or len(raw_ids) != len(active):
            raise HTTPException(
                400, "order must cover every active panel of "
                     f"strip {owner[:8]} exactly once")
        edit["order"] = list(raw_ids)
        edit["confirmed"] = False
        edit["confirmed_at"] = None
        _write_edit(owner, edit)
        _invalidate_pipeline(owner)
    return get_panels(session, combined=True)


def delete_panels(session: str, ids: list[str]) -> dict:
    """Soft-delete (recoverable). panels.json untouched. Namespaced ids
    are routed to the owning strip's edit layer."""
    for i in ids:
        owner, raw = _resolve_panel(session, i)
        have = set(_all_panels(owner, _read_edit(owner)))
        if raw not in have:
            raise HTTPException(400, f"unknown panel {i}")
        edit = _read_edit(owner)
        deleted = set(edit.get("deleted", []))
        deleted.add(raw)
        edit["deleted"] = sorted(deleted)
        edit["confirmed"] = False
        edit["confirmed_at"] = None
        _write_edit(owner, edit)
        _invalidate_pipeline(owner)
    return get_panels(session, combined=True)


def restore_panels(session: str, ids: list[str]) -> dict:
    for i in ids:
        owner, raw = _resolve_panel(session, i)
        edit = _read_edit(owner)
        if raw not in set(_all_panels(owner, edit)):
            raise HTTPException(400, f"unknown panel {i}")
        edit["deleted"] = [x for x in edit.get("deleted", []) if x != raw]
        edit["confirmed"] = False
        edit["confirmed_at"] = None
        _write_edit(owner, edit)
        _invalidate_pipeline(owner)
    return get_panels(session, combined=True)


def set_review(session: str, panel_id: str, status: str) -> dict:
    if status not in VALID_REVIEW:
        raise HTTPException(400, f"invalid review status {status!r}")
    owner, raw = _resolve_panel(session, panel_id)
    if raw not in _all_panels(owner, _read_edit(owner)):
        raise HTTPException(404, f"unknown panel {panel_id}")
    edit = _read_edit(owner)
    edit.setdefault("review", {})[raw] = status
    _write_edit(owner, edit)
    _invalidate_pipeline(owner)
    return get_panels(session, combined=True)


def confirm(session: str, *, review_all: bool = False) -> dict:
    """Lock the current panel set/order as the narration source.

    In a continuation, every strip in the chain is confirmed (a partial
    confirm would let the render silently skip unconfirmed strips' review
    state). Each strip's own active order is locked in its own edit file.
    """
    chain = chain_for(session)
    for strip in [*chain, session]:
        panels = _read_panels_json(strip)
        if not panels:
            continue          # nothing to confirm for this strip
        edit = _read_edit(strip)
        deleted = set(edit.get("deleted", []))
        active_ids = [i for i in edit.get("order", []) if i not in deleted]
        if not active_ids:
            active_ids = [p["id"] for p in sorted(panels, key=lambda p: p.get("y_start", 0))
                          if p["id"] not in deleted]
        if not active_ids:
            continue
        if review_all:
            review = dict(edit.get("review", {}))
            for pid in active_ids:
                review.setdefault(pid, "reviewed")
            edit["review"] = review
        edit["order"] = active_ids
        edit["confirmed"] = True
        edit["confirmed_at"] = time.time()
        _write_edit(strip, edit)
        _invalidate_pipeline(strip)
    return get_panels(session, combined=True)

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
    owner, raw = _resolve_panel(session, panel_id)
    by_id = _all_panels(owner, _read_edit(owner))
    src = by_id.get(raw)
    if src is None:
        raise HTTPException(404, f"unknown panel {panel_id}")
    edit = _read_edit(owner)
    new_id = _alloc_custom_id(owner, edit)
    dest_name = f"{new_id}.png"
    d = _session_dir(owner)
    _save_crop_image(owner, d / src["image_file"], d / dest_name)
    entry = _custom_entry(src, new_id, dest_name,
                          y_start=src["y_start"], y_end=src["y_end"],
                          panel_type=src.get("panel_type", "panel"),
                          confidence=src.get("confidence", 1.0),
                          panel_index=src.get("panel_index", 0))
    entry["derived_from"] = raw
    edit.setdefault("custom", []).append(entry)
    _materialize_order(owner, edit, {raw: [raw, new_id]})
    _write_edit(owner, edit)
    return get_panels(session, combined=True)


def merge_panels(session: str, ids: list[str]) -> dict:
    if len(ids) != 2:
        raise HTTPException(400, "merge takes exactly two panel ids")
    (oa, ra), (ob, rb) = (_resolve_panel(session, i) for i in ids)
    if oa != ob:
        raise HTTPException(400, "cannot merge panels from different strips")
    by_id = _all_panels(oa, _read_edit(oa))
    a, b = by_id.get(ra), by_id.get(rb)
    if a is None or b is None:
        raise HTTPException(404, "unknown panel id in merge")
    if min(a["y_start"], a["y_end"]) > min(b["y_start"], b["y_end"]):
        a, b = b, a                              # a = upper panel
    edit = _read_edit(oa)
    new_id = _alloc_custom_id(oa, edit)
    dest_name = f"{new_id}.png"
    d = _session_dir(oa)
    _save_crop_image(oa, None, d / dest_name,
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
    _materialize_order(oa, edit, {a["id"]: [new_id], b["id"]: []})
    _write_edit(oa, edit)
    return get_panels(session, combined=True)


def split_panel(session: str, panel_id: str, *, fraction: float = 0.5) -> dict:
    if not 0.15 <= fraction <= 0.85:
        raise HTTPException(400, "fraction must be within 0.15-0.85")
    owner, raw = _resolve_panel(session, panel_id)
    by_id = _all_panels(owner, _read_edit(owner))
    src = by_id.get(raw)
    if src is None:
        raise HTTPException(404, f"unknown panel {panel_id}")
    d = _session_dir(owner)
    from PIL import Image
    with Image.open(d / src["image_file"]) as im:
        w, h = im.size
        cut = max(1, min(h - 1, round(h * fraction)))
    edit = _read_edit(owner)
    id1 = _alloc_custom_id(owner, edit)
    id2 = f"u{(int(id1[1:]) + 1):03d}"
    if id2 in by_id:
        raise HTTPException(409, "could not allocate two adjacent custom ids")
    name1, name2 = f"{id1}.png", f"{id2}.png"
    _save_crop_image(owner, d / src["image_file"], d / name1, box=(0, 0, w, cut))
    _save_crop_image(owner, d / src["image_file"], d / name2, box=(0, cut, w, h))
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
    edit["deleted"] = sorted(set(edit.get("deleted", [])) | {raw})
    _materialize_order(owner, edit, {raw: [id1, id2]})
    _write_edit(owner, edit)
    return get_panels(session, combined=True)

