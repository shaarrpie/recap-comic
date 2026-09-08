# webapp/narration_api.py
"""Narration Studio backend.

Layering (nothing here rewrites the AI baseline):

    panels.json / panels_confirmed.json   AI narration (per-panel text)
    panels_edit.json                      Panel Review state (panel_api)
    narration_edit.json                   Narration Studio overrides:
        {version, style, overrides:{panel_id:{text, original, at}}}

Overrides always win over the AI text, are surfaced with their original
beside them (restore = delete the override), and invalidate only the
affected panel's TTS clip (`audio/<panel_id>.mp3`), never the chapter.

Per-panel regeneration reuses adapters.narrate_gemini.generate with a
single-panel OcrArtifact reconstructed from the panel's stored dialogue,
so one bad sentence never reruns the whole chapter.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from fastapi import HTTPException

from .panel_api import (_all_panels, _read_edit, _session_dir,
                        _write_edit, get_panels)

VALID_STYLES = ("recap", "literal")   # the only styles narrator.make_script_from_cut implements

_regenerate_lock = threading.Lock()   # guards the temporary env-var swap


def _narr_edit_path(session: str) -> Path:
    return _session_dir(session) / "narration_edit.json"


def _read_narr_edit(session: str) -> dict:
    p = _narr_edit_path(session)
    if p.is_file():
        try:
            e = json.loads(p.read_text("utf-8"))
            e.setdefault("style", "recap")
            e.setdefault("overrides", {})
            return e
        except Exception:
            pass
    return {"version": 1, "style": "recap", "overrides": {}}


def _write_narr_edit(session: str, edit: dict) -> None:
    p = _narr_edit_path(session)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(edit, indent=2), "utf-8")
    tmp.replace(p)


def _panels_source(session: str) -> Path:
    conf = _session_dir(session) / "panels_confirmed.json"
    return conf if conf.is_file() else _session_dir(session) / "panels.json"


def load_overrides(session: str) -> dict:
    """panel_id -> override text (pipeline-facing helper)."""
    return {pid: o["text"]
            for pid, o in _read_narr_edit(session).get("overrides", {}).items()
            if (o.get("text") or "").strip()}


def apply_overrides_to_cut(session: str, panels: list) -> int:
    """Apply narration overrides onto CutArtifact-style panels (objects with
    a `narration` attr). Returns how many panels were changed."""
    ov = load_overrides(session)
    if not ov:
        return 0
    n = 0
    for p in panels:
        pid = getattr(p, "id", None)
        if pid in ov:
            p.narration = ov[pid]
            n += 1
    return n


def apply_overrides_to_dicts(session: str, panels: list[dict]) -> int:
    """Same, for job.panels-style dict panels."""
    ov = load_overrides(session)
    if not ov:
        return 0
    n = 0
    for p in panels:
        if p.get("id") in ov:
            p["narration"] = ov[p["id"]]
            n += 1
    return n


def get_narration(session: str) -> dict:
    """Per-panel narration for the Narration Studio, overrides applied."""
    panels = get_panels(session)          # 404s when no segmentation yet
    edit = _read_narr_edit(session)
    style = edit.get("style", "recap")
    overrides = edit.get("overrides", {})
    out = []
    for p in panels["panels"]:
        if p.get("deleted"):
            continue
        ov = overrides.get(p["id"])
        out.append({
            "id": p["id"],
            "panel_index": p["display_order"],
            "image_file": p.get("image_file", ""),
            "confidence": p.get("confidence"),
            "review": p.get("review_status") or "ai_detected",
            "narration": (ov["text"] if ov else p.get("narration", "")) or "",
            "ai_text": p.get("narration", "") or "",
            "edited": bool(ov),
            "dialogue": p.get("dialogue", "") or "",
        })
    return {
        "session": session,
        "confirmed": panels["confirmed"],
        "style": style,
        "total": len(out),
        "edited_count": sum(1 for p in out if p["edited"]),
        "panels": out,
    }


def _invalidate_tts(session: str, panel_id: str) -> None:
    """Drop only this panel's TTS clip so the next run re-synthesizes it."""
    mp3 = _session_dir(session) / "audio" / f"{panel_id}.mp3"
    try:
        mp3.unlink(missing_ok=True)
    except OSError:
        pass


def set_text(session: str, panel_id: str, text: str) -> dict:
    if panel_id not in _all_panels(session, _read_edit(session)):
        raise HTTPException(404, f"unknown panel {panel_id}")
    edit = _read_narr_edit(session)
    if not (text or "").strip():
        edit.get("overrides", {}).pop(panel_id, None)
    else:
        # capture the AI text once, so "restore" always has the real original
        prev = edit.get("overrides", {}).get(panel_id)
        if prev and prev.get("original") is not None:
            original = prev["original"]
        else:
            original = None
            src = _panels_source(session)
            try:
                data = json.loads(src.read_text("utf-8"))
                original = next((p.get("narration", "")
                                 for p in data.get("panels", [])
                                 if p.get("id") == panel_id), "")
            except Exception:
                original = ""
        edit.setdefault("overrides", {})[panel_id] = {
            "text": text, "original": original or "", "at": time.time()}
    _write_narr_edit(session, edit)
    review = _read_edit(session)
    review.setdefault("review", {})[panel_id] = "edited"
    _write_edit(session, review)
    _invalidate_tts(session, panel_id)
    return get_narration(session)


def reset_text(session: str, panel_id: str) -> dict:
    """Restore the AI's original narration for one panel."""
    edit = _read_narr_edit(session)
    edit.get("overrides", {}).pop(panel_id, None)
    _write_narr_edit(session, edit)
    _invalidate_tts(session, panel_id)
    return get_narration(session)


def set_style(session: str, style: str) -> dict:
    if style not in VALID_STYLES:
        raise HTTPException(400, f"style must be one of {VALID_STYLES}")
    edit = _read_narr_edit(session)
    edit["style"] = style
    _write_narr_edit(session, edit)
    return get_narration(session)


def regenerate(session: str, panel_id: str, *, api_key: str = "",
               model: str = "") -> dict:
    """Regenerate ONE panel's narration with Gemini; stores it as an override.

    Fails loudly (no fake text) when no key is configured.
    """
    if panel_id not in _all_panels(session, _read_edit(session)):
        raise HTTPException(404, f"unknown panel {panel_id}")
    state = get_narration(session)
    panel = next((p for p in state["panels"] if p["id"] == panel_id), None)
    if panel is None:
        raise HTTPException(404, "panel is deleted; restore it first")

    from adapters._gemini_keys import from_env
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise HTTPException(400,
            "no Gemini key: set GEMINI_API_KEY in .env or pass api_key")

    from adapters.schemas import BBox, Meta, OcrArtifact, OcrRegion
    # Reconstruct a minimal OCR artifact from the panel's stored dialogue so
    # the model sees the same source lines the chapter run saw.
    lines = [ln.strip() for ln in panel["dialogue"].splitlines() if ln.strip()]
    regions = [OcrRegion(id=f"{panel_id}-{i}", panel_id=panel_id, page=1,
                         bbox=BBox(x=0, y=0, w=0, h=0), text=ln,
                         confidence=1.0, kind="dialogue")
               for i, ln in enumerate(lines)]
    ocr = OcrArtifact(meta=Meta(schema_version=1, generator="narration_api",
                                config_hash="", input_hashes={}),
                      backend="webapp", regions=regions)
    request = (f"Write the recap narration for panel {panel_id} only. "
               f"Match the chapter's style and keep it self-contained.")

    with _regenerate_lock:
        old = os.environ.get("GEMINI_API_KEY")
        try:
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key
            from_env()  # raises when still unconfigured
            import adapters.narrate_gemini as ng
            plan = ng.generate(request, ocr, [panel_id],
                               model=model or "gemini-2.0-flash")
        finally:
            if api_key:
                if old is None:
                    os.environ.pop("GEMINI_API_KEY", None)
                else:
                    os.environ["GEMINI_API_KEY"] = old

    entries = [e for e in plan.entries if (e.text or "").strip()]
    if not entries:
        raise HTTPException(502, "regeneration returned no narration text")
    text = "\n".join(e.text for e in entries)
    return set_text(session, panel_id, text)   # stores override + invalidates TTS


