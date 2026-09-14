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

import contextlib
import json
import os
import re
import threading
import time
from pathlib import Path

from fastapi import HTTPException

from .jobs import JobStatus, store
from .panel_api import _all_panels, _read_edit, _session_dir, _write_edit, get_panels

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


_NS_PREFIX_RE = re.compile(r"^s[0-9a-f]{8}_")


def _override_for(ov: dict, pid: str) -> str | None:
    """Override text for a panel id, matching BOTH the raw id and the
    continuation-namespaced form (merge_continuation rewrites ids to
    `s<session8>_<raw>`; overrides saved against raw ids before a merge
    must still apply to the namespaced panel)."""
    if pid in ov:
        return ov[pid]
    raw = _NS_PREFIX_RE.sub("", pid, count=1)
    if raw != pid and raw in ov:
        return ov[raw]
    return None


def apply_overrides_to_cut(session: str, panels: list) -> int:
    """Apply narration overrides onto CutArtifact-style panels (objects with
    a `narration` attr). Returns how many panels were changed."""
    ov = load_overrides(session)
    if not ov:
        return 0
    n = 0
    for p in panels:
        pid = getattr(p, "id", None)
        text = _override_for(ov, pid) if pid else None
        if text is not None:
            p.narration = text
            n += 1
    return n


def apply_overrides_to_dicts(session: str, panels: list[dict]) -> int:
    """Same, for job.panels-style dict panels."""
    ov = load_overrides(session)
    if not ov:
        return 0
    n = 0
    for p in panels:
        text = _override_for(ov, p.get("id"))
        if text is not None:
            p["narration"] = text
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
    with contextlib.suppress(OSError):
        mp3.unlink(missing_ok=True)


def _invalidate_chapter_script(session: str) -> None:
    """Drop script.json so a user narration edit is never shadowed by a
    stale whole-chapter script (Phase 2.5).

    The render's build_narration prefers script.json lines; after an edit
    the per-panel captions (which carry the override via
    panels_confirmed.json) must be the text source. _build_script only
    regenerates script.json when NO overrides exist, so this file stays
    absent until the user restores all AI texts."""
    with contextlib.suppress(OSError):
        (_session_dir(session) / "script.json").unlink(missing_ok=True)


def _invalidate_pipeline_steps(session: str,
                               changed: str = "narration_edit.json") -> None:
    """Step-by-Step ledger: narration edits invalidate build_script+."""
    try:
        from . import checkpoint as _cp
        _cp.apply_edit_invalidation(session, changed)
    except Exception:
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
    _invalidate_chapter_script(session)
    _invalidate_pipeline_steps(session)
    return get_narration(session)


def reset_text(session: str, panel_id: str) -> dict:
    """Restore the AI's original narration for one panel."""
    edit = _read_narr_edit(session)
    edit.get("overrides", {}).pop(panel_id, None)
    _write_narr_edit(session, edit)
    _invalidate_tts(session, panel_id)
    _invalidate_chapter_script(session)
    _invalidate_pipeline_steps(session)
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
    """Regenerate ONE panel's narration; stores it as an override.

    Default path uses the central Qwen3.5-397B-A17B -> Mistral Medium 3.5
    fallback (OpenAI-compatible Xkiro endpoint). A `model` starting with
    "gemini" preserves the legacy Gemini adapter explicitly. Fails loudly
    (no fake text) when no key is configured or both models fail.
    """
    if panel_id not in _all_panels(session, _read_edit(session)):
        raise HTTPException(404, f"unknown panel {panel_id}")
    state = get_narration(session)
    panel = next((p for p in state["panels"] if p["id"] == panel_id), None)
    if panel is None:
        raise HTTPException(404, "panel is deleted; restore it first")

    if (model or "").lower().startswith("gemini"):
        return _regenerate_gemini(session, panel_id, panel, api_key, model)

    from adapters import ai_models as _ai
    key = api_key or _ai.api_key_from_env() or ""
    if not key:
        raise HTTPException(400,
            "no AI key: set XKIRO_API_KEY in .env or pass api_key")

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
    dump = "\n".join(
        f"{r.panel_id or '?'} [{r.kind}] conf={r.confidence}: {r.text}"
        for r in ocr.regions)
    request = (f"Write the recap narration for panel {panel_id} only. "
               f"Match the chapter's style and keep it self-contained.\n"
               f"OCR lines:\n{dump}\n"
               'Return JSON: {"entries": [{"panel_id": "...", "speaker": null, '
               '"text": "..."}]} with exactly 1 entry.')
    import json as _json

    from adapters.narrate_gemini import NarrationArtifact

    def _parse(text: str):
        data = _json.loads(text)
        narration = NarrationArtifact.model_validate(
            {**data, "mode": "narrator", "meta": ocr.meta})
        if [e.panel_id for e in narration.entries] != [panel_id]:
            raise ValueError("panel ids/order do not match panels.json")
        return narration

    with _regenerate_lock:
        try:
            outcome = _ai.generate_text_with_fallback(
                "You are the narrator for a recap video of a manhwa chapter. "
                "Use ONLY the provided OCR text. Output ONLY JSON.",
                request, operation=f"narration-regenerate:{panel_id}",
                api_key=key,
                primary_model=model or _ai.PRIMARY_MODEL)
            plan = _parse(outcome.result)
        except _ai.AIFallbackError as exc:
            raise HTTPException(
                502, f"narration failed: primary ({exc.primary_error}); "
                     f"fallback ({exc.fallback_error})") from exc
        except Exception as exc:
            raise HTTPException(502, f"regeneration failed: {exc}") from exc

    entries = [e for e in plan.entries if (e.text or "").strip()]
    if not entries:
        raise HTTPException(502, "regeneration returned no narration text")
    text = "\n".join(e.text for e in entries)
    return set_text(session, panel_id, text)   # stores override + invalidates TTS


def _run_generate_all(job_id: str, session: str,
                      api_key: str, model: str) -> None:
    """Background worker: AI narration for every cropped panel."""
    job = store.get(job_id)
    if job is None:
        return
    job.status = JobStatus.RUNNING
    job.started_at = time.time()
    job.touch()
    job.log("INFO", "AI narration started (cropped panels only; "
                     "geometry locked)", "ai_narration")
    try:
        import adapters.ai_narration as ain
        summary = ain.narrate_cropped_panels(
            _session_dir(session), api_key=api_key or None,
            model=model or "")
        job.log("INFO",
                f"AI narration complete panels={summary['panels']} "
                f"narrated={summary['narrated']} cached={summary['cached']} "
                f"failed={len(summary['failed'])}", "ai_narration")
        for f in summary["failed"]:
            job.log("WARNING", f"panel kept old text: {f}", "ai_narration")
        job.status = JobStatus.COMPLETED
        job.progress = 100
        job.finished_at = time.time()
        job.touch()
    except Exception as exc:
        import traceback
        job.fail(f"AI narration failed: {type(exc).__name__}: {exc}",
                 traceback.format_exc())


def generate_all(session: str, *, api_key: str = "",
                 model: str = "") -> dict:
    """START button: narrate every cropped panel with Qwen -> Mistral.

    Cropping must already exist (panels.json + panel PNGs), produced with
    or without AI — typically the deterministic blank-row cut. Only words
    are written; panel geometry is asserted unchanged. Runs in a background
    thread; poll /api/jobs/<job_id> for completion, then reload the
    narration view.
    """
    if not _panels_source(session).is_file():
        raise HTTPException(404,
            "no panels.json; crop the strip first (deterministic cut needs "
            "no AI key)")
    from adapters import ai_models as _ai
    key = api_key or _ai.api_key_from_env() or ""
    if not key:
        raise HTTPException(400,
            "no AI key: set XKIRO_API_KEY in .env or pass api_key")
    job = store.create("ai_narration", {"session": session})
    threading.Thread(target=_run_generate_all,
                     args=(job.id, session, api_key, model),
                     daemon=True).start()
    return {"job_id": job.id, "status": job.status.value}


def _regenerate_gemini(session: str, panel_id: str, panel: dict,
                       api_key: str, model: str) -> dict:
    """Legacy explicit-Gemini path (only when model starts with 'gemini')."""
    from adapters._gemini_keys import from_env
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise HTTPException(400,
            "no Gemini key: set GEMINI_API_KEY in .env or pass api_key")

    from adapters.schemas import BBox, Meta, OcrArtifact, OcrRegion
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
        from_env()  # raises when still unconfigured (env key required)
        import adapters.narrate_gemini as ng
        plan = ng.generate(request, ocr, [panel_id],
                           model=model or "gemini-2.0-flash",
                           api_key=api_key or None)

    entries = [e for e in plan.entries if (e.text or "").strip()]
    if not entries:
        raise HTTPException(502, "regeneration returned no narration text")
    text = "\n".join(e.text for e in entries)
    return set_text(session, panel_id, text)   # stores override + invalidates TTS


