# webapp/editor_api.py
"""Editor-specific backend API for recap-comic.

Reuses the existing automation pipeline assets.  Does not regenerate narration
or TTS unless explicitly requested.  All file paths are scoped to the session
directory under OUTPUT_DIR.
"""
from __future__ import annotations

import copy
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from webapp.jobs import Job, JobStatus, store

from adapters.editor import Editor, EditorProject
from recap_video import VideoConfig, render_edited_project

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "webapp_output"
log = logging.getLogger(__name__)

_SESSION_RE = re.compile(r"^[0-9a-f]{12}$")


def _validate_session(session: str) -> None:
    if not _SESSION_RE.match(session):
        raise HTTPException(400, "invalid session id")


def _session_dir(session: str) -> Path:
    _validate_session(session)
    d = (OUTPUT_DIR / session).resolve()
    base = OUTPUT_DIR.resolve()
    if base not in d.parents and d != base:
        raise HTTPException(400, "invalid session path")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _project_path(session: str) -> Path:
    return _session_dir(session) / "editor.json"


def _get_project(session: str) -> EditorProject | None:
    p = _project_path(session)
    if not p.is_file():
        return None
    try:
        return EditorProject.model_validate_json(p.read_text("utf-8"))
    except Exception as exc:
        log.warning("editor load failed session=%s err=%s", session, exc)
        return None


def load_project(session: str) -> dict[str, Any]:
    proj = _get_project(session)
    if proj is None:
        raise FileNotFoundError("editor project not found; generate a video first")
    return proj.model_dump()


def save_project(session: str, data: dict[str, Any]) -> dict[str, Any]:
    d = _session_dir(session)
    proj = EditorProject.model_validate(data)
    Editor(proj).save(d / "editor.json")
    return {"ok": True, "needs_render": proj.needs_render}


def create_project_from_generation(session: str) -> dict[str, Any]:
    """Create an initial editor.json from existing automation outputs."""
    d = _session_dir(session)
    panels_json = d / "panels.json"
    tl_json = d / "timeline.json"
    srt_path = d / "recap.srt"
    narration_json = d / "narration.json"
    audio_json = d / "audio.json"

    if not panels_json.is_file():
        raise FileNotFoundError("missing panels.json; run generation first")

    from adapters.schemas import TimelineArtifact
    from guided_cutter import CutArtifact
    from recap_video import VideoConfig

    cfg = VideoConfig()
    artifact = CutArtifact.model_validate_json(panels_json.read_text("utf-8"))

    # Prefer existing timeline.json; rebuild only if missing/invalid.
    original_timeline = None
    if tl_json.is_file():
        try:
            tl = TimelineArtifact.model_validate_json(tl_json.read_text("utf-8"))
            original_timeline = [e.model_dump() for e in tl.entries]
        except Exception:
            pass

    if original_timeline is None:
        from adapters.editor import timeline_from_cut
        original_timeline = timeline_from_cut(artifact, d, cfg)

    # enrich original timeline with automated start/end for later comparison
    for e in original_timeline:
        e["automated_duration"] = e["duration_seconds"]
        e["automated_start_seconds"] = e["start_seconds"]
        e["automated_end_seconds"] = e["start_seconds"] + e["duration_seconds"]
        e["automated_pan"] = e.get("pan", {})

    edited = copy.deepcopy(original_timeline)

    narration = None
    audio = None
    if narration_json.is_file():
        try:
            from adapters.schemas import NarrationArtifact
            narration = NarrationArtifact.model_validate_json(narration_json.read_text("utf-8"))
        except Exception:
            pass
    if audio_json.is_file():
        try:
            from adapters.schemas import AudioArtifact
            audio = AudioArtifact.model_validate_json(audio_json.read_text("utf-8"))
        except Exception:
            pass

    captions = []
    if narration and audio:
        from adapters.editor import captions_from_timeline
        captions = captions_from_timeline(original_timeline, narration, audio)
    elif srt_path.is_file():
        captions = _parse_srt_captions(srt_path)

    from adapters.editor import transitions_from_timeline, effects_from_timeline
    transitions = transitions_from_timeline(original_timeline)
    effects = effects_from_timeline(original_timeline)

    proj = EditorProject(
        session=session,
        original_timeline=original_timeline,
        edited_timeline=edited,
        captions=captions,
        transitions=transitions,
        effects=effects,
    )
    Editor(proj).save(d / "editor.json")
    return proj.model_dump()


def _parse_srt_captions(path: Path) -> list[dict[str, Any]]:
    import re
    text = path.read_text("utf-8")
    blocks = re.split(r"\n\s*\n", text.strip())
    captions = []
    cid = 0
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        m = re.match(r"(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)", lines[1])
        if not m:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
        cid += 1
        captions.append({
            "id": f"cap_{cid:03d}",
            "panel_id": "",
            "text": " ".join(lines[2:]),
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "automated_text": " ".join(lines[2:]),
            "automated_start_seconds": round(start, 3),
            "automated_end_seconds": round(end, 3),
        })
    return captions


def reorder_panels(session: str, new_order: list[str]) -> dict[str, Any]:
    proj = _get_project(session)
    if proj is None:
        raise FileNotFoundError("editor project not found")
    Editor(proj).reorder_panels(new_order)
    Editor(proj).save(_project_path(session))
    return proj.model_dump()


def set_duration(session: str, panel_id: str, duration: float) -> dict[str, Any]:
    proj = _get_project(session)
    if proj is None:
        raise FileNotFoundError("editor project not found")
    Editor(proj).set_duration(panel_id, duration)
    Editor(proj).save(_project_path(session))
    return proj.model_dump()


def set_effect(session: str, panel_id: str, kind: str, duration: float) -> dict[str, Any]:
    proj = _get_project(session)
    if proj is None:
        raise FileNotFoundError("editor project not found")
    Editor(proj).set_effect(panel_id, kind, duration)
    Editor(proj).save(_project_path(session))
    return proj.model_dump()


def update_caption(session: str, caption_id: str, **kwargs) -> dict[str, Any]:
    proj = _get_project(session)
    if proj is None:
        raise FileNotFoundError("editor project not found")
    Editor(proj).update_caption(caption_id, **kwargs)
    Editor(proj).save(_project_path(session))
    return proj.model_dump()


def set_transition(session: str, from_id: str, to_id: str,
                   type: str, duration: float) -> dict[str, Any]:
    proj = _get_project(session)
    if proj is None:
        raise FileNotFoundError("editor project not found")
    Editor(proj).set_transition(from_id, to_id, type, duration)
    Editor(proj).save(_project_path(session))
    return proj.model_dump()


def remove_panel(session: str, panel_id: str) -> dict[str, Any]:
    proj = _get_project(session)
    if proj is None:
        raise FileNotFoundError("editor project not found")
    Editor(proj).remove_panel(panel_id)
    Editor(proj).save(_project_path(session))
    return proj.model_dump()


def add_panel(session: str, panel_id: str, after: str | None = None) -> dict[str, Any]:
    proj = _get_project(session)
    if proj is None:
        raise FileNotFoundError("editor project not found")
    Editor(proj).add_panel(panel_id, after)
    Editor(proj).save(_project_path(session))
    return proj.model_dump()


def undo(session: str) -> dict[str, Any]:
    proj = _get_project(session)
    if proj is None:
        raise FileNotFoundError("editor project not found")
    Editor(proj).undo()
    Editor(proj).save(_project_path(session))
    return proj.model_dump()


def redo(session: str) -> dict[str, Any]:
    proj = _get_project(session)
    if proj is None:
        raise FileNotFoundError("editor project not found")
    Editor(proj).redo()
    Editor(proj).save(_project_path(session))
    return proj.model_dump()


def reset_to_automated(session: str) -> dict[str, Any]:
    proj = _get_project(session)
    if proj is None:
        raise FileNotFoundError("editor project not found")
    Editor(proj).reset_to_automated()
    Editor(proj).save(_project_path(session))
    return proj.model_dump()


def start_render(session: str, cfg: dict[str, Any]) -> dict[str, Any]:
    d = _session_dir(session)
    out_mp4 = d / "recap_edited.mp4"
    editor_path = d / "editor.json"
    if not editor_path.is_file():
        raise FileNotFoundError("editor project not found")

    job = store.create("render_edited", {"session": session, "cfg": cfg})
    log.info("job=%s created kind=render_edited session=%s", job.id, session)

    def _run():
        try:
            vcfg = VideoConfig(**cfg)
            job.status = JobStatus.RUNNING
            job.stage = "render_edited"
            job.touch()
            job.log("INFO", "render started", "render_edited")
            result = render_edited_project(editor_path, out_mp4, vcfg)
            job.status = JobStatus.COMPLETED
            job.outputs = {k: str(v) for k, v in result.items() if isinstance(v, str)}
            job.progress = 100
            job.stage = "done"
            job.finished_at = time.time()
            job.touch()
            job.log("INFO", f"render completed out={out_mp4}", "render_edited")
        except Exception as exc:
            import traceback
            job.fail(f"render failed: {exc}", traceback.format_exc())

    import threading
    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job.id, "status": job.status.value}
