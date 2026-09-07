# webapp/main.py
"""FastAPI layer: thin. Creates jobs, returns ids, serves status.
All processing lives in webapp/pipeline.py."""
from __future__ import annotations

import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from adapters._logging import get_logger, setup_logging

from . import pipeline
from .editor_api import (
    add_panel,
    create_project_from_generation,
    get_ai_review_data,
    get_panel_confidence,
    load_project,
    redo,
    reorder_panels,
    remove_panel,
    render_edited_project,
    reset_to_automated,
    save_project,
    set_duration,
    set_effect,
    set_transition,
    start_render,
    undo,
    update_caption,
    update_narration,
)
from .jobs import store
from adapters.editor import Editor

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=False)
setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"),
              log_dir=BASE_DIR / "logs")
log = get_logger(__name__)

app = FastAPI(title="recap-comic webapp")
OUTPUT_DIR = BASE_DIR / "webapp_output"
OUTPUT_DIR.mkdir(exist_ok=True)

_last_config_state: dict | None = None


def _default_backend() -> str:
    """Return the backend to use for uploads: 'gemini' if configured, else 'none'."""
    if (os.environ.get("GEMINI_API_KEYS", "").strip()
            or os.environ.get("GEMINI_API_KEY", "").strip()):
        return "gemini"
    return "none"


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "webapp" / "static" / "app.html")


@app.get("/api/config")
async def config():
    global _last_config_state
    key_src = os.environ.get("GEMINI_API_KEYS", "").strip()
    single_key = os.environ.get("GEMINI_API_KEY", "").strip()
    configured = bool(key_src or single_key)
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    state = {"gemini_configured": configured, "model": model}
    if state != _last_config_state:
        log.info("config check gemini_configured=%s model=%s",
                 configured, model)
        _last_config_state = state
    return state


@app.get("/api/projects")
async def projects():
    """List sessions with derived status for the Library/Dashboard view."""
    import json
    result = []
    jobs_by_session: dict[str, list] = {}
    for j in store._jobs.values():
        s = j.config.get("session")
        if s:
            jobs_by_session.setdefault(s, []).append(j)
    for session_dir in sorted(OUTPUT_DIR.iterdir()):
        if not session_dir.is_dir():
            continue
        sid = session_dir.name
        panels_json = session_dir / "panels.json"
        tl_json = session_dir / "timeline.json"
        mp4 = session_dir / "recap.mp4"
        editor_json = session_dir / "editor.json"
        panel_count = 0
        if panels_json.is_file():
            try:
                data = json.loads(panels_json.read_text("utf-8"))
                panel_count = len(data.get("panels", []))
            except Exception:
                pass
        jobs = sorted(jobs_by_session.get(sid, []), key=lambda j: j.created_at)
        last_job = jobs[-1] if jobs else None
        status = last_job.status.value if last_job else "no_job"
        stage = last_job.stage if last_job else None
        result.append({
            "id": sid,
            "name": sid,
            "panels": panel_count,
            "status": status,
            "stage": stage,
            "progress": last_job.progress if last_job else 0,
            "has_video": mp4.is_file(),
            "has_editor_project": editor_json.is_file(),
            "has_timeline": tl_json.is_file(),
            "error": last_job.error if last_job else None,
            "created_at": last_job.created_at if last_job else None,
            "updated_at": last_job.updated_at if last_job else None,
        })
    return {"projects": result}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):  # noqa: B008
    allowed = {"png", "jpg", "jpeg", "webp"}
    suffix = Path(file.filename or "x.png").suffix.lower().lstrip(".")
    if suffix not in allowed:
        raise HTTPException(400, f"unsupported format: {suffix}")
    session = store.create("segment", {})
    session_dir = OUTPUT_DIR / session.id
    session_dir.mkdir(parents=True, exist_ok=True)
    strip_file = f"strip.{suffix}"
    dest = session_dir / strip_file
    max_bytes = 200 * 1024 * 1024
    written = 0
    with dest.open("wb") as fh:
        while True:
            chunk = await file.read(4 * 1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                raise HTTPException(413, "upload too large; max 200 MB")
            fh.write(chunk)
    cfg = {
        "session": session.id,
        "strip_file": strip_file,
        "backend": _default_backend(),
    }
    session.config.update(cfg)
    log.info("job=%s upload received filename=%s bytes=%d backend=%s",
             session.id, file.filename, written, cfg["backend"])
    threading.Thread(target=pipeline.run_job, args=(session.id,),
                     daemon=True).start()
    log.info("job=%s worker started kind=segment", session.id)
    return {"job_id": session.id, "status": session.status.value}


class RunRequest(BaseModel):
    session: str
    order: list[str] | None = None
    tts: str = "edge"
    voice: str = "en-US-AriaNeural"
    style: str = "recap"
    backend: str = "none"
    api_key: str = ""
    model: str = ""
    endpoint: str = ""
    cf_account_id: str = ""
    start_stage: str | None = None  # resume from a specific stage (retry)


@app.post("/api/run")
async def run(body: RunRequest):
    src = store.get(body.session)
    if src is None or not (OUTPUT_DIR / body.session
                           / src.config.get("strip_file", "")).exists():
        raise HTTPException(404, "session/strip not found; upload first")
    job = store.create("generate", {
        "session": body.session,
        "strip_file": src.config["strip_file"],
        "order": body.order,
        "tts": body.tts, "voice": body.voice, "style": body.style,
        "backend": body.backend,
        "start_stage": body.start_stage,
    })
    log.info("job=%s created kind=generate session=%s order=%s",
             job.id, body.session,
             "user" if body.order else "default")
    kwargs = {
        "api_key": body.api_key or None,
        "model": body.model or None,
        "base_url": body.endpoint or None,
        "cf_account_id": body.cf_account_id or None,
    }
    threading.Thread(target=pipeline.run_job, args=(job.id,),
                     kwargs=kwargs, daemon=True).start()
    log.info("job=%s worker started kind=generate", job.id)
    return {"job_id": job.id, "status": job.status.value}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str, logs: int = 0):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return JSONResponse(job.to_dict(include_logs=bool(logs)))


@app.post("/api/jobs/{job_id}/cancel")
async def cancel(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    job.cancel_requested = True
    job.log("INFO", "cancel requested")
    return {"job_id": job_id, "cancel_requested": True}


@app.get("/api/jobs/{job_id}/files/{name:path}")
async def job_file(job_id: str, name: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    session = job.config.get("session", job_id)
    base = (OUTPUT_DIR / session).resolve()
    target = (base / name).resolve()
    if target.parent != base and base not in target.parents:
        raise HTTPException(400, "bad path")
    if not target.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(target)


# --------------------------------------------------------------------------- #
# Editor routes
# --------------------------------------------------------------------------- #
@app.get("/editor/{session}")
async def editor_page(session: str):
    return FileResponse(BASE_DIR / "webapp" / "static" / "editor.html")


@app.get("/api/editor/{session}")
async def editor_get(session: str):
    try:
        return load_project(session)
    except FileNotFoundError:
        raise HTTPException(404, "editor project not found; generate a video first")


@app.post("/api/editor/{session}/create")
async def editor_create(session: str):
    try:
        return create_project_from_generation(session)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/editor/{session}")
async def editor_save(session: str, body: dict):
    try:
        return save_project(session, body)
    except FileNotFoundError:
        raise HTTPException(404, "editor project not found")


@app.post("/api/editor/{session}/reorder")
async def editor_reorder(session: str, body: dict):
    try:
        return reorder_panels(session, body.get("order", []))
    except FileNotFoundError:
        raise HTTPException(404, "editor project not found")


@app.post("/api/editor/{session}/duration")
async def editor_duration(session: str, body: dict):
    try:
        return set_duration(session, body["panel_id"], float(body["duration"]))
    except (FileNotFoundError, KeyError, ValueError, TypeError):
        raise HTTPException(400, "bad request")


@app.post("/api/editor/{session}/effect")
async def editor_effect(session: str, body: dict):
    try:
        return set_effect(session, body["panel_id"], body["kind"], float(body.get("duration", 0.0)))
    except (FileNotFoundError, KeyError, ValueError, TypeError):
        raise HTTPException(400, "bad request")


@app.post("/api/editor/{session}/caption")
async def editor_caption(session: str, body: dict):
    try:
        cid = body.pop("id")
        allowed = {"text", "start_seconds", "end_seconds"}
        filtered = {k: v for k, v in body.items() if k in allowed}
        return update_caption(session, cid, **filtered)
    except (FileNotFoundError, KeyError, TypeError):
        raise HTTPException(400, "bad request")


@app.post("/api/editor/{session}/narration")
async def editor_narration(session: str, body: dict):
    try:
        panel_id = body["panel_id"]
        text = body.get("text", "")
        return update_narration(session, panel_id, text)
    except (FileNotFoundError, KeyError, TypeError):
        raise HTTPException(400, "bad request")


@app.post("/api/editor/{session}/transition")
async def editor_transition(session: str, body: dict):
    try:
        return set_transition(session, body["from_panel_id"], body["to_panel_id"],
                              body["type"], float(body["duration"]))
    except (FileNotFoundError, KeyError, ValueError, TypeError):
        raise HTTPException(400, "bad request")


@app.post("/api/editor/{session}/panel/remove")
async def editor_remove(session: str, body: dict):
    try:
        return remove_panel(session, body["panel_id"])
    except (FileNotFoundError, KeyError):
        raise HTTPException(400, "bad request")


@app.post("/api/editor/{session}/panel/add")
async def editor_add(session: str, body: dict):
    try:
        return add_panel(session, body["panel_id"], body.get("after"))
    except (FileNotFoundError, KeyError):
        raise HTTPException(400, "bad request")


@app.post("/api/editor/{session}/undo")
async def editor_undo(session: str):
    try:
        return undo(session)
    except FileNotFoundError:
        raise HTTPException(404, "editor project not found")


@app.post("/api/editor/{session}/redo")
async def editor_redo(session: str):
    try:
        return redo(session)
    except FileNotFoundError:
        raise HTTPException(404, "editor project not found")


@app.post("/api/editor/{session}/reset")
async def editor_reset(session: str):
    try:
        return reset_to_automated(session)
    except FileNotFoundError:
        raise HTTPException(404, "editor project not found")


@app.post("/api/editor/{session}/render")
async def editor_render(session: str, body: dict | None = None):
    body = body or {}
    try:
        return start_render(session, body.get("cfg", {}))
    except FileNotFoundError:
        raise HTTPException(404, "editor project not found")


@app.get("/api/editor/{session}/ai-review")
async def editor_ai_review(session: str):
    try:
        return editor_api.get_ai_review_data(session)
    except FileNotFoundError:
        raise HTTPException(404, "panels not found")


@app.get("/api/editor/{session}/panel-confidence")
async def editor_panel_confidence(session: str):
    try:
        return editor_api.get_panel_confidence(session)
    except FileNotFoundError:
        raise HTTPException(404, "panels not found")


@app.post("/api/editor/{session}/narration/regenerate")
async def editor_narration_regenerate(session: str, body: dict):
    panel_id = body.get("panel_id")
    if not panel_id:
        raise HTTPException(400, "panel_id required")
    d = OUTPUT_DIR / session
    panels_json = d / "panels.json"
    narration_json = d / "narration.json"
    if not panels_json.is_file() or not narration_json.is_file():
        raise HTTPException(404, "missing panels.json or narration.json")
    try:
        from guided_cutter import CutArtifact
        from adapters.schemas import NarrationArtifact
        artifact = CutArtifact.model_validate_json(panels_json.read_text("utf-8"))
        narration = NarrationArtifact.model_validate_json(narration_json.read_text("utf-8"))
    except Exception as exc:
        raise HTTPException(500, f"cannot read project files: {exc}")
    panel = next((p for p in artifact.panels if p.id == panel_id), None)
    if not panel:
        raise HTTPException(404, f"panel {panel_id} not found")
    entry = next((n for n in narration.entries if n.id == panel_id), None)
    if not entry:
        raise HTTPException(404, f"narration for {panel_id} not found")
    cut_art = CutArtifact(
        source=artifact.source, width=artifact.width, height=artifact.height,
        plan_hash=artifact.plan_hash, config=artifact.config, panels=[panel])
    n_art = NarrationArtifact(
        meta=narration.meta, mode=narration.mode, entries=[entry])
    try:
        import narrator
        new_entry = narrator.make_script_from_cut(cut_art, style="recap").entries[0]
        entry.text = new_entry.text
        narration_json.write_text(narration.model_dump_json(indent=2) + "\n", "utf-8")
        editor_path = d / "editor.json"
        if editor_path.is_file():
            proj = editor_api._get_project(session)
            if proj:
                for e in proj.edited_timeline:
                    if e.get("panel_id") == panel_id:
                        e["narration"] = new_entry.text
                        e["needs_render"] = True
                Editor(proj).save(editor_path)
        return {"ok": True, "panel_id": panel_id, "text": new_entry.text}
    except Exception as exc:
        raise HTTPException(500, f"narration regeneration failed: {exc}")
