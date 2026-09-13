# webapp/main.py
"""FastAPI layer: thin. Creates jobs, returns ids, serves status.
All processing lives in webapp/pipeline.py."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from adapters._logging import get_logger, setup_logging
from adapters.editor import Editor

from . import editor_api, manual_crop_api, pipeline
from .editor_api import (
    add_panel,
    create_project_from_generation,
    load_project,
    redo,
    remove_panel,
    reorder_panels,
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

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=False)
setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"),
              log_dir=BASE_DIR / "logs")
log = get_logger(__name__)


def _silence_win_connection_reset(loop: object) -> None:
    """Drop the noisy Proactor traceback on Windows (uvicorn + browsers).

    When a browser aborts a connection (hard refresh, cancelled image
    loads — this SPA polls every second and fetches many panel PNGs),
    the Proactor transport's connection_lost callback calls
    sock.shutdown() on an already-reset socket. The resulting
    ConnectionResetError (WinError 10054) / ConnectionAbortedError
    (10053) is benign but asyncio's default handler prints a full
    traceback per occurrence. Swallow exactly those; log everything
    else at debug so real bugs stay visible.
    """
    def _handler(loop, context):  # noqa: ANN001
        exc = context.get("exception")
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
            return      # client dropped the connection; nothing to do
        msg = str(context.get("message") or exc or "unknown asyncio error")
        log.debug("asyncio: %s (context=%s)", msg, context)

    loop.set_exception_handler(_handler)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    import asyncio
    _silence_win_connection_reset(asyncio.get_running_loop())
    log.info("webapp startup: asyncio reset-noise filter active")
    yield


app = FastAPI(title="recap-comic webapp", lifespan=lifespan)
OUTPUT_DIR = BASE_DIR / "webapp_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Job records + log buffers survive uvicorn restarts (see webapp/jobs.py).
# Without this, every restart turns all job/log lookups into 404s.
store.configure_persistence(BASE_DIR / ".cache" / "jobs")

_last_config_state: dict | None = None


def _default_backend() -> str:
    """Return the backend for uploads: 'xkiro' (Qwen+Mistral) if any AI
    key is configured, else 'none' (offline deterministic CV only)."""
    if (os.environ.get("XKIRO_API_KEY", "").strip()
            or os.environ.get("XKIRO_API_KEYS", "").strip()
            or os.environ.get("GEMINI_API_KEYS", "").strip()
            or os.environ.get("GEMINI_API_KEY", "").strip()):
        return "xkiro"
    return "none"


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "webapp" / "static" / "index.html")


# --------------------------------------------------------------------------- #
# Settings — server-side persistence for the Automation config (Settings view)
# --------------------------------------------------------------------------- #
SETTINGS_FIELDS = ("backend", "model", "api_key", "endpoint", "cf_account_id",
                   "tts", "voice", "style", "mode")
_SETTINGS_DEFAULTS = {
    "backend": "gemini", "model": "", "api_key": "", "endpoint": "",
    "cf_account_id": "", "tts": "edge", "voice": "en-US-AriaNeural",
    "style": "recap", "mode": "automation",
}


def _settings_path() -> Path:
    return OUTPUT_DIR / "settings.json"


def _read_settings() -> dict:
    """Saved settings, or defaults when never saved / unreadable."""
    try:
        data = json.loads(_settings_path().read_text("utf-8"))
    except (OSError, ValueError):
        return dict(_SETTINGS_DEFAULTS)
    if not isinstance(data, dict):
        return dict(_SETTINGS_DEFAULTS)
    return {k: (data[k] if isinstance(data.get(k), str) else v)
            for k, v in _SETTINGS_DEFAULTS.items()}


def _write_settings(data: dict) -> None:
    """Atomic tmp+replace (same pattern as manual_crop_api._write_manual)."""
    p = _settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), "utf-8")
    tmp.replace(p)


def _settings_api_key() -> str:
    """Saved api_key from webapp_output/settings.json, or "".

    Pipeline/narration fallback when no per-request key is passed; the
    key never appears in logs or job serialization.
    """
    key = _read_settings().get("api_key", "")
    return key.strip()


class SettingsBody(BaseModel):
    backend: str = "gemini"
    model: str = ""
    api_key: str = ""
    endpoint: str = ""
    cf_account_id: str = ""
    tts: str = "edge"
    voice: str = "en-US-AriaNeural"
    style: str = "recap"
    mode: str = "automation"


@app.get("/api/settings")
async def settings_get():
    """Current settings for the Settings view. NEVER returns the raw
    api_key — only api_key_set, so status endpoints cannot leak it."""
    s = _read_settings()
    return {**{k: s[k] for k in SETTINGS_FIELDS if k != "api_key"},
            "api_key_set": bool(s.get("api_key", "").strip())}


@app.get("/api/settings/key")
async def settings_key():
    """Raw api_key, ONLY for pre-filling the Settings view input.

    Local single-user tool: the key is the requesting user's own. Not
    used by any status/log surface.
    """
    return {"api_key": _settings_api_key()}


@app.post("/api/settings")
async def settings_post(body: SettingsBody):
    """Persist settings server-side to webapp_output/settings.json.

    An empty api_key KEEPS the previously saved key: unrelated saves
    (e.g. voice-only) must not silently wipe credentials.
    """
    cur = _read_settings()
    data = {k: getattr(body, k) for k in SETTINGS_FIELDS}
    if not (data.get("api_key") or "").strip():
        data["api_key"] = cur.get("api_key", "")
    _write_settings(data)
    log.info("settings saved backend=%s tts=%s style=%s key=%s",
             data["backend"], data["tts"], data["style"],
             "set" if data["api_key"].strip() else "missing")
    return {**{k: data[k] for k in SETTINGS_FIELDS if k != "api_key"},
            "api_key_set": bool(data["api_key"].strip())}


@app.get("/api/config")
async def config():
    global _last_config_state
    key_src = os.environ.get("GEMINI_API_KEYS", "").strip()
    single_key = os.environ.get("GEMINI_API_KEY", "").strip()
    xkiro_key = (os.environ.get("XKIRO_API_KEY", "").strip()
                 or os.environ.get("XKIRO_API_KEYS", "").strip())
    configured = bool(key_src or single_key or xkiro_key)
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    from adapters import ai_models as _ai
    state = {"gemini_configured": configured, "model": model,
             "ai_configured": configured,
             "primary_model": _ai.PRIMARY_MODEL,
             "fallback_model": _ai.FALLBACK_MODEL,
             "backend": "xkiro" if configured else "none"}
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
async def upload(file: UploadFile = File(...), run: int = 0):  # noqa: B008
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
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "upload too large; max 200 MB")
            fh.write(chunk)
    # Validate that the upload is a real, decodable image. Extension checks
    # alone let corrupt files through until load_images blows up much
    # later, after the user has waited for a job to start.
    try:
        from PIL import Image
        with Image.open(dest) as img:
            img.verify()
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"not a valid {suffix} image: {exc}") from exc
    cfg = {
        "session": session.id,
        "strip_file": strip_file,
        "backend": _default_backend(),
    }
    session.config.update(cfg)
    log.info("job=%s upload received filename=%s bytes=%d backend=%s",
             session.id, file.filename, written, cfg["backend"])
    if run:
        threading.Thread(target=pipeline.run_job, args=(session.id,),
                         daemon=True).start()
        log.info("job=%s worker started kind=segment", session.id)
    else:
        log.info("job=%s uploaded; queued (run=0, auto-run disabled)", session.id)
    return {"job_id": session.id, "status": session.status.value}


class RunRequest(BaseModel):
    session: str
    order: list[str] | None = None
    tts: str = "edge"
    voice: str = "en-US-AriaNeural"
    rate: int = 0    # edge-tts rate offset in %
    pitch: int = 0   # edge-tts pitch offset in Hz
    style: str = "recap"
    backend: str = "none"
    api_key: str = ""
    model: str = ""
    endpoint: str = ""
    cf_account_id: str = ""
    start_stage: str | None = None  # resume from a specific stage (retry)
    continue_from: str | None = None  # previous strip's session (chain)


@app.post("/api/run")
async def run(body: RunRequest):
    # Sessions are DIRECTORIES on disk; job records may be missing for
    # sessions created before job persistence existed (or evicted from
    # the store). Resolve the strip from disk first, the job record only
    # as a fallback for the stored config.
    src = store.get(body.session) or store.get_by_session(body.session)
    strip_file = ""
    if src is not None:
        strip_file = src.config.get("strip_file", "")
    if not strip_file or not (OUTPUT_DIR / body.session / strip_file).exists():
        # disk fallback: whatever strip.* the session directory has
        found = None
        for cand in (OUTPUT_DIR / body.session).glob("strip.*"):
            if cand.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                found = cand.name
                break
        if found is None:
            raise HTTPException(404, "session/strip not found; upload first")
        strip_file = found
    if body.continue_from:
        prev = (store.get(body.continue_from)
                or store.get_by_session(body.continue_from))
        if prev is None and not (OUTPUT_DIR / body.continue_from).is_dir():
            raise HTTPException(404, "continue_from session not found")
        prev_strip_name = (prev.config.get("strip_file", "")
                           if prev else "")
        prev_strip = OUTPUT_DIR / body.continue_from / prev_strip_name
        if not prev_strip.exists():
            # disk fallback for pre-persistence sessions
            found = None
            for cand in (OUTPUT_DIR / body.continue_from).glob("strip.*"):
                if cand.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                    found = cand.name
                    break
            if found is None:
                raise HTTPException(400, "continue_from session has no strip")
            prev_strip_name = found
        prev_panels = OUTPUT_DIR / body.continue_from / "panels.json"
        if not prev_panels.is_file():
            raise HTTPException(400, "continue_from session has no panels "
                                     "yet; run it first, then continue")
    job = store.create("generate", {
        "session": body.session,
        "strip_file": strip_file,
        "order": body.order,
        "tts": body.tts, "voice": body.voice, "style": body.style,
        "rate": body.rate, "pitch": body.pitch,
        "backend": body.backend,
        "start_stage": body.start_stage,
        "continue_from": body.continue_from,
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


# ---------------------------------------------------------------------------
# Panel Review — user verification/correction layer over the AI segmentation.
# panels.json (AI baseline) is never mutated; edits persist in panels_edit.json.
# ---------------------------------------------------------------------------
from . import panel_api as _panel_api  # noqa: E402
from . import voice_api as _voice_api  # noqa: E402


@app.get("/api/panels/{session}")
async def panels_get(session: str, combined: bool = False):
    return _panel_api.get_panels(session, combined=combined)


class PanelIdsBody(BaseModel):
    ids: list[str]


@app.post("/api/panels/{session}/order")
async def panels_order(session: str, body: PanelIdsBody):
    return _panel_api.set_order(session, body.ids)


@app.post("/api/panels/{session}/delete")
async def panels_delete(session: str, body: PanelIdsBody):
    return _panel_api.delete_panels(session, body.ids)


@app.post("/api/panels/{session}/restore")
async def panels_restore(session: str, body: PanelIdsBody):
    return _panel_api.restore_panels(session, body.ids)


class ReviewBody(BaseModel):
    panel_id: str
    status: str


@app.post("/api/panels/{session}/review")
async def panels_review(session: str, body: ReviewBody):
    return _panel_api.set_review(session, body.panel_id, body.status)


class ConfirmBody(BaseModel):
    review_all: bool = False


@app.post("/api/panels/{session}/confirm")
async def panels_confirm(session: str, body: ConfirmBody):
    return _panel_api.confirm(session, review_all=body.review_all)


class PanelIdBody(BaseModel):
    panel_id: str


class MergeBody(BaseModel):
    ids: list[str]


class SplitBody(BaseModel):
    panel_id: str
    fraction: float = 0.5


@app.post("/api/panels/{session}/duplicate")
async def panels_duplicate(session: str, body: PanelIdBody):
    return _panel_api.duplicate_panel(session, body.panel_id)


@app.post("/api/panels/{session}/merge")
async def panels_merge(session: str, body: MergeBody):
    return _panel_api.merge_panels(session, body.ids)


@app.post("/api/panels/{session}/split")
async def panels_split(session: str, body: SplitBody):
    return _panel_api.split_panel(session, body.panel_id, fraction=body.fraction)


# ---------------------------------------------------------------------------
# Manual Cropping — boundary-based panel definition (no AI)
# ---------------------------------------------------------------------------
class ManualBoundariesBody(BaseModel):
    boundaries: list[int]


@app.get("/api/manual-crop/{session}/state")
async def manual_crop_state(session: str):
    return manual_crop_api.get_manual_state(session)


@app.post("/api/manual-crop/{session}/boundaries")
async def manual_crop_save(session: str, body: ManualBoundariesBody):
    return manual_crop_api.save_manual_boundaries(session, body.boundaries)


@app.post("/api/manual-crop/{session}/generate")
async def manual_crop_generate(session: str):
    return manual_crop_api.generate_manual_panels(session)


@app.post("/api/manual-crop/{session}/reset")
async def manual_crop_reset(session: str):
    return manual_crop_api.reset_manual(session)


@app.get("/api/manual-crop/{session}/guides")
async def manual_crop_guides(session: str):
    return {"guides": manual_crop_api.get_auto_guides(session)}


class AutoCropBody(BaseModel):
    backend: str = "none"
    api_key: str = ""
    model: str = ""
    endpoint: str = ""


@app.post("/api/manual-crop/{session}/auto-crop")
async def manual_crop_auto_crop(session: str, body: AutoCropBody):
    """Run ONLY the AI segmentation stages (config → load → segment →
    panel validation) — no narration, no script, no render — then leave
    the detected edges in manual_crop boundaries for the user to refine.

    The existing segment job kind is reused verbatim; this endpoint just
    wraps it and seeds the manual-crop state from the result.
    """
    if not re.match(r"^[0-9a-f]{12}$", session):
        raise HTTPException(400, "invalid session id")
    d = OUTPUT_DIR / session
    if not d.is_dir():
        raise HTTPException(404, "session not found")
    strip_file = _strip_file_for(session)
    if not (d / strip_file).is_file():
        raise HTTPException(400, "session has no strip image")
    job = store.create("segment", {
        "session": session,
        "strip_file": strip_file,
        "backend": body.backend,
        "mode": "auto_crop",
    })
    kwargs = {"api_key": body.api_key or None, "model": body.model or None,
              "base_url": body.endpoint or None}
    threading.Thread(
        target=pipeline.run_job, args=(job.id,), kwargs=kwargs,
        daemon=True).start()
    return {"job_id": job.id, "status": job.status.value}


@app.get("/api/manual-crop/{session}/strip")
async def manual_crop_strip(session: str):
    d = manual_crop_api._session_dir(session)
    strip = manual_crop_api._find_strip(d)
    if strip is None:
        raise HTTPException(404, "strip not found")
    return FileResponse(strip)


# ---------------------------------------------------------------------------
# Narrator / Voice Studio — per-project voice config + previews
# ---------------------------------------------------------------------------
# NOTE: static routes like /api/voice/preview MUST be registered BEFORE the
# dynamic /api/voice/{session} route, or "preview" is captured as a session
# id and the request 400s.
@app.get("/api/voice/preview")
async def voice_preview(voice: str, rate: str = "+0%", pitch: str = "+0Hz",
                        text: str = "This is how your recap will sound."):
    import hashlib

    from . import tts_helpers
    key = hashlib.sha1(f"{voice}|{rate}|{pitch}|{text[:80]}".encode()).hexdigest()[:12]
    cache = BASE_DIR / ".cache" / "voice_preview"
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{key}.mp3"
    if not out.is_file():
        try:
            await tts_helpers.synth_one(text[:160], voice, out, timeout_s=20)
        except Exception as exc:
            raise HTTPException(502, f"preview failed: {exc}") from exc
    return FileResponse(out, media_type="audio/mpeg")


@app.get("/api/voice/{session}")
async def voice_get(session: str):
    return _voice_api.get_voice(session)


@app.post("/api/voice/{session}")
async def voice_put(session: str, body: dict):
    return _voice_api.put_voice(session, body)


@app.get("/api/voices")
async def voices_list(provider: str = "edge"):
    return await _voice_api.list_voices(provider)


class PreviewBody(BaseModel):
    cfg: dict
    text: str = ""


@app.post("/api/voice/{session}/preview")
async def voice_preview_post(session: str, body: PreviewBody):
    return await _voice_api.make_preview(session, body.cfg, body.text)


# Narration Studio — per-panel narration review/edit/regeneration.
# Overrides persist in narration_edit.json; the AI baseline is never mutated.
from . import narration_api as _narration_api  # noqa: E402


@app.get("/api/narration/{session}")
async def narration_get(session: str):
    return _narration_api.get_narration(session)


class NarrTextBody(BaseModel):
    panel_id: str
    text: str = ""


@app.post("/api/narration/{session}/text")
async def narration_text(session: str, body: NarrTextBody):
    return _narration_api.set_text(session, body.panel_id, body.text)


@app.post("/api/narration/{session}/reset")
async def narration_reset(session: str, body: NarrTextBody):
    return _narration_api.reset_text(session, body.panel_id)


class NarrStyleBody(BaseModel):
    style: str


@app.post("/api/narration/{session}/style")
async def narration_style(session: str, body: NarrStyleBody):
    return _narration_api.set_style(session, body.style)


class NarrRegenBody(BaseModel):
    panel_id: str
    api_key: str = ""
    model: str = ""


@app.post("/api/narration/{session}/regenerate")
async def narration_regen(session: str, body: NarrRegenBody):
    return _narration_api.regenerate(session, body.panel_id,
                                     api_key=body.api_key, model=body.model)


class NarrGenAllBody(BaseModel):
    api_key: str = ""
    model: str = ""


@app.post("/api/narration/{session}/generate")
async def narration_generate_all(session: str, body: NarrGenAllBody):
    """START button for post-crop AI narration (Qwen -> Mistral fallback).

    Cropping itself needs no AI (deterministic blank-row cut); this fills
    narration/dialogue for the already-cropped panel PNGs. Runs as a
    background job — poll /api/jobs/{job_id}. Geometry is never modified.
    """
    return _narration_api.generate_all(session, api_key=body.api_key,
                                       model=body.model)


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str, logs: int = 0):
    job = store.get(job_id)
    if job is None:
        # The Logs view addresses jobs by SESSION id, which only matches a
        # job id for upload jobs — generate jobs get fresh UUIDs. Fall back
        # to the session's latest job (memory or rehydrated from disk).
        job = store.get_by_session(job_id)
    if job is None:
        synth = _synth_session_status(job_id, logs=bool(logs))
        if synth is not None:
            return JSONResponse(synth)
        raise HTTPException(404, "job not found")
    return JSONResponse(job.to_dict(include_logs=bool(logs)))


def _synth_session_status(session: str, *, logs: bool) -> dict | None:
    """Last-resort status for a session dir with no job record anywhere.

    Happens for sessions created before log persistence existed, or that
    never ran. Returns a synthetic payload (instead of a bare 404) listing
    what IS on disk, so the Logs view shows an explanation rather than
    "job no longer exists on the server".
    """
    d = OUTPUT_DIR / session
    if not d.is_dir():
        return None
    try:
        files = sorted(p.name for p in d.iterdir() if p.is_file())
    except OSError:
        return None
    panel_count = 0
    pj = d / "panels.json"
    if pj.is_file():
        try:
            import json as _json
            panel_count = len(_json.loads(pj.read_text("utf-8")).get("panels", []))
        except (OSError, ValueError):
            pass
    payload: dict = {
        "job_id": session,
        "kind": "session",
        "status": "no_job",
        "stage": None,
        "progress": 0,
        "error": None,
        "panels": panel_count,
        "outputs": {},
        "synthetic": True,
    }
    if logs:
        payload["logs"] = [{
            "t": time.time(),
            "level": "INFO",
            "stage": None,
            "msg": ("no job history for this session (it predates log "
                    "persistence, or it never ran). Files on disk: "
                    + (", ".join(files) if files else "(none)")
                    + ". Re-run the session to generate fresh logs."),
        }]
    return payload


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
    session = job.config.get("session", job_id) if job is not None else job_id
    base = (OUTPUT_DIR / session).resolve()
    target = (base / name).resolve()
    if target.parent != base and base not in target.parents:
        raise HTTPException(400, "bad path")
    if not target.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(target)


# --------------------------------------------------------------------------- #
# Panel validation + review routes (Part 2.4)
# --------------------------------------------------------------------------- #
import asyncio  # noqa: E402
import json as _json  # noqa: E402

from fastapi.responses import StreamingResponse  # noqa: E402

import panel_validator as pv  # noqa: E402


@app.get("/api/sessions/{session}/validation")
async def get_validation(session: str):
    d = OUTPUT_DIR / session
    rep = pv.load_report(d)
    if rep is None:
        raise HTTPException(404, "no validation report; segment first")
    return pv.apply_decisions(rep, pv.load_review(d))


@app.post("/api/sessions/{session}/panels/{panel_id}/decision")
async def panel_decision(session: str, panel_id: str, body: dict):
    decision = body.get("decision")
    if decision not in ("keep", "delete"):
        raise HTTPException(400, "decision must be keep|delete")
    d = OUTPUT_DIR / session
    review = pv.load_review(d)
    review["decisions"][panel_id] = decision
    pv.save_review(d, review)
    return {"ok": True}


@app.post("/api/sessions/{session}/confirm")
async def confirm_panels(session: str, body: dict):
    d = OUTPUT_DIR / session
    review = pv.load_review(d)
    order = body.get("order")
    if order:
        review["order"] = order
    review["confirmed"] = True
    pv.save_review(d, review)
    return {"ok": True, "order": review.get("order")}


@app.post("/api/sessions/{session}/unconfirm")
async def unconfirm(session: str):
    d = OUTPUT_DIR / session
    review = pv.load_review(d)
    review["confirmed"] = False
    pv.save_review(d, review)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    async def gen():
        last = None
        while True:
            job = store.get(job_id)
            if job is None:
                yield "event: error\ndata: job gone\n\n"
                return
            snap = (job.status.value, job.stage or "",
                    int(job.progress or 0), job.error or "")
            if snap != last:
                last = snap
                yield ("data: " + _json.dumps({
                    "status": snap[0], "stage": snap[1],
                    "progress": snap[2], "error": snap[3]}) + "\n\n")
            if snap[0] in ("completed", "failed", "cancelled"):
                return
            await asyncio.sleep(0.8)
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/jobs/{job_id}/render-cancel")
async def render_cancel(job_id: str):
    from .render_worker import cancel_render
    ok = cancel_render(job_id)
    return {"ok": ok}


# --------------------------------------------------------------------------- #
# Step-by-Step Pipeline / Checkpoints (see webapp/checkpoint.py)
# --------------------------------------------------------------------------- #
from . import checkpoint as _cp  # noqa: E402


class StepRunRequest(BaseModel):
    session: str
    stage: str | None = None          # single stage name (Run Next Step)
    from_stage: str | None = None      # Run Until: start (default: next)
    until_stage: str | None = None     # Run Until: stop after this stage
    tts: str = "edge"
    voice: str = "en-US-AriaNeural"
    style: str = "recap"
    backend: str = "none"
    api_key: str = ""
    model: str = ""
    endpoint: str = ""
    cf_account_id: str = ""


def _step_credentials(body: StepRunRequest) -> dict:
    return {"api_key": body.api_key or None, "model": body.model or None,
            "base_url": body.endpoint or None,
            "cf_account_id": body.cf_account_id or None}


def _validate_step_stage(stage: str | None, allow_none: bool = False
                        ) -> str | None:
    if stage is None:
        if allow_none:
            return None
        raise HTTPException(400, "stage required")
    if stage not in _cp.BY_PIPELINE_NAME:
        raise HTTPException(
            400, f"unknown stage {stage!r}; valid: "
                 f"{sorted(_cp.BY_PIPELINE_NAME)}")
    return stage


def _validate_step_session(session: str) -> None:
    import re as _re
    if not _re.match(r"^[0-9a-f]{12}$", session):
        raise HTTPException(400, "invalid session id")
    if not (OUTPUT_DIR / session).is_dir():
        raise HTTPException(404, "session not found")


@app.get("/api/pipeline/{session}")
async def pipeline_state_get(session: str):
    """Ledger + step list + staleness for the visual pipeline state."""
    _validate_step_session(session)
    state = _cp.load_state(session)
    drift = _cp.detect_stale_steps(session)
    running = None
    for j in store._jobs.values():
        if (j.config.get("session") == session
                and j.kind in ("pipeline_step", "pipeline_until")
                and j.status.value in ("queued", "running")):
            running = {"job_id": j.id, "status": j.status.value,
                       "stage": j.stage}
            break
    return {
        "steps": _cp.STEP_SEQUENCE,
        "state": state,
        "stale": drift["stale"],
        "running": running,
        "events_tail": _cp.load_events(session, limit=50),
    }


@app.get("/api/pipeline/{session}/events")
async def pipeline_events_get(session: str, step: int | None = None,
                              severity: str | None = None,
                              limit: int = 400):
    """Persistent debugger log (survives refresh/restart)."""
    _validate_step_session(session)
    return {"events": _cp.load_events(session, step=step,
                                      severity=severity, limit=limit)}


@app.get("/api/pipeline/{session}/step/{step_no}")
async def pipeline_step_detail(session: str, step_no: int):
    """Completed-step inspection: output, logs, duration, artifacts."""
    _validate_step_session(session)
    if step_no not in _cp.BY_STEP:
        raise HTTPException(400, f"invalid step {step_no}")
    entry = _cp.step_entry(session, step_no)
    events = _cp.load_events(session, step=step_no, limit=200)
    return {"step": _cp.BY_STEP[step_no], "entry": entry, "events": events}


@app.post("/api/pipeline/{session}/run-step")
async def pipeline_run_step(session: str, body: StepRunRequest):
    """Run ONE stage (Step-by-Step): worker executes it, checkpoints,
    and the job ends paused-at-next."""
    _validate_step_session(session)
    state = _cp.load_state(session)
    # default target: the ledger's current step
    stage = body.stage or _cp.BY_STEP.get(state.get("current_step", 1), {}).get(
        "pipeline_name")
    if stage is None:
        raise HTTPException(400, "no runnable next step (ledger empty?)")
    _validate_step_stage(stage)
    step_no = _cp.BY_PIPELINE_NAME[stage]["step"]
    if state.get("status") == "running":
        raise HTTPException(409, "a step is already running for this session")
    cfg = {"session": session, "strip_file": _strip_file_for(session),
           "stage": stage, "tts": body.tts, "voice": body.voice,
           "style": body.style, "backend": body.backend,
           "start_stage": stage, "end_stage": stage}
    job = store.create("pipeline_step", cfg)
    # persist mode on the ledger
    _cp.save_state(session, {**state, "mode": "step"})
    _cp.log_event(session, step=step_no, event_type="user_action",
                  message=f"user requested step {step_no} ({stage})",
                  severity="INFO")
    threading.Thread(
        target=pipeline.run_steps_job, args=(job.id, stage, None),
        kwargs=_step_credentials(body), daemon=True).start()
    return {"job_id": job.id, "step": step_no, "stage": stage,
            "status": job.status.value}


@app.post("/api/pipeline/{session}/run-until")
async def pipeline_run_until(session: str, body: StepRunRequest):
    """Run stages from the current step up to (and including) until_stage,
    then STOP (checkpoint). Automation = run-until with no stop."""
    _validate_step_session(session)
    state = _cp.load_state(session)
    until = _validate_step_stage(body.until_stage, allow_none=True)
    cur = state.get("current_step", 1)
    from_step = _cp.BY_PIPELINE_NAME.get(body.from_stage or "", {}).get(
        "step") or cur
    from_stage = _cp.BY_STEP[from_step]["pipeline_name"]
    if state.get("status") == "running":
        raise HTTPException(409, "a step is already running")
    cfg = {"session": session, "strip_file": _strip_file_for(session),
           "start_stage": from_stage, "end_stage": until or "create_editor_project",
           "tts": body.tts, "voice": body.voice, "style": body.style,
           "backend": body.backend}
    job = store.create("pipeline_until", cfg)
    _cp.save_state(session, {**state, "mode": "automation"})
    _cp.log_event(session, step=from_step, event_type="user_action",
                  message=f"user requested run until {until or 'end'}",
                  severity="INFO")
    last = until or "create_editor_project"
    threading.Thread(
        target=pipeline.run_steps_job, args=(job.id, from_stage, last),
        kwargs=_step_credentials(body), daemon=True).start()
    return {"job_id": job.id, "from_stage": from_stage, "until_stage": last,
            "status": job.status.value}


def _strip_file_for(session: str) -> str:
    d = OUTPUT_DIR / session
    for cand in d.glob("strip.*"):
        if cand.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            return cand.name
    j = store.get_by_session(session)
    if j:
        return j.config.get("strip_file", "strip.png")
    return "strip.png"


@app.post("/api/pipeline/{session}/retry-step")
async def pipeline_retry_step(session: str, body: StepRunRequest):
    """Retry ONLY the failed step (default: the ledger's current)."""
    _validate_step_session(session)
    state = _cp.load_state(session)
    stage = body.stage or _cp.BY_STEP.get(state.get("current_step", 1), {}).get(
        "pipeline_name")
    _validate_step_stage(stage)
    # A retry clears the failed mark; the ledger stays at this step.
    return await pipeline_run_step(session, body)


@app.post("/api/pipeline/{session}/reset-from")
async def pipeline_reset_from(session: str, body: dict):
    """Dependency-aware invalidation: mark `stage` and all later steps
    stale, preserving earlier artifacts/checkpoints."""
    _validate_step_session(session)
    stage = body.get("stage")
    stage = _validate_step_stage(stage)
    step_no = _cp.BY_PIPELINE_NAME[stage]["step"]
    state = _cp.mark_stale_from(
        session, step_no,
        f"user reset from {stage}; this and later steps invalidated")
    _cp.log_event(session, step=step_no, event_type="user_action",
                  message=f"user reset from step {step_no} ({stage})",
                  severity="INFO")
    return {"ok": True, "state": state}


@app.post("/api/pipeline/{session}/invalidate")
async def pipeline_invalidate(session: str, body: dict):
    """Hook for edit APIs: invalidate steps dependent on `changed`."""
    _validate_step_session(session)
    changed = body.get("changed")
    if changed not in _cp.EDIT_INVALIDATION:
        raise HTTPException(400, f"unknown artifact {changed!r}")
    state = _cp.apply_edit_invalidation(session, changed)
    return {"ok": True, "state": state}


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
        raise HTTPException(404, "editor project not found; generate a video first") from None


@app.post("/api/editor/{session}/create")
async def editor_create(session: str):
    try:
        return create_project_from_generation(session)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from None


@app.post("/api/editor/{session}")
async def editor_save(session: str, body: dict):
    try:
        return save_project(session, body)
    except FileNotFoundError:
        raise HTTPException(404, "editor project not found") from None


@app.post("/api/editor/{session}/reorder")
async def editor_reorder(session: str, body: dict):
    try:
        return reorder_panels(session, body.get("order", []))
    except FileNotFoundError:
        raise HTTPException(404, "editor project not found") from None


@app.post("/api/editor/{session}/duration")
async def editor_duration(session: str, body: dict):
    try:
        return set_duration(session, body["panel_id"], float(body["duration"]))
    except (FileNotFoundError, KeyError, ValueError, TypeError):
        raise HTTPException(400, "bad request") from None


@app.post("/api/editor/{session}/effect")
async def editor_effect(session: str, body: dict):
    try:
        return set_effect(session, body["panel_id"], body["kind"], float(body.get("duration", 0.0)))
    except (FileNotFoundError, KeyError, ValueError, TypeError):
        raise HTTPException(400, "bad request") from None


@app.post("/api/editor/{session}/caption")
async def editor_caption(session: str, body: dict):
    try:
        cid = body.pop("id")
        allowed = {"text", "start_seconds", "end_seconds"}
        filtered = {k: v for k, v in body.items() if k in allowed}
        return update_caption(session, cid, **filtered)
    except (FileNotFoundError, KeyError, TypeError):
        raise HTTPException(400, "bad request") from None


@app.post("/api/editor/{session}/narration")
async def editor_narration(session: str, body: dict):
    try:
        panel_id = body["panel_id"]
        text = body.get("text", "")
        return update_narration(session, panel_id, text)
    except (FileNotFoundError, KeyError, TypeError):
        raise HTTPException(400, "bad request") from None


@app.post("/api/editor/{session}/transition")
async def editor_transition(session: str, body: dict):
    try:
        return set_transition(session, body["from_panel_id"], body["to_panel_id"],
                              body["type"], float(body["duration"]))
    except (FileNotFoundError, KeyError, ValueError, TypeError):
        raise HTTPException(400, "bad request") from None


@app.post("/api/editor/{session}/panel/remove")
async def editor_remove(session: str, body: dict):
    try:
        return remove_panel(session, body["panel_id"])
    except (FileNotFoundError, KeyError):
        raise HTTPException(400, "bad request") from None


@app.post("/api/editor/{session}/panel/add")
async def editor_add(session: str, body: dict):
    try:
        return add_panel(session, body["panel_id"], body.get("after"))
    except (FileNotFoundError, KeyError):
        raise HTTPException(400, "bad request") from None


@app.post("/api/editor/{session}/undo")
async def editor_undo(session: str):
    try:
        return undo(session)
    except FileNotFoundError:
        raise HTTPException(404, "editor project not found") from None


@app.post("/api/editor/{session}/redo")
async def editor_redo(session: str):
    try:
        return redo(session)
    except FileNotFoundError:
        raise HTTPException(404, "editor project not found") from None


@app.post("/api/editor/{session}/reset")
async def editor_reset(session: str):
    try:
        return reset_to_automated(session)
    except FileNotFoundError:
        raise HTTPException(404, "editor project not found") from None


@app.post("/api/editor/{session}/render")
async def editor_render(session: str, body: dict | None = None):
    body = body or {}
    try:
        return start_render(session, body.get("cfg", {}))
    except FileNotFoundError:
        raise HTTPException(404, "editor project not found") from None


@app.get("/api/editor/{session}/ai-review")
async def editor_ai_review(session: str):
    try:
        return editor_api.get_ai_review_data(session)
    except FileNotFoundError:
        raise HTTPException(404, "panels not found") from None


@app.get("/api/editor/{session}/panel-confidence")
async def editor_panel_confidence(session: str):
    try:
        return editor_api.get_panel_confidence(session)
    except FileNotFoundError:
        raise HTTPException(404, "panels not found") from None


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
        from adapters.schemas import NarrationArtifact
        from guided_cutter import CutArtifact
        artifact = CutArtifact.model_validate_json(panels_json.read_text("utf-8"))
        narration = NarrationArtifact.model_validate_json(narration_json.read_text("utf-8"))
    except Exception as exc:
        raise HTTPException(500, f"cannot read project files: {exc}") from None
    panel = next((p for p in artifact.panels if p.id == panel_id), None)
    if not panel:
        raise HTTPException(404, f"panel {panel_id} not found")
    entry = next((n for n in narration.entries if n.id == panel_id), None)
    if not entry:
        raise HTTPException(404, f"narration for {panel_id} not found")
    cut_art = CutArtifact(
        source=artifact.source, width=artifact.width, height=artifact.height,
        plan_hash=artifact.plan_hash, config=artifact.config, panels=[panel])
    try:
        import narrator
        new_text = narrator.make_script_from_cut(cut_art, style="recap")
        entry.text = new_text
        narration_json.write_text(narration.model_dump_json(indent=2) + "\n", "utf-8")
        editor_path = d / "editor.json"
        if editor_path.is_file():
            proj = editor_api._get_project(session)
            if proj:
                for e in proj.edited_timeline:
                    if e.get("panel_id") == panel_id:
                        e["narration"] = new_text
                        e["needs_render"] = True
                Editor(proj).save(editor_path)
        return {"ok": True, "panel_id": panel_id, "text": new_text}
    except Exception as exc:
        raise HTTPException(500, f"narration regeneration failed: {exc}") from None
