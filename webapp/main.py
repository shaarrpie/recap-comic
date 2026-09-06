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
from .jobs import store

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=False)
setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"),
              log_dir=BASE_DIR / "logs")
log = get_logger(__name__)

app = FastAPI(title="recap-comic webapp")
OUTPUT_DIR = BASE_DIR / "webapp_output"
OUTPUT_DIR.mkdir(exist_ok=True)

_last_config_state: dict | None = None


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "webapp" / "static" / "index.html")


@app.get("/api/config")
async def config():
    global _last_config_state
    configured = bool(os.environ.get("GEMINI_API_KEY"))
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    state = {"gemini_configured": configured, "model": model}
    if state != _last_config_state:
        log.info("config check gemini_configured=%s model=%s",
                 configured, model)
        _last_config_state = state
    return state


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
    content = await file.read()
    (session_dir / strip_file).write_bytes(content)
    session.config.update({"session": session.id,
                           "strip_file": strip_file})
    log.info("job=%s upload received filename=%s bytes=%d",
             session.id, file.filename, len(content))
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
    })
    log.info("job=%s created kind=generate session=%s order=%s",
             job.id, body.session,
             "user" if body.order else "default")
    threading.Thread(target=pipeline.run_job, args=(job.id,),
                     daemon=True).start()
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
