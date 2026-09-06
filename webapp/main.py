# webapp/main.py
"""FastAPI webapp for recap-comic.

Endpoints:
  GET  /            — serve the single-page UI
  POST /api/upload  — upload a strip, start processing
  GET  /api/status/{job_id} — poll progress
  GET  /api/download/{job_id}/{filename} — download output files
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env without overriding variables already set in the process
# (the user may have exported them in the shell before starting uvicorn).
load_dotenv(BASE_DIR / ".env", override=False)

app = FastAPI(title="recap-comic webapp")

OUTPUT_DIR = BASE_DIR / "webapp_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory job store (sufficient for single-user local use)
jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _json_safe(obj: Any) -> Any:
    """Convert numpy / non-JSON-serializable types to Python natives."""
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _update(job_id: str, **kwargs: Any) -> None:
    with _lock:
        jobs[job_id].update(kwargs)


def _run_pipeline(job_id: str, strip_path: Path, backend: str,
                  out_dir: Path) -> None:
    try:
        _update(job_id, status="running", step="phase1",
                message="Phase 1: AI pre-read...")
        import guided_pipeline as gp
        import strip_analyzer as sa
        from narrator import narrate_plan

        cache_dir = BASE_DIR / ".cache" / "recap-comic"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1 + 2: guided run
        plan, artifact, used = gp.run_guided(
            strip_path, out_dir, backend_name=backend,
            cache_dir=cache_dir, force=False)

        _update(job_id, step="narration",
                message="Generating narration script...")
        plan_path = out_dir / "plan.json"
        plan_path.write_text(plan.model_dump_json(indent=2), "utf-8")

        narration_path = out_dir / "narration.txt"
        narrate_plan(plan_path, narration_path, style="recap")

        panels = []
        if artifact is not None:
            for p in artifact.panels:
                panels.append({
                    "id": p.id,
                    "panel_index": p.panel_index,
                    "y_start": p.y_start,
                    "y_end": p.y_end,
                    "narration": p.narration,
                    "dialogue": p.dialogue,
                    "panel_type": p.panel_type,
                    "confidence": p.confidence,
                    "image_file": p.image_file,
                })

        _update(job_id, status="done", step="done", message="Complete",
                panels=panels,
                plan_path=str(plan_path.relative_to(BASE_DIR)),
                narration_path=str(narration_path.relative_to(BASE_DIR)),
                panels_count=len(panels),
                used_fallback=used,
                provenance=plan.provenance,
                model=plan.model)

    except Exception as exc:
        _update(job_id, status="error", step="error",
                message=str(exc))


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return FileResponse(BASE_DIR / "webapp" / "static" / "index.html")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...),
                 backend: str = "cloudflare") -> JSONResponse:
    allowed = {"png", "jpg", "jpeg", "webp"}
    suffix = Path(file.filename or "upload.png").suffix.lower().lstrip(".")
    if suffix not in allowed:
        raise HTTPException(400, f"unsupported format: {suffix}")

    job_id = uuid.uuid4().hex[:12]
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    strip_path = job_dir / f"strip{suffix}"
    content = await file.read()
    strip_path.write_bytes(content)

    with _lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "step": "queued",
            "message": "Queued",
            "backend": backend,
            "filename": file.filename,
            "strip_path": str(strip_path.relative_to(BASE_DIR)),
            "created_at": time.time(),
            "panels": [],
            "panels_count": 0,
            "used_fallback": False,
            "provenance": "",
            "model": "",
            "plan_path": "",
            "narration_path": "",
            "error": None,
        }

    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, strip_path, backend, job_dir),
        daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id, "status": "queued"})


@app.get("/api/status/{job_id}")
async def status(job_id: str) -> JSONResponse:
    if job_id not in jobs:
        raise HTTPException(404, "job not found")
    with _lock:
        data = _json_safe(dict(jobs[job_id]))
    return JSONResponse(data)


@app.get("/api/download/{job_id}/{filename}")
async def download(job_id: str, filename: str) -> FileResponse:
    if job_id not in jobs:
        raise HTTPException(404, "job not found")
    path = OUTPUT_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(404, "file not found")
    return FileResponse(path, filename=filename)


# Mount static files AFTER API routes so they don't shadow them
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "webapp" / "static")),
          name="static")
