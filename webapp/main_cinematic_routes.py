# webapp/main_cinematic_routes.py
"""Cinematic Studio routes for the recap-comic webapp.

Attach once from webapp/main.py:

    from .main_cinematic_routes import attach_routes as _attach_cin
    _attach_cin(app)

Semi-auto `/api/pipeline/step` delegates to the EXISTING checkpointed
step worker (pipeline.run_steps_job) rather than a duplicate runner:
the ledger, artifact-validation gates, and 409 concurrency guards from
webapp/main.py apply unchanged. `tts_audio` is accepted as an alias for
the real stage name for frontend compatibility.
"""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.environ.get("RECAP_OUTPUT_DIR")
                  or BASE_DIR / "webapp_output")


# ── Request schemas ──────────────────────────────────────────────
class CinematicConfigBody(BaseModel):
    enabled: bool = True
    style: str = "dynamic"            # dynamic | subtle
    glitch_transitions: bool = True
    letterbox: bool = False
    bgm_volume: float = 0.18
    color_preset: str = "manhwa"
    global_overrides: dict[str, Any] = {}
    caption_style: dict[str, Any] | None = None


class PanelOverrideBody(BaseModel):
    panel_id: str
    overrides: dict[str, Any] | None = None   # None = clear override


class PipelineStepBody(BaseModel):
    """Semi-auto: run exactly ONE named stage."""
    session: str
    stage: str
    tts: str = "edge"
    voice: str = "en-US-AriaNeural"
    rate: int = 0
    pitch: int = 0
    style: str = "recap"
    backend: str = "none"
    api_key: str = ""
    model: str = ""


# Frontend stage name -> real pipeline stage name (see webapp/pipeline.py
# PIPELINES["generate"] and webapp/checkpoint.py BY_PIPELINE_NAME)
_STAGE_ALIASES = {
    "segment_panels": "segment_panels",
    "gemini_narration": "gemini_narration",
    "build_script": "build_script",
    "tts_audio": "tts_audio",     # legacy alias; validated below
    "render_video": "render_video",
}


def attach_routes(app: FastAPI) -> None:  # noqa: C901
    from . import cinematic_api as _cin
    from . import pipeline as _pipe
    from .jobs import store

    # ── Cinematic config ───────────────────────────────────────────
    @app.get("/api/cinematic/{session}/config")
    async def cinematic_config_get(session: str):
        return _cin.get_config(session)

    @app.post("/api/cinematic/{session}/config")
    async def cinematic_config_post(session: str, body: CinematicConfigBody):
        if body.style not in ("dynamic", "subtle"):
            raise HTTPException(400, "style must be dynamic|subtle")
        if body.color_preset not in _cin.COLOR_PRESETS:
            raise HTTPException(400, f"unknown color_preset "
                                     f"{body.color_preset!r}")
        cfg = _cin.get_config(session)
        cfg.update(body.model_dump(exclude_none=True))
        return _cin.save_config(session, cfg)

    @app.post("/api/cinematic/{session}/panel-override")
    async def cinematic_panel_override(session: str, body: PanelOverrideBody):
        return _cin.set_panel_override(session, body.panel_id,
                                       body.overrides)

    # ── Effects manifest (static; frontend renders sliders from it) ─
    @app.get("/api/cinematic/effects")
    async def cinematic_effects_manifest():
        return _cin.get_effects_manifest()

    # ── BGM upload ─────────────────────────────────────────────────
    @app.post("/api/cinematic/{session}/bgm")
    async def cinematic_bgm_upload(session: str,
                                    file: UploadFile = File(...)):  # noqa: B008
        allowed = {".mp3", ".m4a", ".ogg", ".flac", ".wav"}
        suffix = Path(file.filename or "bgm.mp3").suffix.lower()
        if suffix not in allowed:
            raise HTTPException(400, f"unsupported audio format: {suffix}")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)
        try:
            return _cin.upload_bgm(session, tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    # ── Apply / render ─────────────────────────────────────────────
    @app.post("/api/cinematic/{session}/apply")
    async def cinematic_apply(session: str):
        """Background cinematic render for the session."""
        return _cin.apply_cinematic(session, store)

    # ── Panel preview (short clip, blocking) ────────────────────────
    @app.get("/api/cinematic/{session}/preview/{panel_id}")
    async def cinematic_preview(session: str, panel_id: str):
        out_path = _cin.generate_panel_preview(session, panel_id)
        return FileResponse(out_path, media_type="video/mp4",
                            filename=f"preview_{panel_id}.mp4")

    # ── Serve the cinematic output video ───────────────────────────
    @app.get("/api/cinematic/{session}/video")
    async def cinematic_video(session: str):
        d = _cin._session_dir(session)
        mp4 = d / "cinematic_recap.mp4"
        if not mp4.is_file():
            raise HTTPException(404, "cinematic video not rendered yet")
        return FileResponse(mp4, media_type="video/mp4",
                            filename="cinematic_recap.mp4")

    # ── Semi-auto: run a single pipeline stage on demand ────────────
    @app.post("/api/pipeline/step")
    async def pipeline_step(body: PipelineStepBody):
        """Semi-automation: run exactly ONE real pipeline stage via the
        existing checkpointed step worker (run_steps_job), so the ledger,
        validation gates, and 409 concurrency guard all apply."""
        from webapp import checkpoint as _cp
        from webapp.main import _guard_no_active_step_job

        stage = _STAGE_ALIASES.get(body.stage)
        if stage is None:
            raise HTTPException(400, f"unknown stage {body.stage!r}; valid: "
                                     f"{sorted(_STAGE_ALIASES)}")
        # The legacy tts_audio alias is no longer a schedulable stage
        # (TTS runs inside the render stage): map it to render_video so
        # old UIs keep working instead of silently doing nothing.
        if stage == "tts_audio":
            stage = "render_video"
        if stage not in _cp.BY_PIPELINE_NAME:
            raise HTTPException(400, f"unknown stage {stage!r}; valid: "
                                     f"{sorted(_cp.BY_PIPELINE_NAME)}")

        _guard_no_active_step_job(body.session)
        from webapp.main import _strip_file_for, _validate_step_session
        _validate_step_session(body.session)

        job = store.create("pipeline_step", {
            "session": body.session,
            "start_stage": stage,
            "end_stage": stage,
            "stage": stage,
            "strip_file": _strip_file_for(body.session),
            "tts": body.tts, "voice": body.voice,
            "style": body.style, "backend": body.backend,
            "rate": body.rate, "pitch": body.pitch,
        })
        kwargs = {
            "api_key": body.api_key or None,
            "model": body.model or None,
        }
        threading.Thread(
            target=_pipe.run_steps_job,
            args=(job.id, stage, stage),
            kwargs=kwargs,
            daemon=True,
        ).start()
        return {"job_id": job.id, "stage": stage, "status": "queued"}

    # ── Semi-auto: stage completion checklist for a session ────────
    @app.get("/api/pipeline/{session}/stages")
    async def pipeline_stages(session: str):
        """Which stages have produced artifacts, for the semi-auto
        checklist UI (artifact-based, same signals the checkpoint
        ledger validates)."""
        import re as _re
        if not _re.match(r"^[0-9a-f]{12}$", session):
            raise HTTPException(400, "invalid session id")
        d = OUTPUT_DIR / session
        if not d.is_dir():
            raise HTTPException(404, "session not found")

        def has(name: str) -> bool:
            return (d / name).is_file()

        return {
            "session": session,
            "stages": {
                "segment_panels": has("panels.json"),
                "gemini_narration": has("narration.json"),
                "tts_audio": has("audio.json"),
                "render_video": has("recap.mp4"),
                "cinematic_render": has("cinematic_recap.mp4"),
            },
            "cinematic_enabled":
                _cin.get_config(session).get("enabled", True),
        }
