# webapp/cinematic_api.py
"""Cinematic-effects API for the recap-comic webapp.

Manages per-session cinematic config (`cinematic_config.json`), triggers
background renders via cinematic_effects.make_cinematic_video, and
generates short per-panel previews for the Manual FX mode.

The upstream draft called a `CinematicEffects` class that does not exist
in cinematic_effects.py; this adaptation composes a CinematicConfig and
calls the real entry points. Global overrides map onto CinematicConfig
fields; the COLOR_PRESETS map onto the grade_* fields.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.environ.get("RECAP_OUTPUT_DIR") or BASE_DIR / "webapp_output")

_SESSION_RE = re.compile(r"^[0-9a-f]{12}$")

# Default cinematic config written to new sessions
DEFAULT_CINEMATIC_CONFIG: dict[str, Any] = {
    "enabled": True,
    "style": "dynamic",          # "dynamic" | "subtle"
    "glitch_transitions": True,
    "letterbox": False,
    "bgm_path": None,
    "bgm_volume": 0.18,
    "color_preset": "manhwa",
    "global_overrides": {},      # keys must be CinematicConfig fields
    "panel_overrides": {},       # panel_id -> partial effect hints
    "caption_style": None,       # set by the enhanced editor
}

EFFECTS_MANIFEST = [
    {"id": "punch_zoom", "name": "Punch Zoom",
     "triggers": "action",
     "params": {"punch_scale": {"type": "float", "min": 1.05, "max": 1.5,
                               "default": 1.22},
                "punch_frames": {"type": "int", "min": 3, "max": 18,
                                "default": 9}}},
    {"id": "ken_burns", "name": "Ken Burns",
     "triggers": "all",
     "params": {"kb_zoom_end": {"type": "float", "min": 1.0, "max": 1.18,
                                "default": 1.12}}},
    {"id": "shake", "name": "Screen Shake",
     "triggers": "action",
     "params": {"shake_amplitude_px": {"type": "float", "min": 1,
                                       "max": 20, "default": 7},
                "shake_duration": {"type": "float", "min": 0.1, "max": 0.8,
                                   "default": 0.25}}},
    {"id": "glitch", "name": "Glitch Cut",
     "triggers": "transition",
     "params": {"glitch_duration": {"type": "float", "min": 0.1, "max": 0.6,
                                    "default": 0.22}}},
    {"id": "speed_lines", "name": "Speed Lines",
     "triggers": "action",
     "params": {"speedlines_opacity": {"type": "float", "min": 0.05,
                                       "max": 0.5, "default": 0.22}}},
    {"id": "vignette", "name": "Vignette",
     "triggers": "all",
     "params": {"vignette_angle": {"type": "float", "min": 0.2, "max": 1.5,
                                   "default": 0.8}}},
    {"id": "color_grade", "name": "Color Grade",
     "triggers": "all",
     "params": {"color_preset": {
         "type": "select",
         "options": ["manhwa", "dark", "warm", "cold", "vivid", "bw"],
         "default": "manhwa"}}},
]

# Color preset -> CinematicConfig grade_* field overrides
COLOR_PRESETS: dict[str, dict[str, float]] = {
    "manhwa": {"grade_contrast": 1.06, "grade_saturation": 1.10,
               "grade_shadows": -0.04, "grade_highlights": 0.02,
               "grade_brightness": -0.01},
    "dark":   {"grade_contrast": 1.12, "grade_saturation": 0.95,
               "grade_shadows": -0.06, "grade_highlights": 0.01,
               "grade_brightness": -0.06},
    "warm":   {"grade_contrast": 1.04, "grade_saturation": 1.15,
               "grade_shadows": 0.02, "grade_highlights": 0.04,
               "grade_brightness": 0.01},
    "cold":   {"grade_contrast": 1.05, "grade_saturation": 0.92,
               "grade_shadows": -0.04, "grade_highlights": -0.02,
               "grade_brightness": -0.02},
    "vivid":  {"grade_contrast": 1.10, "grade_saturation": 1.35,
               "grade_shadows": -0.02, "grade_highlights": 0.02,
               "grade_brightness": 0.00},
    "bw":     {"grade_contrast": 1.08, "grade_saturation": 0.0,
               "grade_shadows": 0.0, "grade_highlights": 0.0,
               "grade_brightness": 0.0},
}

# Frontend slider keys -> CinematicConfig fields
_OVERRIDE_FIELD_MAP = {
    "zoom_peak": "punch_scale",
    "shake_intensity": "shake_amplitude_px",
    "speedlines_opacity": "speedlines_opacity",
    "kb_zoom_range": "kb_zoom_end",      # 0..0.2 slider -> 1.0..1.2 scale
    "enable_speedlines": "speedlines_enabled",
    "enable_shake": "shake_enabled",
    "enable_vignette": "vignette_enabled",
}


def _session_dir(session: str) -> Path:
    """Validated session dir. Session ids are uuid4().hex[:12]; anything
    else (e.g. '..') must never reach the filesystem."""
    if not _SESSION_RE.fullmatch(session or ""):
        raise HTTPException(400, "invalid session id")
    d = OUTPUT_DIR / session
    if not d.is_dir():
        raise HTTPException(404, f"session not found: {session}")
    return d


def _config_path(session: str) -> Path:
    return _session_dir(session) / "cinematic_config.json"


def get_config(session: str) -> dict[str, Any]:
    p = _config_path(session)
    if p.is_file():
        try:
            cfg = json.loads(p.read_text("utf-8"))
            if isinstance(cfg, dict):
                merged = dict(DEFAULT_CINEMATIC_CONFIG)
                merged.update(cfg)
                return merged
        except Exception:  # noqa: BLE001 - fall back to defaults
            pass
    return dict(DEFAULT_CINEMATIC_CONFIG)


def save_config(session: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Persist a merged config (unknown keys dropped, types trusted only
    for known-safe fields)."""
    merged = get_config(session)
    for k in DEFAULT_CINEMATIC_CONFIG:
        if k in cfg:
            merged[k] = cfg[k]
    # coerce numeric knobs so a bad type can never poison a render
    try:
        merged["bgm_volume"] = max(0.0, min(1.0, float(merged["bgm_volume"])))
    except (TypeError, ValueError):
        merged["bgm_volume"] = 0.18
    p = _config_path(session)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(merged, indent=2), "utf-8")
    tmp.replace(p)
    return merged


def get_effects_manifest() -> dict[str, Any]:
    return {"effects": EFFECTS_MANIFEST,
            "color_presets": list(COLOR_PRESETS.keys())}


def set_panel_override(session: str, panel_id: str,
                       overrides: dict[str, Any] | None) -> dict[str, Any]:
    cfg = get_config(session)
    cfg.setdefault("panel_overrides", {})
    if overrides is None:
        cfg["panel_overrides"].pop(panel_id, None)
    else:
        cfg["panel_overrides"][panel_id] = overrides
    return save_config(session, cfg)


def upload_bgm(session: str, src_path: Path) -> dict[str, Any]:
    d = _session_dir(session)
    dest = d / ("bgm" + src_path.suffix.lower())
    shutil.copy2(src_path, dest)
    cfg = get_config(session)
    cfg["bgm_path"] = str(dest)
    save_config(session, cfg)
    return {"ok": True, "bgm_path": str(dest)}


def _build_engine_config(cfg: dict[str, Any],
                         panel_overrides_ignored: dict | None = None
                         ) -> Any:
    """Compose the real CinematicConfig from the stored session config.

    (Panel-level overrides currently only affect the preview helper; the
    engine classifies panels itself via narration/dialogue keywords.)
    """
    import dataclasses

    from cinematic_effects import DEFAULT_DYNAMIC, DEFAULT_SUBTLE

    preset = (DEFAULT_DYNAMIC
              if cfg.get("style", "dynamic") == "dynamic"
              else DEFAULT_SUBTLE)
    fields = {f.name for f in dataclasses.fields(preset)}

    updates: dict[str, Any] = {}
    # color preset -> grade fields
    color = COLOR_PRESETS.get(cfg.get("color_preset", "manhwa"))
    if color:
        updates.update(color)
    # frontend toggles
    updates["glitch_enabled"] = bool(cfg.get("glitch_transitions", True))
    updates["letterbox_enabled"] = bool(cfg.get("letterbox", False))
    updates["bgm_path"] = cfg.get("bgm_path") or None
    updates["bgm_volume"] = float(cfg.get("bgm_volume", 0.18))
    # global overrides: map frontend keys onto config fields, with direct
    # CinematicConfig field names accepted as-is
    for k, v in (cfg.get("global_overrides") or {}).items():
        if k in _OVERRIDE_FIELD_MAP:
            k = _OVERRIDE_FIELD_MAP[k]
        if k in fields:
            updates[k] = v
    # slider semantics: kb zoom range 0..0.2 -> end scale 1.0..1.2
    if "kb_zoom_end" in updates:
        with contextlib.suppress(TypeError, ValueError):
            updates["kb_zoom_end"] = 1.0 + max(
                0.0, min(0.2, float(updates["kb_zoom_end"])))
    # caption_style etc. never reach the engine
    safe = {k: v for k, v in updates.items() if k in fields}
    return dataclasses.replace(preset, **safe)


def apply_cinematic(session: str, job_store: Any) -> dict[str, Any]:
    """Kick off a background cinematic render for the session."""
    d = _session_dir(session)
    panels_json = d / "panels.json"
    if not panels_json.is_file():
        raise HTTPException(400, "run generation first; panels.json missing")
    cfg = get_config(session)
    job = job_store.create("cinematic_render", {"session": session})
    log.info("job=%s cinematic render started session=%s", job.id, session)

    def _run() -> None:
        try:
            from webapp.jobs import JobStatus
            job.status = JobStatus.RUNNING
            job.stage = "cinematic_render"
            job.touch()
            job.log("INFO", "cinematic render started", "cinematic_render")

            from cinematic_effects import make_cinematic_video
            engine_cfg = _build_engine_config(cfg)
            out_path = d / "cinematic_recap.mp4"
            # tmp first, atomic replace: a failed render must never
            # clobber a previous good cinematic_recap.mp4
            tmp = d / "cinematic_recap.partial.mp4"
            summary = make_cinematic_video(
                Path(panels_json), tmp,
                audio_dir=d / "audio", cfg=engine_cfg)
            tmp.replace(out_path)

            job.status = JobStatus.COMPLETED
            job.outputs = {"cinematic_mp4": str(out_path)}
            job.progress = 100
            job.stage = "done"
            job.finished_at = time.time()
            job.touch()
            job.log("INFO", f"cinematic render done out={out_path} "
                            f"panels={summary.get('panels')} "
                            f"dur={summary.get('duration_s')}s",
                    "cinematic_render")
        except Exception as exc:  # noqa: BLE001 - job-level failure surface
            import traceback
            job.fail(f"cinematic render failed: {exc}",
                     traceback.format_exc())
        finally:
            job_store.flush(job)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job.id, "status": "queued"}


def generate_panel_preview(session: str, panel_id: str) -> Path:
    """Short preview clip for one panel with the current effect settings."""
    d = _session_dir(session)
    panels_json = d / "panels.json"
    if not panels_json.is_file():
        raise HTTPException(400, "no panels.json found")

    from guided_cutter import CutArtifact
    try:
        artifact = CutArtifact.model_validate_json(
            panels_json.read_text("utf-8"))
    except Exception as e:
        raise HTTPException(500, f"cannot parse panels.json: {e}") from e

    panel = next((p for p in artifact.panels if p.id == panel_id), None)
    if panel is None:
        raise HTTPException(404, f"panel {panel_id} not found")

    cfg = get_config(session)
    out_path = d / f"preview_{panel_id}.mp4"

    from cinematic_effects import _build_panel_clip, _resolve_ffmpeg
    engine_cfg = _build_engine_config(cfg)

    img_path = d / panel.image_file
    if not img_path.is_file():
        raise HTTPException(404, f"panel image missing: {panel.image_file}")
    audio_path = d / "audio" / f"{panel_id}.mp3"
    duration = 3.5

    tmp = out_path.with_suffix(".partial.mp4")
    _build_panel_clip(
        panel={"id": panel.id, "narration": panel.narration,
               "dialogue": panel.dialogue,
               "output_width": panel.output_width,
               "output_height": panel.output_height,
               "panel_type": panel.panel_type},
        image_path=img_path,
        audio_path=audio_path if audio_path.is_file() else None,
        duration_s=duration,
        clip_out=tmp,
        speedlines_png=None,
        cfg=engine_cfg,
        ffmpeg=_resolve_ffmpeg(engine_cfg.ffmpeg_exe),
    )
    tmp.replace(out_path)
    return out_path
