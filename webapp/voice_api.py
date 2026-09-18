# webapp/voice_api.py
"""Narrator / voice studio support (local Kokoro speech, offline).

Per-session voice configuration is persisted to `<session>/voice.json` and
merged (not clobbered) when a project is reopened. Kokoro voices are a
fixed local catalogue (no network), so `list_voices` serves a static list.
Voice *previews* are synthesised to the session under `.previews/ui-<hash>.mp3`
(via webapp.tts_helpers: Kokoro WAV transcoded to MP3) and content-hashed,
so re-previewing wins a cache hit and previews never accumulate.

Kokoro takes a `speed` multiplier (0.5-2.0) and a voice id (default
af_heart). The legacy `rate`/`pitch` knobs are still accepted in stored
configs for backward compatibility but are inert.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path

from fastapi import HTTPException

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.environ.get("RECAP_OUTPUT_DIR") or BASE_DIR / "webapp_output")
PROVIDERS = ("kokoro", "none")

DEFAULT_VOICE = {
    "provider": "kokoro",
    "voice": "af_heart",
    "rate": 0,       # legacy edge-tts knob: accepted, ignored by Kokoro
    "pitch": 0,      # legacy edge-tts knob: accepted, ignored by Kokoro
    "speed": 1.0,    # kokoro speed multiplier
    "style": "recap",
}

# Static Kokoro voice catalogue (Kokoro-82M voice pack: American/British
# female/male). Served without any network call.
KOKORO_VOICES = [
    {"id": "af_heart", "shortname": "af_heart", "gender": "Female",
     "locale": "en-US", "language": "en"},
    {"id": "af_bella", "shortname": "af_bella", "gender": "Female",
     "locale": "en-US", "language": "en"},
    {"id": "af_nicole", "shortname": "af_nicole", "gender": "Female",
     "locale": "en-US", "language": "en"},
    {"id": "af_sarah", "shortname": "af_sarah", "gender": "Female",
     "locale": "en-US", "language": "en"},
    {"id": "af_sky", "shortname": "af_sky", "gender": "Female",
     "locale": "en-US", "language": "en"},
    {"id": "am_adam", "shortname": "am_adam", "gender": "Male",
     "locale": "en-US", "language": "en"},
    {"id": "am_michael", "shortname": "am_michael", "gender": "Male",
     "locale": "en-US", "language": "en"},
    {"id": "bf_emma", "shortname": "bf_emma", "gender": "Female",
     "locale": "en-GB", "language": "en"},
    {"id": "bf_isabella", "shortname": "bf_isabella", "gender": "Female",
     "locale": "en-GB", "language": "en"},
    {"id": "bm_george", "shortname": "bm_george", "gender": "Male",
     "locale": "en-GB", "language": "en"},
    {"id": "bm_lewis", "shortname": "bm_lewis", "gender": "Male",
     "locale": "en-GB", "language": "en"},
]

SAMPLE_TEXT = ("The protagonist suddenly realizes something is wrong. "
               "The city will never be the same again.")


def _session_dir(session: str, *, create: bool = False) -> Path:
    import re
    if not re.match(r"^[0-9a-f]{12}$", session):
        raise HTTPException(400, "invalid session id")
    d = (OUTPUT_DIR / session).resolve()
    base = OUTPUT_DIR.resolve()
    if base not in d.parents and d != base:
        raise HTTPException(400, "invalid session path")
    # Read paths must NOT materialize directories: probing
    # /api/voice/<hex-id> used to create a phantom empty session that then
    # showed up in /api/projects (see panel_api._session_dir).
    if create or d.is_dir():
        d.mkdir(parents=True, exist_ok=True)
    return d


def _coerce_voice_value(k: str, v) -> object:
    """Validate/coerce one voice config value. A string rate/pitch/speed
    in voice.json or a request body must not raise a raw 500 deep in
    synthesis — coerce valid numeric strings, 400 anything else."""
    if k == "provider":
        if v not in PROVIDERS:
            raise HTTPException(400, f"unknown provider {v!r}")
        return v
    if k == "voice":
        if not isinstance(v, str) or not v.strip():
            raise HTTPException(400, "voice must be a non-empty string")
        return v
    if k == "style":
        if not isinstance(v, str) or not v.strip():
            raise HTTPException(400, "style must be a non-empty string")
        return v
    # numeric knobs
    if isinstance(v, bool):
        raise HTTPException(400, f"{k} must be a number")
    if isinstance(v, (int, float)):
        num = v
    elif isinstance(v, str):
        try:
            num = float(v)
        except ValueError:
            raise HTTPException(
                400, f"{k} must be a number, got {v!r}") from None
    else:
        raise HTTPException(400, f"{k} must be a number")
    if k in ("rate", "pitch"):
        if not (-100 <= num <= 100):
            raise HTTPException(400, f"{k} must be within [-100, 100]")
        return int(num)
    if k == "speed":
        if not (0.5 <= num <= 2.0):
            raise HTTPException(400, "speed must be within [0.5, 2.0]")
        return round(float(num), 2)
    return v


def get_voice(session: str) -> dict:
    p = _session_dir(session) / "voice.json"
    cfg = dict(DEFAULT_VOICE)
    if p.is_file():
        with contextlib.suppress(Exception):
            loaded = json.loads(p.read_text("utf-8"))
            # tolerate legacy string values but never crash on them
            for k in DEFAULT_VOICE:
                if k in loaded:
                    with contextlib.suppress(HTTPException):
                        cfg[k] = _coerce_voice_value(k, loaded[k])
    # Migrate legacy cloud providers to local Kokoro on read (never persist
    # here; put_voice normalizes on the next save).
    if cfg.get("provider") not in PROVIDERS:
        cfg["provider"] = "kokoro"
    if not (cfg.get("voice") or "").strip():
        cfg["voice"] = DEFAULT_VOICE["voice"]
    return cfg


def put_voice(session: str, cfg: dict) -> dict:
    cur = get_voice(session)
    changed = False
    for k in DEFAULT_VOICE:
        if k in cfg and cfg[k] != cur.get(k):
            changed = True
    for k in DEFAULT_VOICE:
        if k in cfg:
            cur[k] = _coerce_voice_value(k, cfg[k])
    p = _session_dir(session, create=True) / "voice.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, indent=2), "utf-8")
    tmp.replace(p)
    if changed:
        # Step-by-Step ledger: a voice change only invalidates the render.
        try:
            from . import checkpoint as _cp
            _cp.apply_edit_invalidation(session, "voice.json")
        except Exception:
            pass
    return cur


async def list_voices(provider: str = "kokoro") -> dict:
    if provider != "kokoro":
        raise HTTPException(400, f"provider {provider!r} exposes no catalogue")
    voices = sorted(KOKORO_VOICES,
                    key=lambda v: (v["language"], v["shortname"]))
    return {"provider": provider, "count": len(voices), "voices": voices}


def _speed_value(cfg: dict) -> float:
    """Kokoro speed multiplier from a studio config (clamped)."""
    try:
        speed = float(cfg.get("speed", 1.0) or 1.0)
    except (TypeError, ValueError):
        raise HTTPException(400, "speed must be a number") from None
    return max(0.5, min(2.0, round(speed, 2)))


async def _preview_mp3(session: str, cfg: dict, text: str) -> Path:
    from . import tts_helpers

    d = _session_dir(session, create=True)
    pre = d / ".previews"
    pre.mkdir(exist_ok=True)
    voice = (cfg.get("voice") or DEFAULT_VOICE["voice"]).strip()
    speed = _speed_value(cfg)
    text = (text or "").strip() or SAMPLE_TEXT
    key = hashlib.sha1(
        f"kokoro|{voice}|{speed}|{text}".encode()
    ).hexdigest()[:16]
    out = pre / f"ui-{key}.mp3"
    if out.is_file():
        return out
    try:
        # tts_helpers resolves Kokoro weights (or raises with fetch
        # instructions).
        await tts_helpers.synth_one(text, voice, out, speed=speed,
                                    timeout_s=180)
    except Exception as exc:
        raise HTTPException(502, f"preview synthesis failed: {exc}") from exc
    # prune old previews (keep newest 30 by mtime)
    files = sorted(pre.glob("ui-*.mp3"), key=lambda p: p.stat().st_mtime)
    for old in files[:-30]:
        with contextlib.suppress(OSError):
            old.unlink()
    return out


async def make_preview(session: str, cfg: dict, text: str) -> dict:
    if (cfg.get("provider") or "kokoro") == "none":
        raise HTTPException(400, "no voice preview for provider='none'")
    pre = await _preview_mp3(session, cfg, text)
    return {"session": session, "url": f"/api/jobs/{session}/files/.previews/{pre.name}",
            "voice": cfg.get("voice"), "speed": _speed_value(cfg)}
