# webapp/voice_api.py
"""Narrator / voice studio support.

Per-session voice configuration is persisted to `<session>/voice.json` and
merged (not clobbered) when a project is reopened.  Edge TTS exposes a live
voice catalogue, so `list_voices` pulls it on demand.  Voice *previews* are
synthesised to the session under `.previews/ui-<hash>.mp3` and content-hashed,
so re-previewing wins a cache hit and previews never accumulate.

Rate/Pitch are native edge-tts values ("+0%", "+0Hz").  `speed` is a reserved
multiplier for providers that expose it; edge maps speed onto rate, so the
studio folds it into `rate` and does not pretend edge has an independent speed.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
from pathlib import Path

import edge_tts
from fastapi import HTTPException

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.environ.get("RECAP_OUTPUT_DIR") or BASE_DIR / "webapp_output")
PROVIDERS = ("edge", "none")

DEFAULT_VOICE = {
    "provider": "edge",
    "voice": "en-US-AriaNeural",
    "rate": 0,       # edge-tts rate offset in % (e.g. +8)
    "pitch": 0,      # edge-tts pitch offset in Hz
    "speed": 1.0,    # reserved multiplier (edge folds this into rate)
    "style": "recap",
}

SAMPLE_TEXT = ("The protagonist suddenly realizes something is wrong. "
               "The city will never be the same again.")


def _session_dir(session: str) -> Path:
    import re
    if not re.match(r"^[0-9a-f]{12}$", session):
        raise HTTPException(400, "invalid session id")
    d = (OUTPUT_DIR / session).resolve()
    base = OUTPUT_DIR.resolve()
    if base not in d.parents and d != base:
        raise HTTPException(400, "invalid session path")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _coerce_voice_value(k: str, v) -> object:
    """Validate/coerce one voice config value. A string rate/pitch/speed
    in voice.json or a request body must not raise a raw 500 deep in
    _rate_args — coerce valid numeric strings, 400 anything else."""
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
    p = _session_dir(session) / "voice.json"
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


async def list_voices(provider: str = "edge") -> dict:
    if provider != "edge":
        raise HTTPException(400, f"provider {provider!r} exposes no catalogue")
    try:
        raw = await edge_tts.list_voices()
    except Exception as exc:  # network / auth
        raise HTTPException(503, f"could not fetch voice list: {exc}") from exc
    voices = [{
        "id": v["Name"],
        "shortname": v.get("ShortName", ""),
        "gender": v.get("Gender", ""),
        "locale": v.get("Locale", ""),
        "language": v.get("Language", ""),
    } for v in raw]
    voices.sort(key=lambda v: (v["language"], v["shortname"]))
    return {"provider": provider, "count": len(voices), "voices": voices}


def _rate_args(cfg: dict) -> dict:
    """Map the studio config onto edge-tts rate/pitch strings. Values come
    from put_voice-validated storage, but preview bodies pass cfg directly
    — coerce here too so a bad type can never raise a raw 500."""
    try:
        rate = int(float(cfg.get("rate", 0) or 0))
        speed = float(cfg.get("speed", 1.0) or 1.0)
        pitch = int(float(cfg.get("pitch", 0) or 0))
    except (TypeError, ValueError):
        raise HTTPException(400, "rate/pitch/speed must be numbers") from None
    rate = max(-100, min(100, rate))
    speed = max(0.5, min(2.0, speed))
    pitch = max(-100, min(100, pitch))
    # edge has a single speech-rate control; fold the speed multiplier in
    # as a percentage offset so speed=1.0 is neutral (rate*speed would be
    # stuck at 0% whenever rate is 0).
    combined = rate + round((speed - 1.0) * 100)
    return {"rate": f"+{combined}%" if combined >= 0 else f"{combined}%",
            "pitch": f"+{pitch}Hz" if pitch >= 0 else f"{pitch}Hz"}


async def _preview_mp3(session: str, cfg: dict, text: str) -> Path:
    d = _session_dir(session)
    pre = d / ".previews"
    pre.mkdir(exist_ok=True)
    key = hashlib.sha1(
        f"{cfg.get('voice')}|{text}|{cfg.get('rate')}|{cfg.get('pitch')}|{cfg.get('speed')}".encode()
    ).hexdigest()[:16]
    out = pre / f"ui-{key}.mp3"
    if out.is_file():
        return out
    rate_cfg = _rate_args(cfg)
    voice = cfg.get("voice") or DEFAULT_VOICE["voice"]
    try:
        text = text.strip() or SAMPLE_TEXT
        import edge_tts as et
        comm = et.Communicate(text, voice=voice,
                              rate=rate_cfg["rate"], pitch=rate_cfg["pitch"],
                              boundary="WordBoundary")
        audio = bytearray()
        async def _pull():
            async for msg in comm.stream():
                if msg["type"] == "audio":
                    audio.extend(msg["data"])
        await asyncio.wait_for(_pull(), timeout=45)
        if not audio:
            raise RuntimeError("no audio for this voice/text")
        out.with_suffix(".tmp").write_bytes(bytes(audio))
        out.with_suffix(".tmp").replace(out)
    except Exception as exc:
        raise HTTPException(502, f"preview synthesis failed: {exc}") from exc
    # prune old previews (keep newest 30 by mtime)
    files = sorted(pre.glob("ui-*.mp3"), key=lambda p: p.stat().st_mtime)
    for old in files[:-30]:
        with contextlib.suppress(OSError):
            old.unlink()
    return out


async def make_preview(session: str, cfg: dict, text: str) -> dict:
    if (cfg.get("provider") or "edge") == "none":
        raise HTTPException(400, "no voice preview for provider='none'")
    pre = await _preview_mp3(session, cfg, text)
    return {"session": session, "url": f"/api/jobs/{session}/files/.previews/{pre.name}",
            "voice": cfg.get("voice"), "rate_cfg": _rate_args(cfg)}
