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
import hashlib
import json
import time
from pathlib import Path

from fastapi import HTTPException

import edge_tts

from .tts_helpers import synth_one

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "webapp_output"
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


def get_voice(session: str) -> dict:
    p = _session_dir(session) / "voice.json"
    cfg = dict(DEFAULT_VOICE)
    if p.is_file():
        try:
            cfg.update({k: v for k, v in json.loads(p.read_text("utf-8")).items()
                        if k in cfg})
        except Exception:
            pass
    return cfg


def put_voice(session: str, cfg: dict) -> dict:
    cur = get_voice(session)
    for k in DEFAULT_VOICE:
        if k in cfg:
            cur[k] = cfg[k]
    p = _session_dir(session) / "voice.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, indent=2), "utf-8")
    tmp.replace(p)
    return cur


def list_voices(provider: str = "edge") -> dict:
    if provider != "edge":
        raise HTTPException(400, f"provider {provider!r} exposes no catalogue")
    try:
        raw = asyncio.run(edge_tts.list_voices())
    except Exception as exc:  # network / auth
        raise HTTPException(503, f"could not fetch voice list: {exc}")
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
    """Map the studio config onto edge-tts rate/pitch strings."""
    rate = int(cfg.get("rate", 0) or 0)
    speed = float(cfg.get("speed", 1.0) or 1.0)
    # edge has a single speech-rate control; fold the speed multiplier in.
    combined = int(round(rate * speed))
    pitch = int(cfg.get("pitch", 0) or 0)
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
        raise HTTPException(502, f"preview synthesis failed: {exc}")
    # prune old previews (keep newest 30 by mtime)
    files = sorted(pre.glob("ui-*.mp3"), key=lambda p: p.stat().st_mtime)
    for old in files[:-30]:
        try:
            old.unlink()
        except OSError:
            pass
    return out


async def make_preview(session: str, cfg: dict, text: str) -> dict:
    if (cfg.get("provider") or "edge") == "none":
        raise HTTPException(400, "no voice preview for provider='none'")
    pre = await _preview_mp3(session, cfg, text)
    return {"session": session, "url": f"/api/jobs/{session}/files/.previews/{pre.name}",
            "voice": cfg.get("voice"), "rate_cfg": _rate_args(cfg)}