# adapters/tts_edge.py
"""edge-tts adapter (cloud TTS via Microsoft Edge).

edge-tts 7.x is installed; this is the only cloud provider the dispatcher
needs.  Verified against edge-tts 7.2.8:
  - Communicate(text, voice=..., rate=..., pitch=..., boundary=...)
    boundary="WordBoundary" is REQUIRED to get per-word timing (the default
    "SentenceBoundary" yields no word events, so captions would fall back to
    one cue per panel).
  - .stream() yields dicts with type 'audio' (key 'data') and
    'WordBoundary' (keys 'offset', 'duration', 'text'); offset/duration are
    in 100-ns ticks (TICKS_PER_SECOND == 10_000_000).
  - stream() may only be called ONCE per Communicate instance.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

import edge_tts

from .schemas import AudioEntry, NarrationEntry

TICKS_PER_SECOND = 10_000_000  # 100-ns intervals; edge_tts.constants.TICKS_PER_SECOND


async def _synthesize(text: str, voice: str, rate: str, pitch: str,
                      out_path: Path) -> tuple[bytes, list[dict]]:
    comm = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch,
                                boundary="WordBoundary")
    words: list[dict] = []
    audio_bytes = bytearray()
    async for msg in comm.stream():
        if msg["type"] == "audio":
            audio_bytes.extend(msg["data"])
        elif msg["type"] == "WordBoundary":
            words.append({
                "start": msg["offset"] / TICKS_PER_SECOND,
                "end": (msg["offset"] + msg["duration"]) / TICKS_PER_SECOND,
                "text": msg["text"],
            })
    return bytes(audio_bytes), words


def synthesize_entry(entry: NarrationEntry, out_dir: Path, *, voice: str,
                     rate: str = "+0%", pitch: str = "+0Hz",
                     probe_duration: Callable[[Path], float],
                     retries: int = 3) -> AudioEntry | None:
    """Synthesize one narration entry with edge-tts.

    Returns an AudioEntry on success, None on empty text.  Raises after
    `retries` attempts (text saved to <id>.txt for manual retry).
    """
    if not entry.text.strip():
        return None
    out_path = out_dir / f"{entry.id}.mp3"
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            audio_bytes, words = asyncio.run(
                _synthesize(entry.text, voice, rate, pitch, out_path))
            if not audio_bytes:
                raise RuntimeError(
                    f"edge-tts returned empty audio for entry {entry.id}")
            out_path.write_bytes(audio_bytes)
            return AudioEntry(entry_id=entry.id, path=out_path.name,
                              duration_seconds=probe_duration(out_path),
                              words=words)
        except Exception as exc:  # noqa: BLE001 - re-raised after retries
            last_exc = exc
            if attempt < retries:
                time.sleep(2 ** attempt)
    retry_txt = out_path.with_suffix(".txt")
    retry_txt.parent.mkdir(parents=True, exist_ok=True)
    retry_txt.write_text(entry.text, encoding="utf-8")
    raise RuntimeError(
        f"edge-tts failed for entry {entry.id} after {retries} attempts "
        f"(text saved for retry at {retry_txt}): {last_exc}"
    ) from last_exc


async def synthesize_entry_async(entry: NarrationEntry, out_dir: Path, *,
                                 voice: str, rate: str = "+0%",
                                 pitch: str = "+0Hz",
                                 probe_duration: Callable[[Path], float],
                                 retries: int = 3) -> AudioEntry | None:
    """Async variant for use inside an existing event loop."""
    if not entry.text.strip():
        return None
    out_path = out_dir / f"{entry.id}.mp3"
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            audio_bytes, words = await _synthesize(
                entry.text, voice, rate, pitch, out_path)
            if not audio_bytes:
                raise RuntimeError(
                    f"edge-tts returned empty audio for entry {entry.id}")
            out_path.write_bytes(audio_bytes)
            return AudioEntry(entry_id=entry.id, path=out_path.name,
                              duration_seconds=probe_duration(out_path),
                              words=words)
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                await asyncio.sleep(2 ** attempt)
    retry_txt = out_path.with_suffix(".txt")
    retry_txt.parent.mkdir(parents=True, exist_ok=True)
    retry_txt.write_text(entry.text, encoding="utf-8")
    raise RuntimeError(
        f"edge-tts failed for entry {entry.id} after {retries} attempts "
        f"(text saved for retry at {retry_txt}): {last_exc}"
    ) from last_exc