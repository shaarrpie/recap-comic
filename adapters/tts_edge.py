# adapters/tts_edge.py
"""edge-tts adapter (default TTS).

Verified against edge-tts 7.x source, 2026-09-05:
- src/edge_tts/communicate.py: Communicate(text, voice=..., rate=...,
  volume=..., pitch=..., boundary="SentenceBoundary"|"WordBoundary",
  connect_timeout=10, receive_timeout=60). 7.2.8 DEFAULTS TO
  "SentenceBoundary" — word timing requires boundary="WordBoundary".
  .stream() yields dicts; types 'audio' (key 'data') and
  'WordBoundary'/'SentenceBoundary' (keys 'offset', 'duration', 'text');
  offset/duration are in 100-ns ticks (submaker.py divides by 10 to get
  microseconds, i.e. TICKS_PER_SECOND == 10_000_000).
- .save(audio_fname, metadata_fname) writes JSONL of boundary events.
- stream() may only be called once per Communicate instance.
- CLI: `edge-tts --list-voices` lists all voices.
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
                      out_path: Path) -> list[dict]:
    # boundary="WordBoundary" is REQUIRED: the 7.2.8 default is
    # "SentenceBoundary" (verified via inspect.signature on the installed
    # version). connect/receive timeouts are built in (10s / 60s).
    comm = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch,
                                boundary="WordBoundary")
    words: list[dict] = []
    audio_bytes = bytearray()
    async for msg in comm.stream():  # stream() is single-use
        if msg["type"] == "audio":
            audio_bytes.extend(msg["data"])
        elif msg["type"] == "WordBoundary":
            words.append({
                "start": msg["offset"] / TICKS_PER_SECOND,
                "end": (msg["offset"] + msg["duration"]) / TICKS_PER_SECOND,
                "text": msg["text"],
            })
    out_path.write_bytes(bytes(audio_bytes))  # disk I/O after the async loop
    return words


def synthesize_entry(entry: NarrationEntry, out_dir: Path, *, voice: str,
                     rate: str = "+0%", pitch: str = "+0Hz",
                     probe_duration: Callable[[Path], float],
                     retries: int = 3) -> AudioEntry | None:
    """One mp3 per entry, named by the entry's stable panel id.

    `probe_duration` must return a MEASURED duration (ffprobe); the timeline
    stage must never estimate speech rate. Returns None for empty text so
    silent panels simply get no audio file.
    """
    if not entry.text.strip():
        return None
    out_path = out_dir / f"{entry.id}.mp3"
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            words = asyncio.run(_synthesize(entry.text, voice, rate, pitch,
                                            out_path))
            return AudioEntry(entry_id=entry.id, path=out_path.name,
                              duration_seconds=probe_duration(out_path),
                              words=words)
        except Exception as exc:  # noqa: BLE001 - re-raised after retries
            last_exc = exc
            if attempt < retries:
                time.sleep(2 ** attempt)  # backoff: 2,4,8s (no event loop needed)
    # Persist the text so the user can retry manually without re-generating
    # the narration; this is the file the error message points at.
    retry_txt = out_path.with_suffix(".txt")
    retry_txt.write_text(entry.text, encoding="utf-8")
    raise RuntimeError(
        f"edge-tts failed for entry {entry.id} after {retries} attempts "
        f"(text saved for retry at {retry_txt}): {last_exc}"
    ) from last_exc
