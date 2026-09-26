"""Local TTS helper for voice previews (Kokoro, offline).

Synthesizes `text` with the local Kokoro weights to WAV, then transcodes
to MP3 with the bundled imageio-ffmpeg binary so preview caches stay .mp3
(the routes and tests gate on that extension). No cloud calls: the only
network AI in this project is the Agnes gateway.

Signature is kept stable (text, voice, out, rate, pitch, timeout_s):
rate/pitch are accepted and ignored (Kokoro voices take `speed`, which
preview callers leave at the default). Atomic write: an interrupted
synthesis never leaves a partial mp3 behind.
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _synth_wav_to_mp3(text: str, voice: str, out: Path,
                      speed: float = 1.0) -> None:
    from adapters.tts_kokoro import resolve_model_files, synthesize

    model_path, voices_path = resolve_model_files()
    tmp_wav = out.with_suffix(".wav.tmp")
    tmp_mp3 = out.with_suffix(".mp3.tmp")
    try:
        synthesize(text, tmp_wav, model_path=model_path,
                   voices_path=voices_path, voice=voice, speed=speed,
                   probe_duration=lambda _p: 0.0)
        cmd = [_ffmpeg_exe(), "-y", "-nostdin", "-i", str(tmp_wav),
               "-codec:a", "libmp3lame", "-b:a", "128k", str(tmp_mp3)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False,
                              timeout=120)
        if proc.returncode != 0 or not tmp_mp3.is_file():
            tail = "\n".join((proc.stderr or "").splitlines()[-5:])
            raise RuntimeError(f"mp3 transcode failed: {tail}")
        if tmp_mp3.stat().st_size == 0:
            raise RuntimeError("tts returned no audio for this text")
        tmp_mp3.replace(out)
    finally:
        for tmp in (tmp_wav, tmp_mp3):
            try:
                if tmp.is_file() and tmp.resolve() != out.resolve():
                    tmp.unlink()
            except OSError:
                pass


async def synth_one(text: str, voice: str, out: Path,
                    rate: str = "", pitch: str = "",
                    speed: float = 1.0,
                    timeout_s: float = 120) -> None:
    # rate/pitch are legacy knobs: accepted for caller compat, ignored by
    # Kokoro (which voices text with `speed` instead).
    text = (text or "").strip()
    if not text:
        raise RuntimeError("tts received empty text")
    await asyncio.wait_for(
        asyncio.to_thread(_synth_wav_to_mp3, text, voice, out, speed),
        timeout=timeout_s)
    if not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError("tts returned no audio for this text")


async def synth_one_edge(text: str, voice: str, out: Path,
                         rate: str = "+0%", pitch: str = "+0Hz",
                         timeout_s: float = 180) -> None:
    """Synthesize one preview clip with cloud edge-tts (network call).

    Only reached when the user explicitly selects provider='edge'. Writes an
    mp3 atomically and raises on empty audio so voice_api surfaces a 502
    rather than caching a silent clip (edge-tts fails silently on a bad voice
    id). Unlike Kokoro, edge honours ``rate``/``pitch``.
    """
    text = (text or "").strip()
    if not text:
        raise RuntimeError("tts received empty text")
    import edge_tts

    tmp = out.with_suffix(".mp3.tmp")

    async def _run() -> None:
        comm = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch)
        await comm.save(str(tmp))
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise RuntimeError("edge-tts returned no audio for this text")
        tmp.replace(out)

    try:
        await asyncio.wait_for(_run(), timeout=timeout_s)
    finally:
        try:
            if tmp.is_file():
                tmp.unlink()
        except OSError:
            pass
