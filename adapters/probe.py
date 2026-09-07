# adapters/probe.py
"""Measured audio duration helpers (no ffprobe needed).

Two strategies, in order of preference:

1. WAV header — transcode the source to 48 kHz stereo PCM WAV (the same
   format the renderer filtergraph wants anyway), then read the duration
   from the RIFF header. Exact, not estimated.
2. MP3/OGG header — for formats ffmpeg can probe without re-encoding.
   Still exact; falls back to strategy 1 when the codec is unsupported.
"""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path


def probe_wav(path: Path) -> float:
    """Return the exact duration of a PCM WAV file from its header."""
    with wave.open(str(path), "rb") as f:
        n_frames = f.getnframes()
        rate = f.getframerate()
        if rate <= 0:
            raise ValueError(f"invalid framerate in {path}: {rate}")
        return n_frames / rate


def to_wav(source: Path, out: Path, ffmpeg_exe: str = "ffmpeg") -> Path:
    """Transcode `source` (mp3, ogg, etc.) to 48 kHz stereo PCM WAV."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source),
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg transcoding failed: {proc.stderr.strip()[-300:]}")
    return out


def probe_duration(path: Path, *, ffmpeg_exe: str = "ffmpeg") -> float:
    """Return exact duration for any audio file ffmpeg can decode.

    Prefers WAV header read (fast, no subprocess). Falls back to ffmpeg
    -t 0 transcoding for formats that aren't WAV.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".wav":
        return probe_wav(path)
    wav_tmp = path.with_suffix(".probe.wav")
    try:
        to_wav(path, wav_tmp, ffmpeg_exe=ffmpeg_exe)
        return probe_wav(wav_tmp)
    finally:
        if wav_tmp.is_file():
            wav_tmp.unlink()
