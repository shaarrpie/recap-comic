# adapters/tts_kokoro.py
"""Kokoro TTS adapter (offline alternative to edge-tts).

kokoro-onnx (thewh1teagle/kokoro-onnx, MIT code + Apache-2.0 model, 2.7k
stars) runs the Kokoro-82M model fully offline on CPU via onnxruntime.
Weights (~300MB, or ~80MB quantized) are NOT downloaded at import time: the
user manually fetches kokoro-v1.0.onnx and voices-v1.0.bin per the README
(opened 2026-09-05). API below is quoted from the repo README:
  from kokoro_onnx import Kokoro; kokoro = Kokoro(model_path, voices_path)
  samples, sample_rate = kokoro.create(text, voice=..., speed=..., lang=...)

# UNVERIFIED AGAINST kokoro-onnx==0.6.1 — confirm create() signature in the
# installed version before relying. Pilot: pilots/pilot_tts_kokoro.py
# (NOT executed in this session: requires the ~300MB model download).
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def synthesize(text: str, out_path: Path, *, model_path: Path,
               voices_path: Path, voice: str = "af_heart", speed: float = 1.0,
               probe_duration: Callable[[Path], float]) -> float:
    import soundfile as sf  # optional dependency
    from kokoro_onnx import Kokoro

    kokoro = Kokoro(str(model_path), str(voices_path))
    samples, sample_rate = kokoro.create(text, voice=voice, speed=speed,
                                         lang="en-us")  # type: ignore[call-arg]
    sf.write(out_path, samples, sample_rate)
    return probe_duration(out_path)
