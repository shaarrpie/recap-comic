# adapters/tts_kokoro.py
"""Kokoro TTS adapter (offline speech synthesis).

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
from typing import Any

_KOKORO_CACHE: dict[tuple[str, str], Any] = {}


def resolve_model_files(model_path: Path | None = None,
                        voices_path: Path | None = None
                        ) -> tuple[Path, Path]:
    """Locate the Kokoro weights, or raise a RuntimeError telling the user
    exactly how to get them.

    Order: explicit args -> KOKORO_MODEL_PATH / KOKORO_VOICES_PATH env vars
    -> ./models/kokoro-v1.0.onnx + ./models/voices-v1.0.bin (repo root).
    The weights (~300MB, or ~80MB quantized) are fetched once by the user
    per the README ("Offline speech") links; they are never downloaded
    implicitly, so an offline machine fails here with instructions instead
    of hanging on a network fetch.
    """
    import os

    root = Path(__file__).resolve().parent.parent
    model = Path(model_path) if model_path else None
    voices = Path(voices_path) if voices_path else None
    if model is None:
        env = os.environ.get("KOKORO_MODEL_PATH", "").strip()
        model = Path(env) if env else root / "models" / "kokoro-v1.0.onnx"
    if voices is None:
        env = os.environ.get("KOKORO_VOICES_PATH", "").strip()
        voices = Path(env) if env else root / "models" / "voices-v1.0.bin"
    missing = [str(p) for p in (model, voices) if not p.is_file()]
    if missing:
        raise RuntimeError(
            "Kokoro voice weights not found: "
            + ", ".join(missing)
            + ". Fetch kokoro-v1.0.onnx + voices-v1.0.bin once (see README "
            "'Offline speech' for the links), place them in ./models/, or "
            "point KOKORO_MODEL_PATH / KOKORO_VOICES_PATH at them — "
            "or re-run with --tts none for a silent video.")
    return model, voices


def synthesize(text: str, out_path: Path, *, model_path: Path,
               voices_path: Path, voice: str = "af_heart", speed: float = 1.0,
               probe_duration: Callable[[Path], float]) -> float:
    import soundfile as sf  # optional dependency
    from kokoro_onnx import Kokoro

    key = (str(model_path), str(voices_path))
    if key not in _KOKORO_CACHE:
        _KOKORO_CACHE[key] = Kokoro(str(model_path), str(voices_path))
    kokoro = _KOKORO_CACHE[key]
    samples, sample_rate = kokoro.create(text, voice=voice, speed=speed,
                                         lang="en-us")  # type: ignore[call-arg]
    sf.write(out_path, samples, sample_rate)
    return probe_duration(out_path)
