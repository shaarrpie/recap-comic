# adapters/tts.py
"""TTS provider dispatcher.

Providers:
  - edge   : adapters.tts_edge (cloud, default)
  - kokoro : adapters.tts_kokoro (offline, CPU)
  - none   : silent, no audio generated

The dispatcher selects the provider, passes the correct kwargs, and
returns a uniform (AudioEntry | None, error | None) tuple.
"""
from __future__ import annotations

from pathlib import Path

from .schemas import AudioEntry, NarrationEntry


def synthesize_entry(entry: NarrationEntry, out_dir: Path, *,
                     provider: str = "edge",
                     voice: str = "en-US-AriaNeural",
                     rate: str = "+0%",
                     pitch: str = "+0Hz",
                     speed: float = 1.0,
                     probe_duration=None,
                     kokoro_model_path: Path | None = None,
                     kokoro_voices_path: Path | None = None,
                     retries: int = 3) -> tuple[AudioEntry | None, Exception | None]:
    """Synthesize one narration entry with the selected provider.

    Returns (AudioEntry, None) on success, (None, error) on failure.
    """
    if provider == "none" or not entry.text.strip():
        return None, None

    if provider == "edge":
        try:
            from .tts_edge import synthesize_entry as edge_synth
        except ImportError as exc:
            return None, RuntimeError(
                f"edge-tts not installed: {exc}. pip install edge-tts "
                f"or use --tts kokoro / --tts none")
        try:
            result = edge_synth(
                entry, out_dir, voice=voice, rate=rate, pitch=pitch,
                probe_duration=probe_duration, retries=retries)
            return result, None
        except Exception as exc:
            return None, exc

    if provider == "kokoro":
        if kokoro_model_path is None or kokoro_voices_path is None:
            return None, RuntimeError(
                "kokoro requires --kokoro-model-path and "
                "--kokoro-voices-path")
        try:
            from .tts_kokoro import synthesize as kokoro_synth
        except ImportError as exc:
            return None, RuntimeError(
                f"kokoro-onnx not installed: {exc}. pip install kokoro-onnx")
        out_path = out_dir / f"{entry.id}.wav"
        try:
            dur = kokoro_synth(
                entry.text, out_path,
                model_path=kokoro_model_path,
                voices_path=kokoro_voices_path,
                voice=voice, speed=speed,
                probe_duration=probe_duration)
            return AudioEntry(entry_id=entry.id, path=out_path.name,
                              duration_seconds=dur, words=[]), None
        except Exception as exc:
            return None, exc

    return None, RuntimeError(f"unknown TTS provider: {provider}")
