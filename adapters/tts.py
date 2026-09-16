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
        if rate != "+0%" or pitch != "+0Hz":
            log.warning("kokoro ignores edge-tts rate/pitch options "
                        "(rate=%r pitch=%r); using defaults", rate, pitch)
        if retries != 3:
            log.warning("kokoro performs a single synthesis attempt; "
                        "ignoring retries=%r", retries)
        if probe_duration is None:
            return None, RuntimeError(
                "kokoro requires a probe_duration callable to measure audio")
        safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", entry.id).strip("_") or "panel"
        if safe_name != entry.id:
            log.warning("sanitised kokoro output name %r -> %r",
                        entry.id, safe_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{safe_name}.wav"
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
