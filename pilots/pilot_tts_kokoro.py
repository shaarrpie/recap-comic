# pilots/pilot_tts_kokoro.py
"""Pilot for the offline Kokoro adapter. NOT run in this session (requires a
~300MB manual model download; see adapters/tts_kokoro.py docstring).
Run after downloading kokoro-v1.0.onnx + voices-v1.0.bin from the
kokoro-onnx README links.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.tts_kokoro import synthesize


def probe(path: Path) -> float:
    import soundfile as sf
    return len(sf.read(path)[0]) / sf.info(path).samplerate


if __name__ == "__main__":
    model, voices = Path(sys.argv[1]), Path(sys.argv[2])
    out = Path("pilot_kokoro.wav")
    dur = synthesize("Kokoro offline pilot.", out, model_path=model,
                     voices_path=voices, probe_duration=probe)
    print(f"wav duration: {dur:.2f}s")
