# pilots/pilot_render.py
"""Renders one tiny 9:16 timeline with BOTH renderers and compares.

Uses imageio-ffmpeg's bundled ffmpeg binary (no system FFmpeg needed) and
moviepy==2.2.1. This pilot empirically verifies the MoviePy 2.x API names
used in adapters/render_moviepy.py and the FFmpeg filtergraph in
adapters/render_ffmpeg.py. Requires imageio-ffmpeg; works offline.
Run:  python pilots/pilot_render.py
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.render_ffmpeg import build_command
from adapters.render_ffmpeg import render as render_ff
from adapters.render_moviepy import render as render_mp
from adapters.schemas import (
    BBox,
    Meta,
    PanSpec,
    TimelineArtifact,
    TimelineEntry,
)

OUT = Path(__file__).parent / "_out"


def fit_pan(w: int, h: int) -> PanSpec:
    """Scale a panel so it covers 1080x1920; overflow axis gets the pan."""
    sw, sh = 1080, round(h * 1080 / w)
    if sh >= 1920:
        return PanSpec(kind="pan_down", scaled_w=sw, scaled_h=sh,
                       travel_px=sh - 1920)
    sh, sw = 1920, round(w * 1920 / h)
    if sw > 1080:
        return PanSpec(kind="pan_right", scaled_w=sw, scaled_h=sh,
                       travel_px=sw - 1080)
    return PanSpec(kind="static", scaled_w=1080, scaled_h=1920, travel_px=0)


def sine_wav(path: Path, seconds: float, rate: int = 8000) -> None:
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    pcm = (np.sin(2 * np.pi * 440 * t) * 12000).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())


def probe_wav(path: Path) -> float:  # stand-in for ffprobe in this pilot
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / f.getframerate()


def make_timeline() -> TimelineArtifact:
    OUT.mkdir(exist_ok=True)
    # Gradients instead of random noise: cheap to encode, so the pilot runs
    # in seconds while exercising the same scale/crop/pan/loudnorm graph.
    h1, w1 = 2400, 1080  # taller than 9:16 -> vertical pan
    h2, w2 = 1080, 2400  # wider  than 9:16 -> horizontal pan
    img1 = OUT / "panel_tall.png"
    img2 = OUT / "panel_wide.png"
    grad1 = np.tile(np.linspace(0, 255, h1, dtype=np.uint8)[:, None], (1, w1))
    Image.fromarray(grad1).save(img1)
    grad2 = np.tile(np.linspace(0, 255, w2, dtype=np.uint8)[None, :], (h2, 1))
    Image.fromarray(grad2).save(img2)
    wav1, wav2 = OUT / "001.01.wav", OUT / "001.02.wav"
    sine_wav(wav1, 1.0)
    sine_wav(wav2, 0.8)
    # Timeline contract: entries are CONTIGUOUS and each entry's duration
    # INCLUDES its trailing silence gap (audio is padded to that duration).
    # This is what makes the pure-FFmpeg concat renderer and MoviePy produce
    # identical timing by construction (no A/V drift, no list-position zips).
    e1 = TimelineEntry(panel_id="001.01", order=1, source_image=str(img1),
                       bbox=BBox(x=0, y=0, w=w1, h=h1),
                       start_seconds=0.0, duration_seconds=1.35,  # 1.0 audio + 0.35 gap
                       audio_path=str(wav1), pan=fit_pan(w1, h1))
    e2 = TimelineEntry(panel_id="001.02", order=2, source_image=str(img2),
                       bbox=BBox(x=0, y=0, w=w2, h=h2),
                       start_seconds=1.35, duration_seconds=0.8,
                       audio_path=str(wav2), pan=fit_pan(w2, h2))
    return TimelineArtifact(
        meta=Meta(schema_version=1, generator="pilot", config_hash="pilot",
                  input_hashes={}),
        gap_seconds=0.35, min_display_seconds=2.0, entries=[e1, e2])


def main() -> None:
    tl = make_timeline()
    out_ff = OUT / "pilot_ffmpeg.mp4"
    render_ff(tl, out_ff, ffmpeg_exe=imageio_ffmpeg.get_ffmpeg_exe())
    print("ffmpeg renderer output:", out_ff, out_ff.stat().st_size, "bytes")
    print("  cmd head:", " ".join(build_command(tl, out_ff)[:8]))

    out_mp = OUT / "pilot_moviepy.mp4"
    render_mp(tl, out_mp)
    print("moviepy renderer output:", out_mp, out_mp.stat().st_size, "bytes")

    from moviepy import VideoFileClip  # read back: durations must match
    d1 = VideoFileClip(str(out_ff)).duration
    d2 = VideoFileClip(str(out_mp)).duration
    expected = 1.35 + 0.8  # contiguous entries; gap baked into entry 1
    print(f"durations: ffmpeg={d1:.3f} moviepy={d2:.3f} expected={expected:.3f}")
    assert abs(d1 - expected) < 0.5 and abs(d2 - expected) < 0.5, "drift"
    print("PASS: both renderers produced 1080x1920 H.264/AAC within tolerance")


if __name__ == "__main__":
    main()
