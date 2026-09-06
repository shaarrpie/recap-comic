# adapters/timeline.py
"""Timeline assembly: PanelsArtifact + NarrationArtifact + AudioArtifact
→ TimelineArtifact.

This is the brain of the video stage. It is intentionally independent of
any particular panel detector (Stack A or Stack B) or TTS backend.
"""
from __future__ import annotations

import logging
import re

from .schemas import (
    SCHEMA_VERSION,
    AudioArtifact,
    Meta,
    NarrationArtifact,
    PanelsArtifact,
    PanSpec,
    TimelineArtifact,
    TimelineEntry,
)

log = logging.getLogger(__name__)

WIDTH, HEIGHT = 1080, 1920


def fit_pan(w: int, h: int) -> PanSpec:
    """Scale a panel so it covers 1080×1920; overflow axis gets the pan.

    Never centre-crops away content: a tall panel pans down, a wide one pans
    right, an exact-fit stays static.
    """
    if w <= 0 or h <= 0:
        raise ValueError(f"panel must have positive size, got {w}x{h}")
    scale = max(WIDTH / w, HEIGHT / h)
    scaled_w = max(WIDTH, round(w * scale))
    scaled_h = max(HEIGHT, round(h * scale))
    over_h = scaled_h - HEIGHT
    over_w = scaled_w - WIDTH
    if over_h > 2 and over_h >= over_w:
        return PanSpec(kind="pan_down", scaled_w=scaled_w, scaled_h=scaled_h,
                       travel_px=over_h)
    if over_w > 2:
        return PanSpec(kind="pan_right", scaled_w=scaled_w, scaled_h=scaled_h,
                       travel_px=over_w)
    return PanSpec(kind="static", scaled_w=scaled_w, scaled_h=scaled_h,
                   travel_px=0)


def _audio_duration(audio: AudioArtifact, panel_id: str) -> float | None:
    for e in audio.entries:
        if e.entry_id == panel_id:
            return e.duration_seconds
    return None


def _word_count(text: str) -> int:
    latin = len(re.compile(r"[A-Za-z0-9']+").findall(text))
    cjk = len(re.compile(
        r"[\u3040-\u30ff\uac00-\ud7af\u4e00-\u9fff]").findall(text))
    return latin + cjk


def display_seconds(*, audio_seconds: float | None, words: int,
                    travel_px: int, gap: float, min_display: float,
                    max_display: float, silent_wpm: int,
                    pan_speed: int = 450) -> float:
    """How long a panel stays on screen (including its trailing gap)."""
    pan_floor = travel_px / pan_speed if travel_px else 0.0
    if audio_seconds is not None:
        return max(audio_seconds + gap, min_display, pan_floor)
    read = (words / silent_wpm) * 60.0 if words else 0.0
    return min(max(read + gap, min_display), max_display)


def build(panels: PanelsArtifact, narration: NarrationArtifact,
          audio: AudioArtifact, *, gap: float = 0.35,
          min_display: float = 2.0, max_display: float = 12.0,
          silent_wpm: int = 160, fps: int = 30,
          config_hash: str = "", input_hashes: dict[str, str] | None = None,
          pan_speed: int = 450) -> TimelineArtifact:
    """Assemble a contiguous timeline from already-built stages."""
    by_audio = {a.entry_id: a for a in audio.entries}
    by_text = {n.id: n for n in narration.entries}
    entries: list[TimelineEntry] = []
    t = 0.0
    for order, p in enumerate(panels.panels, start=1):
        h = p.bbox.h
        if h <= 0:
            log.warning("skipping zero-height panel %s", p.id)
            continue
        a = by_audio.get(p.id)
        text = by_text[p.id].text if p.id in by_text else ""
        dur = display_seconds(
            audio_seconds=a.duration_seconds if a else None,
            words=_word_count(text),
            travel_px=fit_pan(p.bbox.w, h).travel_px,
            gap=gap, min_display=min_display,
            max_display=max_display, silent_wpm=silent_wpm,
            pan_speed=pan_speed)
        entries.append(TimelineEntry(
            panel_id=p.id, order=order,
            source_image=p.source_image,
            bbox=p.bbox,
            start_seconds=round(t, 3),
            duration_seconds=round(dur, 3),
            audio_path=a.path if a else None,
            pan=fit_pan(p.bbox.w, h)))
        t += dur
    if not entries:
        raise ValueError("no usable panels in PanelsArtifact")
    meta_input_hashes = dict(input_hashes or {})
    return TimelineArtifact(
        meta=Meta(schema_version=SCHEMA_VERSION,
                  generator="adapters.timeline",
                  config_hash=config_hash,
                  input_hashes=meta_input_hashes),
        width=WIDTH, height=HEIGHT, fps=fps,
        gap_seconds=gap, min_display_seconds=min_display,
        entries=entries)
