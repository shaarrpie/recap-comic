# adapters/render_moviepy.py
"""MoviePy renderer (renderer implementation 2 of 2).

Target: MoviePy 2.x ONLY (pinned moviepy==2.2.1). Never mix 1.x names
(set_duration/resize/crop/set_audio) with 2.x names
(with_duration/resized/cropped/with_audio). The 1.x->2.x rename table page
did not render via fetch, so the 2.x method names below are verified
EMPIRICALLY by pilots/pilot_render.py executing against moviepy==2.2.1.
No TextClip is used anywhere, so ImageMagick is NOT required (MoviePy 2.x
only needs ImageMagick for TextClip).
"""
from __future__ import annotations

from pathlib import Path

from .schemas import TimelineArtifact

WIDTH, HEIGHT = 1080, 1920


def render(timeline: TimelineArtifact, out_path: Path) -> None:
    from moviepy import AudioFileClip, ColorClip, CompositeVideoClip, ImageClip

    tracks = []
    for e in timeline.entries:
        clip = ImageClip(e.source_image).resized(
            (e.pan.scaled_w, e.pan.scaled_h))
        travel = e.pan.travel_px
        dur = max(e.duration_seconds, 1e-6)
        if e.pan.kind == "pan_down":
            clip = clip.with_position(
                lambda t, tr=travel, d=dur: (0, -min(int(tr * t / d), tr)))
        elif e.pan.kind == "pan_right":
            clip = clip.with_position(
                lambda t, tr=travel, d=dur: (-min(int(tr * t / d), tr), 0))
        else:
            clip = clip.with_position((0, 0))
        clip = clip.with_start(e.start_seconds).with_duration(
            e.duration_seconds)
        if e.audio_path:
            clip = clip.with_audio(AudioFileClip(e.audio_path))
        tracks.append(clip)

    total = timeline.entries[-1].start_seconds + \
        timeline.entries[-1].duration_seconds
    background = ColorClip(size=(WIDTH, HEIGHT), color=(0, 0, 0),
                           duration=total)
    video = CompositeVideoClip([background, *tracks], size=(WIDTH, HEIGHT))
    video.write_videofile(
        str(out_path), fps=timeline.fps, codec="libx264",
        audio_codec="aac", preset="veryfast",
        ffmpeg_params=["-pix_fmt", "yuv420p"])
