"""Style preset learned from "Recording 2026-09-16 064103.mp4".

Measured profile -> concrete CinematicConfig + render overrides.

Headline finding: the reference edit is CLEAN and RESTRAINED, whereas this
pipeline's default ("dynamic") is a loud YouTube-recap aesthetic. Matching the
reference is mostly SUBTRACTION (no glitch, no shake, no speed lines, no
vignette, no teal-orange grade) plus one structural change: the reference is
16:9 LANDSCAPE, but every renderer here is hardcoded to 9:16 portrait.
"""
from __future__ import annotations

from cinematic_effects import CinematicConfig

# Measured on the reference clip (style_ref/style_profile.json)
REF = {
    "aspect": "16:9",
    "resolution": (1292, 706),
    "saturation": 0.109,        # low / muted
    "brightness_mean": 138.7,
    "neutral_white_balance": True,
    "cuts": 10,
    "avg_shot_s": 2.33,
    "shot_lengths": [0.6, 2.4, 2.6, 4.0, 4.4, 2.8, 1.8, 2.0, 2.0, 1.8, 1.2],
    "zoom_factor": 1.10,        # Ken Burns in bursts, never punch
    "letterbox_px": 0,
    "vignette": False,
    "caption_style": "white on near-black, thin band, horizontally centered",
}


def recording_preset() -> CinematicConfig:
    """CinematicConfig that reproduces the reference edit's look."""
    return CinematicConfig(
        style="subtle",

        # --- Structure: 16:9 landscape, not 9:16 portrait ---
        fps=30,
        output_width=1920,
        output_height=1080,

        # --- Motion: gentle Ken Burns only, no punch/smash ---
        kb_zoom_start=1.00,
        kb_zoom_end=1.10,        # measured zoom factor in the reference
        kb_zoom_end_fast=1.10,   # action panels get the SAME gentle zoom here

        # Disable the punch-zoom entirely by making it a no-op.
        punch_frames=0,
        punch_scale=1.00,
        punch_settle=1.00,

        # --- No screen shake ---
        shake_enabled=False,

        # --- No glitch / chroma transitions: reference uses hard cuts ---
        glitch_enabled=False,

        # --- Color: muted + neutral, NOT teal-orange ---
        grade_contrast=1.00,
        grade_saturation=0.88,   # reference is desaturated (0.109)
        grade_shadows=0.00,      # neutral white balance
        grade_highlights=0.00,
        grade_brightness=0.00,

        # --- No vignette, no letterbox, no speed lines ---
        vignette_enabled=False,
        letterbox_enabled=False,
        speedlines_enabled=False,
    )


# Pacing model fitted to the reference shot-length distribution.
# Reference: fast 0.6s intro hook -> long 4.0-4.4s mid holds -> accelerating
# 1.8-2.0s outro. avg 2.33s/shot.
PACING = {
    # Timeline.build() kwargs (adapters/timeline.py)
    "timeline": {
        "min_display": 1.8,      # reference allows shots down to ~0.6s hook
        "max_display": 12.0,     # unchanged
        "gap": 0.0,              # reference has no silence gaps; hard cuts
        "silent_wpm": 160,       # unchanged
        "fps": 30,
        "pan_speed": 450,        # unchanged
    },
    # render_ffmpeg transitions: plain cuts, no xfade
    "transitions": [{"type": "cut"}],
}


def render_overrides() -> dict:
    """Non-CinematicConfig changes needed in the ffmpeg renderer.

    render_ffmpeg.py hardcodes WIDTH, HEIGHT = 1080, 1920. To emit 16:9 those
    must become 1920, 1080 (landscape). This is returned for the caller to
    apply; it is NOT monkey-patched here, because fit_pan()/display_seconds()
    read the module constants and a silent swap would change pan geometry for
    every panel without touching the regression suite.
    """
    return {
        "render_ffmpeg.WIDTH": 1920,
        "render_ffmpeg.HEIGHT": 1080,
        "note": ("fit_pan() uses max(WIDTH/w, HEIGHT/h); swapping to landscape "
                 "rotates the overflow axis -- wide panels become pan_down "
                 "candidates. Verify tests/test_recap_video.py geometry."),
    }
