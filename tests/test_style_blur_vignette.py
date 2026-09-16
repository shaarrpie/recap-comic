"""Tests for the manhwa-recap visual style: blur background, vignette,
colour grade — and their interaction with pan geometry + cache keys.

The filter graph is validated by construction here (string shape); the
end-to-end ffmpeg encode was verified manually with the bundled ffmpeg.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import recap_video as rv
from adapters.render_ffmpeg import (
    StyleConfig,
    _blur_bg_chain,
    _style_post_filters,
    build_command,
)
from adapters.schemas import BBox, Meta, PanSpec, TimelineArtifact, TimelineEntry


def _tl(pan_kind: str = "static", w: int = 1080, h: int = 1920,
        travel: int = 0) -> TimelineArtifact:
    return TimelineArtifact(
        meta=Meta(schema_version="1", generator="t", config_hash="h",
                  input_hashes={}),
        width=w, height=h, fps=30, gap_seconds=0.0, min_display_seconds=1.0,
        entries=[TimelineEntry(panel_id="p1", order=1, source_image="x.png",
                               bbox=BBox(x=0, y=0, w=1080, h=h),
                               start_seconds=0.0, duration_seconds=2.0,
                               audio_path=None,
                               pan=PanSpec(kind=pan_kind, scaled_w=1080,
                                           scaled_h=h, travel_px=travel))])


def _fc(cmd: list[str]) -> str:
    return cmd[cmd.index("-filter_complex") + 1]


# ------------------------------------------------------------------ defaults
def test_style_config_defaults_match_the_shipped_patch():
    """README of the style patch: blur ON, colour grade OFF, vignette ON,
    angle PI/2.5 (strong). These defaults must not drift silently."""
    s = StyleConfig()
    assert s.blur_background is True
    assert s.color_grade is False
    assert s.vignette is True
    assert s.vignette_angle == "PI/2.5"
    assert s.blur_sigma == pytest.approx(40.0)


def test_video_config_style_defaults_match_style_config():
    """VideoConfig is the user-facing surface; it must agree with the
    renderer's StyleConfig defaults so a plain `guided video` produces the
    documented look."""
    cfg = rv.VideoConfig()
    assert cfg.blur_background is True
    assert cfg.color_grade is False
    assert cfg.vignette is True
    assert cfg.vignette_angle == "PI/2.5"
    assert cfg.blur_sigma == pytest.approx(40.0)


# ------------------------------------------------------------- post filters
def test_no_style_means_no_post_filters():
    assert _style_post_filters(None) == []


def test_all_off_means_no_post_filters():
    assert _style_post_filters(StyleConfig(blur_background=True,
                                           color_grade=False,
                                           vignette=False)) == []


def test_vignette_only_emits_single_vignette():
    post = _style_post_filters(StyleConfig(color_grade=False))
    assert post == [f"vignette=angle={StyleConfig().vignette_angle}"]


def test_color_grade_emits_eq_and_channel_mixer():
    post = _style_post_filters(StyleConfig(color_grade=True, vignette=False))
    assert len(post) == 2
    assert post[0].startswith("eq=brightness=")
    assert post[1].startswith("colorchannelmixer=")
    # cool tint: blue kept, red reduced
    assert "bb=0.90" in post[1] and "rr=0.92" in post[1]


def test_vignette_angle_is_forwarded_verbatim():
    post = _style_post_filters(StyleConfig(vignette_angle="PI/3.5"))
    assert post == ["vignette=angle=PI/3.5"]


# ------------------------------------------------------------ blur bg chain
def test_blur_bg_chain_shape():
    chain = _blur_bg_chain(0, 1080, 1920, 40.0)
    # split feeds both branches; background covers+blurs, foreground contains
    assert chain.startswith("split=2[bgr0][fgr0]")
    assert "[bgr0]scale=1080:1920:force_original_aspect_ratio=increase" in chain
    assert "gblur=sigma=40" in chain
    assert "eq=brightness=-0.10:saturation=1.3" in chain
    assert "[fgr0]scale=1080:1920:force_original_aspect_ratio=decrease" in chain
    # composited centred; the caller appends setsar/fps/[vN]
    assert chain.endswith("overlay=x=(W-w)/2:y=(H-h)/2")


def test_blur_bg_chain_indices_are_unique_per_panel():
    a = _blur_bg_chain(0, 1080, 1920, 40.0)
    b = _blur_bg_chain(1, 1080, 1920, 40.0)
    assert "[bg0]" in a and "[bg1]" in b
    assert "[bg1]" not in a and "[bg0]" not in b


def test_blur_sigma_reaches_the_chain():
    chain = _blur_bg_chain(2, 540, 960, 12.5)
    assert "gblur=sigma=12.5" in chain
    assert "scale=540:960" in chain  # honours the declared (draft) canvas


# ------------------------------------------------------------- build_command
def test_blur_background_replaces_scale_crop_in_command():
    """In blur mode the panel chain must NOT contain a pan crop (there is no
    overflow to pan through once the panel is contain-fitted), and must
    contain the split/bg/fg composite instead."""
    tl = _tl(pan_kind="pan_down", travel=500)
    cmd = build_command(tl, Path("out.mp4"), style=StyleConfig())
    fc = _fc(cmd)
    assert "split=2[bgr0][fgr0]" in fc
    assert "gblur=sigma=40" in fc
    # the pan expression would move the crop window; it must be absent
    assert "crop=1080:1920:x=0:y='(ih-1920)*t" not in fc


def test_no_blur_keeps_the_legacy_pan_chain():
    """Disabling the blur must fall back to the exact scale+crop+pan path
    (regression guard for the non-style render)."""
    tl = _tl(pan_kind="pan_down", travel=500)
    cmd = build_command(tl, Path("out.mp4"),
                        style=StyleConfig(blur_background=False, vignette=False))
    fc = _fc(cmd)
    assert "split=2" not in fc
    assert "gblur" not in fc
    assert "crop=1080:1920:x=0:y='(ih-1920)*t/2.000'" in fc


def test_vignette_applied_once_on_composited_output():
    """Grade + vignette are frame-local: one pass after concat, not one per
    panel clip (identical result, one filter pass instead of N)."""
    tl = TimelineArtifact(
        meta=Meta(schema_version="1", generator="t", config_hash="h",
                  input_hashes={}),
        width=1080, height=1920, fps=30, gap_seconds=0.0,
        min_display_seconds=1.0,
        entries=[_tl().entries[0], _tl().entries[0]])
    cmd = build_command(tl, Path("out.mp4"), style=StyleConfig())
    fc = _fc(cmd)
    assert fc.count("vignette=angle=PI/2.5") == 1
    assert "[vcat]vignette=angle=PI/2.5[vout]" in fc
    # [vout] is what gets mapped, not [vcat]
    assert cmd[cmd.index("-map") + 1] == "[vout]"


def test_style_off_maps_vcat_directly():
    tl = _tl()
    cmd = build_command(tl, Path("out.mp4"),
                        style=StyleConfig(blur_background=False, vignette=False,
                                          color_grade=False))
    fc = _fc(cmd)
    assert "vignette" not in fc and "colorchannelmixer" not in fc
    assert cmd[cmd.index("-map") + 1] == "[vcat]"


def test_xfade_path_also_gets_the_style():
    """The xfade branch builds its own tail; the vignette must land there
    too or crossfaded videos would render ungraded."""
    tl = TimelineArtifact(
        meta=Meta(schema_version="1", generator="t", config_hash="h",
                  input_hashes={}),
        width=1080, height=1920, fps=30, gap_seconds=0.0,
        min_display_seconds=1.0,
        entries=[_tl().entries[0], _tl().entries[0]])
    transitions = [{"type": "fade", "duration": 0.5}]  # exactly n-1
    cmd = build_command(tl, Path("out.mp4"), transitions=transitions,
                        style=StyleConfig())
    fc = _fc(cmd)
    assert "[vcat]vignette=angle=PI/2.5[vout]" in fc
    assert "xfade=transition=fade" in fc


def test_grade_and_vignette_order_in_command():
    tl = _tl()
    cmd = build_command(tl, Path("out.mp4"), style=StyleConfig(color_grade=True))
    fc = _fc(cmd)
    tail = fc[fc.index("[vcat]"):]
    # eq + mixer run before the vignette (grade first, then darkening)
    assert tail.index("eq=brightness=") < tail.index("colorchannelmixer=") \
        < tail.index("vignette=")


# ------------------------------------------------------- pan geometry (blur)
def test_compute_pan_blur_mode_is_static_and_contains_fitted():
    """A tall panel that would normally pan must become static in blur mode:
    the whole panel is shown floating on the blurred background, so there is
    no overflow to pan through and no pan floor padding the duration."""
    p = rv.compute_pan(780, 1400, 1080, 1920, blur_background=False)
    assert p.kind == "pan_down" and p.travel_px > 0
    b = rv.compute_pan(780, 1400, 1080, 1920, blur_background=True)
    assert b.kind == "static"
    assert b.travel_px == 0
    # contain-fit: both scaled dims within the canvas
    assert b.scaled_w <= 1080 and b.scaled_h <= 1920
    # height is the binding axis; width scales proportionally (ceil-rounded)
    assert b.scaled_h == 1920
    assert abs(b.scaled_w - 780 * (1920 / 1400)) <= 1


def test_compute_pan_blur_mode_wide_panel_also_static():
    b = rv.compute_pan(2000, 800, 1080, 1920, blur_background=True)
    assert b.kind == "static" and b.travel_px == 0
    assert b.scaled_w == 1080
    assert abs(b.scaled_h - 800 * (1080 / 2000)) <= 1


def test_compute_pan_blur_mode_never_upscales_beyond_cap():
    """A tiny panel stays tiny-ish (4x cap) instead of exploding to fill the
    canvas; the blurred background fills the rest."""
    b = rv.compute_pan(10, 10, 1080, 1920, blur_background=True)
    assert b.scaled_w == 40 and b.scaled_h == 40   # 4.0 cap
    assert b.travel_px == 0


# ------------------------------------------------------------- cache keys
def test_style_flags_do_not_change_hash_essentials():
    """Narration and TTS depend only on text/voice/provider/pacing. Toggling
    a visual flag must NOT invalidate the audio cache (which would silently
    re-synthesize identical clips and re-bill the TTS provider)."""
    base = rv.VideoConfig()
    off = rv.VideoConfig(blur_background=False, vignette=False,
                         vignette_angle="PI/3", blur_sigma=10.0)
    assert base.hash() != off.hash()                 # style changes identity
    assert base.hash_essentials() == off.hash_essentials()


def test_non_style_change_does_change_hash_essentials():
    a = rv.VideoConfig()
    b = rv.VideoConfig(voice="en-US-GuyNeural", gap_seconds=1.0)
    assert a.hash_essentials() != b.hash_essentials()


def test_hash_essentials_excludes_exactly_the_style_fields():
    import json
    from dataclasses import asdict
    cfg = rv.VideoConfig()
    d = json.loads(json.dumps(asdict(cfg), default=str))
    for f in rv.VideoConfig._STYLE_FIELDS:
        del d[f]
    import hashlib
    expected = hashlib.sha256(
        json.dumps(d, sort_keys=True).encode()).hexdigest()
    assert cfg.hash_essentials() == expected


def test_color_grade_off_by_default_saves_a_pass():
    """The shipped patch ships the grade OFF; the default command must
    therefore contain no colour-channel mixer (cheapest default path)."""
    cmd = build_command(_tl(), Path("out.mp4"), style=StyleConfig())
    fc = _fc(cmd)
    assert "colorchannelmixer" not in fc
    assert "eq=brightness=-0.07" not in fc
