"""Offline tests for the reference-motion preset (no ffmpeg / TTS).

The shipped preset is a STRICT sequential 8-beat cycle: 1) zoom_in 2) pan_down
3) pan_right 4) zoom_out 5) pan_up 6) pan_left 7) zoom_in 8) pan_down, repeated
continuously (panel i -> segment i mod 8).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import motion_presets as mp
import recap_video as rv
from adapters.render_ffmpeg import _blur_bg_chain, build_command
from adapters.schemas import (
    AudioArtifact,
    BBox,
    Meta,
    NarrationArtifact,
    NarrationEntry,
    PanSpec,
    TimelineArtifact,
    TimelineEntry,
)
from guided_cutter import CutArtifact, CutPanel


def _preset() -> mp.MotionPreset:
    return mp.load_preset(None)


def _panel(i: int, y0: int, y1: int, narration: str = "He runs.") -> CutPanel:
    return CutPanel(id=f"{i:03d}", panel_index=i, y_start=y0, y_end=y1,
                    narration=narration, dialogue="",
                    panel_type="single", confidence=0.9,
                    image_file=f"panel_{i:03d}.png")


def _meta() -> Meta:
    return Meta(schema_version=1, generator="t", config_hash="c",
                input_hashes={})


# ------------------------------------------------------------ preset loading
def test_preset_loads_8_beat_cycle_in_order():
    p = _preset()
    assert len(p) == 8
    assert [s.seg for s in (p.segments or [])] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert abs(p.total_duration - 18.0) < 0.05
    assert p.reference_width == pytest.approx(1312.0)
    # The strict cycle contract. The old 4-beat template could only ever
    # produce zoom + vertical pan; the 8-beat one adds lateral slides and a
    # second zoom pass so a recap is not nothing-but-up-and-down sliding.
    kinds = [mp.pan_kind_for_seg(s, p) for s in (p.segments or [])]
    assert kinds == ["zoom_in", "pan_down", "pan_right", "zoom_out",
                     "pan_up", "pan_left", "zoom_in", "pan_down"]


def test_normalized_movement_uses_reference_dims():
    p = _preset()
    seg2 = (p.segments or [])[1]  # pan_down beat
    n = mp.normalized_movement(seg2, p)
    assert n["ndx"] == pytest.approx(0.0)
    assert n["ndy"] == pytest.approx(150.0 / p.reference_height)
    assert n["ndyps"] == pytest.approx(60.0 / p.reference_height)


def test_static_vs_moving_classification():
    p = _preset()
    segs = {s.seg: s for s in (p.segments or [])}
    assert mp.is_static(segs[1], p) is True   # zoom_in beat: zoom only
    assert mp.is_static(segs[2], p) is False  # pan_down beat: moving
    assert mp.is_static(segs[3], p) is False  # pan_up beat: moving
    assert mp.is_static(segs[4], p) is True   # zoom_out beat: zoom only


def test_low_confidence_damps_but_keeps_motion():
    # Synthetic low-confidence copy of the pan_down beat: the shipped cycle
    # is all high-confidence, so damping is exercised on a crafted segment.
    p = _preset()
    segs = {s.seg: s for s in (p.segments or [])}
    strong = segs[2]
    weak = mp.ReferenceSegment(seg=strong.seg, t0=strong.t0, t1=strong.t1,
                               dur=strong.dur, zoom=strong.zoom,
                               matchScore=0.5, dx=strong.dx, dy=strong.dy,
                               dxps=strong.dxps, dyps=strong.dyps)
    assert mp.confidence_damping(strong, p) == pytest.approx(1.0)
    damped = mp.confidence_damping(weak, p)
    assert 0.0 < damped < 1.0
    assert damped == pytest.approx(p.low_confidence_damping)


def test_zoom_strength_preserves_ordering_and_stays_slow():
    p = _preset()
    segs = {s.seg: s for s in (p.segments or [])}
    z_out = mp.zoom_strength_for_seg(segs[4])  # 0.92
    z_norm = mp.zoom_strength_for_seg(segs[2])  # 1.00
    z_in = mp.zoom_strength_for_seg(segs[1])  # 1.14
    assert z_out < z_norm < z_in
    # Slow/cinematic cap: no push-in exceeds 0.4.
    assert 0.0 <= z_out <= 0.4 and 0.0 <= z_in <= 0.4


def test_pan_kind_sign_convention():
    p = _preset()
    segs = {s.seg: s for s in (p.segments or [])}
    assert mp.pan_kind_for_seg(segs[1], p) == "zoom_in"    # static + zoom in
    assert mp.pan_kind_for_seg(segs[2], p) == "pan_down"   # dy>0 dominant
    assert mp.pan_kind_for_seg(segs[3], p) == "pan_right"  # dx>0 dominant
    assert mp.pan_kind_for_seg(segs[4], p) == "zoom_out"   # static + zoom out
    assert mp.pan_kind_for_seg(segs[5], p) == "pan_up"     # dy<0 dominant
    assert mp.pan_kind_for_seg(segs[6], p) == "pan_left"   # dx<0 dominant
    assert mp.pan_kind_for_seg(segs[7], p) == "zoom_in"
    assert mp.pan_kind_for_seg(segs[8], p) == "pan_down"


def test_mapping_cycles_strictly_from_first_beat():
    p = _preset()
    assert mp.map_segments_to_panels(4, p) == [0, 1, 2, 3]
    # More panels than beats: the 8-beat pattern repeats continuously.
    assert mp.map_segments_to_panels(16, p) == list(range(8)) * 2
    assert mp.map_segments_to_panels(9, p) == [0, 1, 2, 3, 4, 5, 6, 7, 0]
    # Fewer panels than beats: truncate IN ORDER from beat 1 (never an
    # off-beat or mid-pattern start).
    assert mp.map_segments_to_panels(3, p) == [0, 1, 2]
    assert mp.map_segments_to_panels(1, p) == [0]
    assert mp.map_segments_to_panels(0, p) == []


# ---------------------------------------------------------------- resolve
def test_blur_resolve_never_reveals_empty_and_keeps_direction():
    p = _preset()
    segs = {s.seg: s for s in (p.segments or [])}
    r = mp.resolve_for_panel(png_w=390, png_h=800, canvas_w=1080,
                             canvas_h=1920, seg=segs[2], preset=p,
                             blur_background=True)
    assert r["kind"] == "pan_down"
    assert r["pan_y_px"] > 0  # downward preserved
    # The swept distance is a fixed fraction of the closer crop's hidden
    # height. The old "never pan more than 8% of the canvas" clamp is gone: it
    # was written for the bare contain-fit and would have neutralised the
    # 2.0-2.2x framing entirely.
    assert r["travel_px"] == round((r["panel_scale"] - 1.0) * 1920
                                   * mp.PAN_TRAVEL_FRACTION)
    assert r["ndy"] == pytest.approx(150.0 / p.reference_height)
    # Safety invariant for EVERY beat and panel shape: the renderer holds the
    # foreground at canvas_h * panel_scale, so a vertical reveal can never
    # exceed (panel_scale - 1) * canvas_h, and a lateral slide is bounded by
    # LATERAL_SLIDE_FRACTION of the canvas width (the panel drifts over the
    # blurred backdrop, which always fills the frame, so no empty area exists).
    for seg in (p.segments or []):
        for pw, ph in ((390, 800), (800, 1600), (900, 700), (200, 3000)):
            rr = mp.resolve_for_panel(png_w=pw, png_h=ph, canvas_w=1080,
                                      canvas_h=1920, seg=seg, preset=p,
                                      blur_background=True)
            assert abs(rr["pan_y_px"]) \
                <= (rr["panel_scale"] - 1.0) * 1920 + 1
            assert abs(rr["pan_x_px"]) \
                <= mp.LATERAL_SLIDE_FRACTION * 1080 + 1
            assert rr["travel_px"] >= 0


def test_blur_static_seg_has_no_pan_but_keeps_zoom():
    p = _preset()
    segs = {s.seg: s for s in (p.segments or [])}
    r = mp.resolve_for_panel(png_w=390, png_h=800, canvas_w=1080,
                             canvas_h=1920, seg=segs[1], preset=p,
                             blur_background=True)
    assert r["static"] is True
    assert r["pan_x_px"] == 0.0 and r["pan_y_px"] == 0.0
    # zoom_in beat pushes harder than the normal-framing pan beat.
    assert r["zoom_strength"] > mp.zoom_strength_for_seg(segs[2])


def test_cover_resolve_clamps_to_overflow():
    p = _preset()
    segs = {s.seg: s for s in (p.segments or [])}
    # Exact-fit panel: no overflow, so even a large dy must clamp to static.
    r = mp.resolve_for_panel(png_w=1080, png_h=1920, canvas_w=1080,
                             canvas_h=1920, seg=segs[2], preset=p,
                             blur_background=False)
    assert r["travel_px"] == 0
    assert r["scaled_w"] >= 1080 and r["scaled_h"] >= 1920
    # Tall panel: travel fits inside the overflow, direction kept.
    r2 = mp.resolve_for_panel(png_w=800, png_h=3600, canvas_w=1080,
                              canvas_h=1920, seg=segs[2], preset=p,
                              blur_background=False)
    assert r2["kind"] == "pan_down"
    assert 0 < r2["travel_px"] <= (r2["scaled_h"] - 1920)


# ---------------------------------------------------------------- timeline
def _timeline_with_preset(tmp_path: Path, n: int = 3) -> TimelineArtifact:
    panels = []
    y = 0
    for i in range(1, n + 1):
        h = 1200
        panels.append(_panel(i, y, y + h, f"Panel {i} narration here."))
        Image.new("RGB", (800, h), "white").save(
            tmp_path / f"panel_{i:03d}.png")
        y += h
    art = CutArtifact(source="strip.png", width=800, height=y,
                      plan_hash="x", config={}, panels=panels)
    narration = NarrationArtifact(
        meta=_meta(), mode="narrator",
        entries=[NarrationEntry(id=p.id, panel_id=p.id, order=i + 1,
                                text=f"Panel {i + 1} narration here.")
                 for i, p in enumerate(panels)])
    audio = AudioArtifact(meta=_meta(), voice="none", entries=[])
    cfg = rv.VideoConfig(tts="none", motion_preset="reference")
    return rv.build_timeline(art, tmp_path, narration, audio, tmp_path, cfg,
                             panels_hash="x")


def test_timeline_threads_strict_cycle_and_preserves_rhythm(tmp_path: Path):
    tl = _timeline_with_preset(tmp_path, 9)
    assert len(tl.entries) == 9
    kinds = [e.pan.kind for e in tl.entries]
    # The 8-beat cycle, repeated continuously in strict order.
    assert kinds == ["zoom_in", "pan_down", "pan_right", "zoom_out",
                     "pan_up", "pan_left", "zoom_in", "pan_down",
                     "zoom_in"]
    # Longer beats stay long relative to shorter ones (rhythm preserved).
    durs = [e.duration_seconds for e in tl.entries]
    assert durs[1] >= durs[0]  # seg2 (2.5s beat) >= seg1 (2.0s beat)
    for e in tl.entries:
        assert e.motion is not None
        assert {"seg", "zoom", "dx", "dy", "ndx", "ndy",
                "confidence"}.issubset(set(e.motion.keys()))


def test_default_video_config_enforces_the_cycle(tmp_path: Path):
    """The default pipeline now reproduces the strict cycle: every timeline
    entry carries motion metadata and the kinds follow the 8-beat order."""
    tl = _timeline_with_preset(tmp_path, 4)
    assert [e.pan.kind for e in tl.entries] == \
        ["zoom_in", "pan_down", "pan_right", "zoom_out"]
    assert tl.entries[0].motion is not None
    # Every shot is framed closer than the contain-fit, and the shot only
    # sweeps a fraction of that crop so the camera drifts instead of racing.
    for e in tl.entries:
        assert e.motion["panel_scale"] == pytest.approx(
            mp.NORMAL_PANEL_SCALE)
        assert e.motion["pan_travel_frac"] == pytest.approx(
            mp.PAN_TRAVEL_FRACTION)


def test_motion_report_rows(tmp_path: Path):
    tl = _timeline_with_preset(tmp_path, 3)
    rows = rv.build_motion_report(tl)
    assert len(rows) == 3
    assert rows[0]["seg"] == 1
    assert "ndx" in rows[0] and "ndy" in rows[0]
    assert "confidence" in rows[0]


def test_default_path_has_no_motion(tmp_path: Path):
    panels = [_panel(1, 0, 1200, "Hello world here.")]
    Image.new("RGB", (800, 1200), "white").save(tmp_path / "panel_001.png")
    art = CutArtifact(source="s.png", width=800, height=1200,
                      plan_hash="x", config={}, panels=panels)
    narration = NarrationArtifact(
        meta=_meta(), mode="narrator",
        entries=[NarrationEntry(id="001", panel_id="001", order=1,
                                text="Hello world here.")])
    audio = AudioArtifact(meta=_meta(), voice="none", entries=[])
    tl = rv.build_timeline(art, tmp_path, narration, audio, tmp_path,
                           rv.VideoConfig(tts="none", motion_preset="none"),
                           panels_hash="x")
    assert tl.entries[0].motion is None
    assert rv.build_motion_report(tl) == []


# ---------------------------------------------------------------- renderer
def _tl_entry(motion: dict | None = None,
              kind: str = "static") -> TimelineArtifact:
    return TimelineArtifact(
        meta=Meta(schema_version="1", generator="t", config_hash="h",
                  input_hashes={}),
        width=1080, height=1920, fps=30, gap_seconds=0.0,
        min_display_seconds=1.0,
        entries=[TimelineEntry(panel_id="p1", order=1, source_image="x.png",
                               bbox=BBox(x=0, y=0, w=1080, h=1920),
                               start_seconds=0.0, duration_seconds=2.0,
                               audio_path=None,
                               pan=PanSpec(kind=kind, scaled_w=1080,
                                           scaled_h=1920, travel_px=0),
                               motion=motion)])


def test_blur_chain_default_unchanged_without_pan():
    chain = _blur_bg_chain(0, 1080, 1920, 40.0)
    assert chain.endswith("overlay=x=(W-w)/2:y=(H-h)/2")


def test_blur_chain_pan_animates_overlay_smoothly():
    chain = _blur_bg_chain(0, 1080, 1920, 40.0, zoom=0.3, dur=2.0,
                           pan_x=0.0, pan_y=60.0)
    assert "overlay=x='(W-w)/2+(0)*t/2.000'" in chain
    assert "(60)*t/2.000" in chain


def test_blur_chain_cinematic_kinds_are_distinct():
    """With a motion preset the four cycle kinds must each produce a
    DISTINCT, full-clip (slow) move -- this is what makes the edit-rotation
    cycle actually visible."""
    zin = _blur_bg_chain(0, 1080, 1920, 40.0, dur=6.0, kind="zoom_in")
    zout = _blur_bg_chain(0, 1080, 1920, 40.0, dur=6.0, kind="zoom_out")
    pdown = _blur_bg_chain(0, 1080, 1920, 40.0, dur=6.0, kind="pan_down")
    pup = _blur_bg_chain(0, 1080, 1920, 40.0, dur=6.0, kind="pan_up")
    # zoom_in grows over the whole clip; zoom_out shrinks over the whole clip
    assert "(1920/ih)*(1+0.35*t/6.000)" in zin
    assert "(1920/ih)*(1.35-0.35*t/6.000)" in zout
    # pan shots hold a fixed taller foreground and animate ONLY overlay y
    assert "y='(H-h)/2+" in pdown and "*t/6.000" in pdown
    assert "y='(H-h)/2-" in pup and "*t/6.000" in pup
    # all four moves are different
    assert len({zin, zout, pdown, pup}) == 4


def test_build_command_uses_cinematic_kind_when_preset_present():
    from adapters.render_ffmpeg import StyleConfig
    tl = _tl_entry(motion={"preset": "reference_v1", "zoom_strength": 0.22,
                           "pan_x_px": 0.0, "pan_y_px": 0.0}, kind="zoom_in")
    cmd = build_command(tl, Path("out.mp4"), style=StyleConfig())
    fc = cmd[cmd.index("-filter_complex") + 1]
    # kind-driven grow, floored at the cinematic magnitude (0.35)
    assert "(1920/ih)*(1+0.35*t/2.000)" in fc
    # the legacy contain->cover push-in must NOT appear once a preset drives it
    assert "min(1080/iw,1920/ih)" not in fc


# ------------------------------------------------------- tall/long panels
def test_resolve_tall_panel_is_closer_and_honours_the_beat():
    """Narrow portrait art is framed at TALL_PANEL_SCALE and the BEAT chooses
    the move.

    The resolver used to hard-code `pan_down if beat even else pan_up` for tall
    panels, which threw away every zoom and slide beat -- that is why a whole
    recap slid only up and down. Travel is now only PAN_TRAVEL_FRACTION of the
    hidden height, so the framing stays close while the camera drifts.
    """
    p = _preset()
    segs = p.segments or []
    kw = dict(png_w=800, png_h=1600, canvas_w=1920, canvas_h=1080,
              preset=p, blur_background=True)
    hidden = (mp.TALL_PANEL_SCALE - 1.0) * 1080        # 1296px out of frame
    v_travel = round(hidden * mp.PAN_TRAVEL_FRACTION)  # 454px swept
    h_travel = round(1920 * mp.LATERAL_SLIDE_FRACTION)  # 154px sideways
    r_zoom = mp.resolve_for_panel(seg=segs[0], **kw)    # zoom_in beat
    r_vpan = mp.resolve_for_panel(seg=segs[1], **kw)    # pan_down beat
    r_slide = mp.resolve_for_panel(seg=segs[2], **kw)   # pan_right beat
    for r in (r_zoom, r_vpan, r_slide):
        assert r["tall_panel"] is True
        assert r["panel_scale"] == pytest.approx(mp.TALL_PANEL_SCALE)
        assert r["pan_travel_frac"] == pytest.approx(mp.PAN_TRAVEL_FRACTION)
        assert 0 <= r["travel_px"] <= max(hidden, h_travel)
    # A zoom beat stays a zoom on tall art (previously coerced into a pan).
    assert r_zoom["kind"] == "zoom_in"
    assert r_zoom["travel_px"] == 0
    assert r_zoom["zoom_strength"] > 0.0
    # Vertical pan: swept fraction of the hidden height, direction from beat.
    assert r_vpan["kind"] == "pan_down"
    assert r_vpan["travel_px"] == v_travel
    assert r_vpan["pan_y_px"] == pytest.approx(float(v_travel))
    # Lateral slide: whole-panel drift over the blurred backdrop.
    assert r_slide["kind"] == "pan_right"
    assert r_slide["travel_px"] == h_travel
    assert r_slide["pan_x_px"] == pytest.approx(float(h_travel))
    assert r_slide["pan_y_px"] == 0.0
    # Direction comes from the beat, never from parity.
    r_up = mp.resolve_for_panel(seg=segs[4], **kw)      # pan_up beat
    assert r_up["kind"] == "pan_up"
    assert r_up["pan_y_px"] < 0


def test_resolve_normal_panel_is_not_tall():
    p = _preset()
    seg = (p.segments or [])[0]  # zoom_in beat
    r = mp.resolve_for_panel(
        png_w=1000, png_h=1000, canvas_w=1920, canvas_h=1080,
        seg=seg, preset=p, blur_background=True)
    assert r["tall_panel"] is False
    # Wide art gets its own closer framing too (not the bare contain-fit):
    # the blurred pillars around a contain-fit panel are what read as "blank
    # space around the manhwa".
    assert r["panel_scale"] == pytest.approx(mp.NORMAL_PANEL_SCALE)
    assert r["kind"] == "zoom_in"
    assert r["zoom_strength"] > 0.0


def test_blur_chain_tall_pan_holds_scale_and_moves_y():
    """pan_frac = panel_scale - 1 must hold a FIXED closer foreground and
    animate ONLY the overlay y (no per-frame scale growth == no zoom)."""
    chain = _blur_bg_chain(0, 1920, 1080, 40.0, dur=6.0,
                           kind="pan_down", pan_frac=0.2)
    # foreground held at 1.2x contain (constant factor, no t term)
    assert "(1080/ih)*(1+0.2)" in chain
    assert "*(1+0.2*t" not in chain                 # no zoom growth
    # overlay y traverses the full 0.2*h overflow slowly across the clip
    assert "y='(H-h)/2+384.000-384.000*t/6.000'" in chain or \
           "y='(H-h)/2+216.000-216.000*t/6.000'" in chain


def test_build_command_tall_panel_closer_and_no_zoom():
    from adapters.render_ffmpeg import StyleConfig
    tl = _tl_entry(motion={"preset": "reference_v1", "zoom_strength": 0.0,
                           "pan_x_px": 0.0, "pan_y_px": 384.0,
                           "tall_panel": True, "panel_scale": 1.2},
                   kind="pan_down")
    cmd = build_command(tl, Path("out.mp4"), style=StyleConfig())
    fc = cmd[cmd.index("-filter_complex") + 1]
    # closer 1.2x hold (portrait canvas h=1920 -> factor 0.2 -> travel 384)
    assert "(1920/ih)*(1+0.2)" in fc
    assert "384.000" in fc
    # strictly a pan: no zoom push/pull, no legacy contain->cover push-in
    assert "*(1+0.2*t" not in fc
    assert "(1+0.35*t" not in fc
    assert "min(1080/iw,1920/ih)" not in fc


# ------------------------------------------------- super-tall split layout
def test_resolve_super_tall_panel_splits_side_by_side():
    """An 8:1 strip is too thin to read even closer, so it splits into two
    bands shown side by side (whole panel SPLIT_CLOSENESS x closer at once,
    static, no overlay pan -- the bands drift per-band in the renderer)."""
    p = _preset()
    seg = (p.segments or [])[0]
    r = mp.resolve_for_panel(
        png_w=800, png_h=6395, canvas_w=1920, canvas_h=1080,
        seg=seg, preset=p, blur_background=True)
    assert r["split_columns"] == 2
    assert r["tall_panel"] is True
    assert r["kind"] == "static"          # per-band pan is renderer-driven
    assert r["travel_px"] == 0
    assert r["zoom_strength"] == 0.0       # never a zoom
    assert r["split_dir"] == 1             # first split panel default
    assert r["split_gap_frac"] == pytest.approx(0.05)
    # SPLIT_CLOSENESS is the requested whole-panel closeness; the per-band
    # overflow follows from it (2.5 / 2 bands - 1 = 0.25, was 0.15).
    assert r["split_pan_frac"] == pytest.approx(
        mp.SPLIT_CLOSENESS / mp.SPLIT_COLUMNS - 1.0)
    # composite (two 800x3197 bands side by side) contained to 1920x1080:
    # ~540 wide (side pillars remain), ~1080 tall.
    assert 500 <= r["scaled_w"] <= 600
    assert 1075 <= r["scaled_h"] <= 1085
    assert r["scaled_w"] < 1920


def test_resolve_moderately_tall_panel_not_split():
    p = _preset()
    segs = p.segments or []
    kw = dict(png_w=800, png_h=1645, canvas_w=1920, canvas_h=1080,
              preset=p, blur_background=True)  # 2.06:1 < SPLIT_MIN_ASPECT 3.0
    r = mp.resolve_for_panel(seg=segs[0], **kw)
    assert r["split_columns"] == 0
    assert r["tall_panel"] is True             # still the closer + pan path
    assert r["panel_scale"] == pytest.approx(mp.TALL_PANEL_SCALE)
    assert r["kind"] == "zoom_in"              # beat 1 honoured, not coerced
    r2 = mp.resolve_for_panel(seg=segs[1], **kw)   # beat 2 is a vertical pan
    assert r2["split_columns"] == 0
    assert r2["kind"] == "pan_down"
    assert r2["travel_px"] == round((mp.TALL_PANEL_SCALE - 1.0) * 1080
                                    * mp.PAN_TRAVEL_FRACTION)


def test_blur_chain_split_has_gap_and_overlays_on_bg():
    """Bands are overlaid directly on the blurred bg (no hstack composite) so
    a `gap` (round(W*SPLIT_GAP_FRAC)) of blurred background shows between them."""
    chain = _blur_bg_chain(0, 1920, 1080, 40.0, columns=2, dur=6.0)
    # single frame is loop-held (blur/scale computed once) before splitting
    assert "[fgr0]loop=" in chain
    assert "split=2[sl0_0][sl1_0]" in chain
    assert "crop=iw:trunc(ih/2):0:trunc(ih*0/2)" in chain
    assert "crop=iw:trunc(ih/2):0:trunc(ih*1/2)" in chain
    assert "hstack" not in chain                      # no side-by-side composite
    # (the blurred-background branch legitimately uses force_original_aspect_ratio)
    # gap = round(1920*0.05) = 96; band slots separated by (w + 96)
    assert "overlay=x='(W-(2*w+96))/2+0*(w+96)':y=0" in chain
    assert "overlay=x='(W-(2*w+96))/2+1*(w+96)':y=0" in chain
    assert "[bg0][vp0_0]overlay=" in chain            # composited onto blurred bg
    assert "eval=frame" not in chain                  # never a zoom push/pull


def test_blur_chain_split_bands_pan_in_opposite_directions():
    """split_dir=+1 -> band0 pans down, band1 pans up; split_dir=-1 flips both.
    With 3x vertical supersampling (h=1080): viewport 3240, band 3726, so the
    pan travels 486 supersampled px (== 162 final px) then downscales to 1080."""
    fwd = _blur_bg_chain(0, 1920, 1080, 40.0, columns=2, dur=6.0, split_dir=1)
    assert "crop=iw:3240:0:'486.000*t/6.000'[cd0_0]" in fwd          # band0 down
    assert "crop=iw:3240:0:'486.000-486.000*t/6.000'[cd1_0]" in fwd  # band1 up
    assert "[cd0_0]scale=-2:1080:flags=lanczos[vp0_0]" in fwd        # downscale
    rev = _blur_bg_chain(0, 1920, 1080, 40.0, columns=2, dur=6.0, split_dir=-1)
    assert "crop=iw:3240:0:'486.000-486.000*t/6.000'[cd0_0]" in rev  # band0 up
    assert "crop=iw:3240:0:'486.000*t/6.000'[cd1_0]" in rev          # band1 down


def test_blur_chain_split_supersamples_for_smoothness():
    """A higher supersample factor multiplies the pan travel (finer sub-pixel
    steps) without changing the final viewport height (still h)."""
    chain = _blur_bg_chain(0, 1920, 1080, 40.0, columns=2, dur=6.0,
                           split_dir=1, split_ss=4.0)
    # vh=round(1080*4)=4320, bh=round(1080*1.15*4)=4968, travel=648
    assert "crop=iw:4320:0:'648.000*t/6.000'[cd0_0]" in chain
    assert "[cd0_0]scale=-2:1080:flags=lanczos[vp0_0]" in chain


def test_blur_chain_lateral_uses_resolver_travel():
    """pan_left/pan_right travel comes from the resolver's pan_x_px, which
    already includes PAN_TRAVEL_FRACTION -- multiplying it a second time made
    the slide imperceptible (54px instead of 154px). Same for the vertical
    reveal: a supplied pan_y_px wins over the derived fallback."""
    right = _blur_bg_chain(0, 1920, 1080, 40.0, dur=2.0, kind="pan_right",
                           pan_x=154.0, pan_frac=1.2, pan_travel_frac=0.35)
    assert "x='(W-w)/2-154.000+308.000*t/2.000'" in right
    left = _blur_bg_chain(0, 1920, 1080, 40.0, dur=2.0, kind="pan_left",
                          pan_x=-154.0, pan_frac=1.2, pan_travel_frac=0.35)
    assert "x='(W-w)/2+154.000-308.000*t/2.000'" in left
    down = _blur_bg_chain(0, 1920, 1080, 40.0, dur=2.0, kind="pan_down",
                          pan_y=454.0, pan_frac=1.2, pan_travel_frac=0.35)
    assert "y='(H-h)/2+454.000-454.000*t/2.000'" in down
    # Legacy timeline entry with no per-panel travel keeps the derived form.
    legacy = _blur_bg_chain(0, 1920, 1080, 40.0, dur=2.0, kind="pan_down",
                            pan_frac=1.2, pan_travel_frac=0.35)
    assert "y='(H-h)/2+453.600-453.600*t/2.000'" in legacy


def test_build_command_super_tall_uses_split():
    from adapters.render_ffmpeg import StyleConfig
    tl = _tl_entry(motion={"preset": "reference_v1", "zoom_strength": 0.0,
                           "pan_x_px": 0.0, "pan_y_px": 0.0,
                           "tall_panel": True, "panel_scale": 1.0,
                           "split_columns": 2, "split_dir": 1}, kind="static")
    cmd = build_command(tl, Path("out.mp4"), style=StyleConfig())
    fc = cmd[cmd.index("-filter_complex") + 1]
    # canvas 1080x1920, 3x ss -> viewport 5760, band 6624, travel 864, gap 54
    assert "hstack" not in fc
    assert "crop=iw:5760:0:'864.000*t/2.000'" in fc
    assert "(W-(2*w+54))" in fc
    # the 1.2x pan and the zoom cycle must NOT appear for a split shot
    assert "(1+0.2" not in fc
    assert "(1+0.35*t" not in fc


def _split_timeline(tmp_path: Path, n: int) -> TimelineArtifact:
    """Build a reference timeline of `n` super-tall (5:1) panels -- every one
    splits, so the split_dir alternation across split panels is observable."""
    panels = []
    y = 0
    for i in range(1, n + 1):
        h = 4000
        panels.append(_panel(i, y, y + h, f"Panel {i} narration here."))
        Image.new("RGB", (800, h), "white").save(
            tmp_path / f"panel_{i:03d}.png")
        y += h
    art = CutArtifact(source="strip.png", width=800, height=y,
                      plan_hash="x", config={}, panels=panels)
    narration = NarrationArtifact(
        meta=_meta(), mode="narrator",
        entries=[NarrationEntry(id=p.id, panel_id=p.id, order=i + 1,
                                text=f"Panel {i + 1} narration here.")
                 for i, p in enumerate(panels)])
    audio = AudioArtifact(meta=_meta(), voice="none", entries=[])
    cfg = rv.VideoConfig(tts="none", motion_preset="reference")
    return rv.build_timeline(art, tmp_path, narration, audio, tmp_path, cfg,
                             panels_hash="x")


def test_ref_timeline_alternates_split_dir(tmp_path: Path):
    tl = _split_timeline(tmp_path, 4)
    assert all(e.motion["split_columns"] == 2 for e in tl.entries)
    # +1, -1, +1, -1 across consecutive split panels
    assert [e.motion["split_dir"] for e in tl.entries] == [1, -1, 1, -1]


def test_build_command_uses_per_panel_motion():
    from adapters.render_ffmpeg import StyleConfig
    tl = _tl_entry(motion={"zoom_strength": 0.58, "pan_x_px": -12.0,
                           "pan_y_px": 4.0}, kind="pan_left")
    cmd = build_command(tl, Path("out.mp4"), style=StyleConfig())
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "max(1080/iw,1920/ih)*(1+0.58)" in fc
    assert "(-12)*t/2.000" in fc


def test_build_command_without_motion_uses_global_style():
    from adapters.render_ffmpeg import StyleConfig
    cmd = build_command(_tl_entry(None), Path("out.mp4"),
                        style=StyleConfig(zoom_strength=0.8))
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "max(1080/iw,1920/ih)*(1+0.8)" in fc
    assert fc.count("overlay=x=(W-w)/2:y=(H-h)/2") == 1


def test_cover_preset_never_pads_black(tmp_path: Path):
    # Consistent cover path: timeline built with blur OFF renders with blur
    # OFF. Preset travels are clamped to the cover overflow, so the legacy
    # pad=black fallback must never trigger (no empty borders).
    panels = []
    y = 0
    for i in range(1, 4):
        h = 1200
        panels.append(_panel(i, y, y + h, f"Panel {i} narration here."))
        Image.new("RGB", (800, h), "white").save(
            tmp_path / f"panel_{i:03d}.png")
        y += h
    art = CutArtifact(source="strip.png", width=800, height=y,
                      plan_hash="x", config={}, panels=panels)
    narration = NarrationArtifact(
        meta=_meta(), mode="narrator",
        entries=[NarrationEntry(id=p.id, panel_id=p.id, order=i + 1,
                                text=f"Panel {i + 1} narration here.")
                 for i, p in enumerate(panels)])
    audio = AudioArtifact(meta=_meta(), voice="none", entries=[])
    cfg = rv.VideoConfig(tts="none", motion_preset="reference",
                         blur_background=False)
    tl = rv.build_timeline(art, tmp_path, narration, audio, tmp_path, cfg,
                           panels_hash="x")
    from adapters.render_ffmpeg import StyleConfig
    cmd = build_command(tl, Path("out.mp4"),
                        style=StyleConfig(blur_background=False,
                                          vignette=False))
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "color=black" not in fc


# ------------------------------------------------------------------ cache
def test_motion_changes_hash_but_not_essentials():
    a = rv.VideoConfig(motion_preset="none")
    b = rv.VideoConfig(motion_preset="reference")
    assert a.hash() != b.hash()
    assert a.hash_essentials() == b.hash_essentials()
    c = rv.VideoConfig(motion_preset="reference", motion_strength=0.5)
    assert b.hash() != c.hash()
    assert b.hash_essentials() == c.hash_essentials()


def test_legacy_timeline_without_motion_still_validates():
    raw = {
        "meta": {"schema_version": 1, "generator": "t", "config_hash": "h",
                 "input_hashes": {}},
        "width": 1080, "height": 1920, "fps": 30, "gap_seconds": 0.35,
        "min_display_seconds": 2.0,
        "entries": [{
            "panel_id": "001", "order": 1, "source_image": "p.png",
            "bbox": {"x": 0, "y": 0, "w": 100, "h": 200},
            "start_seconds": 0.0, "duration_seconds": 2.0,
            "audio_path": None,
            "pan": {"kind": "static", "scaled_w": 1080,
                    "scaled_h": 1920, "travel_px": 0}}],
        "skipped_panels": [],
    }
    tl = TimelineArtifact.model_validate(raw)
    assert tl.entries[0].motion is None
