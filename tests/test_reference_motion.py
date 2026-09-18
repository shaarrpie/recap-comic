"""Offline tests for the reference-motion preset (no ffmpeg / TTS)."""
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
def test_preset_loads_14_ordered_segments():
    p = _preset()
    assert len(p) == 14
    assert [s.seg for s in (p.segments or [])] == list(range(1, 15))
    assert abs(p.total_duration - 27.23) < 0.05
    assert p.reference_width == pytest.approx(1312.0)


def test_normalized_movement_uses_reference_dims():
    p = _preset()
    seg1 = (p.segments or [])[0]
    n = mp.normalized_movement(seg1, p)
    assert n["ndx"] == pytest.approx(0.0)
    assert n["ndy"] == pytest.approx(104.0 / p.reference_height)
    assert n["ndyps"] == pytest.approx(58.9 / p.reference_height)


def test_static_vs_moving_classification():
    p = _preset()
    segs = {s.seg: s for s in (p.segments or [])}
    assert mp.is_static(segs[4], p) is True  # zoom-in hold from the request
    assert mp.is_static(segs[1], p) is False  # large vertical move
    assert mp.is_static(segs[12], p) is True


def test_low_confidence_damps_but_keeps_motion():
    p = _preset()
    segs = {s.seg: s for s in (p.segments or [])}
    assert mp.confidence_damping(segs[1], p) == pytest.approx(1.0)
    assert mp.confidence_damping(segs[6], p) == pytest.approx(0.6)
    assert mp.confidence_damping(segs[10], p) == pytest.approx(0.6)


def test_zoom_strength_preserves_ordering():
    p = _preset()
    segs = {s.seg: s for s in (p.segments or [])}
    z_out = mp.zoom_strength_for_seg(segs[2])  # 0.92
    z_norm = mp.zoom_strength_for_seg(segs[1])  # 1.00
    z_in = mp.zoom_strength_for_seg(segs[4])  # 1.14
    assert z_out < z_norm < z_in
    assert 0.0 <= z_out <= 0.8 and 0.0 <= z_in <= 0.8


def test_pan_kind_sign_convention():
    p = _preset()
    segs = {s.seg: s for s in (p.segments or [])}
    assert mp.pan_kind_for_seg(segs[1], p) == "pan_down"  # dy>0
    assert mp.pan_kind_for_seg(segs[3], p) == "pan_up"  # dy<0
    assert mp.pan_kind_for_seg(segs[4], p) == "zoom_in"  # static + zoom
    assert mp.pan_kind_for_seg(segs[2], p) == "zoom_out"  # static-ish + zoom


def test_mapping_cycles_and_spans():
    p = _preset()
    assert mp.map_segments_to_panels(14, p) == list(range(14))
    cycled = mp.map_segments_to_panels(16, p)
    assert cycled[:14] == list(range(14))
    assert cycled[14:] == [0, 1]  # rhythm repeats, order preserved
    few = mp.map_segments_to_panels(3, p)
    assert few[0] == 0 and few[-1] == 13  # spans first..last, no random cut
    assert few == sorted(few)


# ------------------------------------------------------------------ resolve
def test_blur_resolve_never_reveals_empty_and_keeps_direction():
    p = _preset()
    segs = {s.seg: s for s in (p.segments or [])}
    r = mp.resolve_for_panel(png_w=390, png_h=800, canvas_w=1080,
                             canvas_h=1920, seg=segs[1], preset=p,
                             blur_background=True)
    assert r["kind"] == "pan_down"
    assert r["pan_y_px"] > 0  # downward preserved
    assert abs(r["pan_y_px"]) <= 1080 * 0.08 + 1e-6 or \
        abs(r["pan_y_px"]) <= 1920 * 0.08 + 1e-6
    assert r["ndy"] == pytest.approx(104.0 / p.reference_height)


def test_blur_static_seg_has_no_pan_but_keeps_zoom():
    p = _preset()
    segs = {s.seg: s for s in (p.segments or [])}
    r = mp.resolve_for_panel(png_w=390, png_h=800, canvas_w=1080,
                             canvas_h=1920, seg=segs[4], preset=p,
                             blur_background=True)
    assert r["static"] is True
    assert r["pan_x_px"] == 0.0 and r["pan_y_px"] == 0.0
    assert r["zoom_strength"] > mp.zoom_strength_for_seg(segs[1])


def test_cover_resolve_clamps_to_overflow():
    p = _preset()
    segs = {s.seg: s for s in (p.segments or [])}
    # Exact-fit panel: no overflow, so even a large dy must clamp to static.
    r = mp.resolve_for_panel(png_w=1080, png_h=1920, canvas_w=1080,
                             canvas_h=1920, seg=segs[1], preset=p,
                             blur_background=False)
    assert r["travel_px"] == 0
    assert r["scaled_w"] >= 1080 and r["scaled_h"] >= 1920
    # Tall panel: travel fits inside the overflow, direction kept.
    r2 = mp.resolve_for_panel(png_w=800, png_h=3600, canvas_w=1080,
                              canvas_h=1920, seg=segs[1], preset=p,
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


def test_timeline_threads_preset_and_preserves_rhythm(tmp_path: Path):
    tl = _timeline_with_preset(tmp_path, 14)
    assert len(tl.entries) == 14
    kinds = [e.pan.kind for e in tl.entries]
    # Exact reference sequence (not simplified to zoom->pan->zoom->pan).
    assert kinds[0] == "pan_down"
    assert kinds[2] == "pan_up"
    assert kinds[3] == "zoom_in"
    assert kinds[4] == "pan_down"
    # Short reference shots stay short relative to long ones.
    durs = [e.duration_seconds for e in tl.entries]
    assert durs[1] < durs[5]  # seg2 (1.07s) < seg6 (3.07s)
    for e in tl.entries:
        assert e.motion is not None
        assert {"seg", "zoom", "dx", "dy", "ndx", "ndy",
                "confidence"}.issubset(set(e.motion.keys()))


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
                           rv.VideoConfig(tts="none"), panels_hash="x")
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
    a = rv.VideoConfig()
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
