# tests/test_recap_video.py
"""Offline tests for recap_video.py — plumbing only, no ffmpeg / TTS."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

import recap_video as rv
from adapters.render_ffmpeg import build_command
from adapters.schemas import AudioArtifact, AudioEntry, Meta
from guided_cutter import CutArtifact, CutPanel


# ----------------------------------------------------------------- fixtures --
def _panel(i: int, y0: int, y1: int, narration: str = "",
           dialogue: str = "") -> CutPanel:
    return CutPanel(id=f"{i:03d}", panel_index=i, y_start=y0, y_end=y1,
                    narration=narration, dialogue=dialogue,
                    panel_type="action", confidence=0.9,
                    image_file=f"panel_{i:03d}.png")


@pytest.fixture()
def cut_dir(tmp_path: Path) -> tuple[Path, CutArtifact]:
    panels = [
        _panel(1, 0, 1200, "Jin wakes up in the dungeon.", '"Where am I?"'),
        _panel(2, 1200, 4800, "He runs down an endless corridor."),
        _panel(3, 4800, 5400, "", ""),  # silent panel
    ]
    for p in panels:
        Image.new("RGB", (800, p.y_end - p.y_start), "white").save(
            tmp_path / p.image_file)
    art = CutArtifact(source="strip.png", width=800, height=5400,
                      plan_hash="x", config={}, panels=panels)
    (tmp_path / "panels.json").write_text(art.model_dump_json(), "utf-8")
    return tmp_path, art


def _meta() -> Meta:
    return Meta(schema_version=1, generator="t", config_hash="c",
                input_hashes={})


# ------------------------------------------------------------- script text --
def test_script_text_appends_dialogue_once():
    p = _panel(1, 0, 10, "Jin gasps.", '"Where am I?"')
    assert rv.script_text(p) == 'Jin gasps. "Where am I?"'
    # dialogue already quoted by narrator -> not repeated
    p2 = _panel(2, 0, 10, 'Jin gasps, "Where am I?"', '"Where am I?"')
    assert rv.script_text(p2) == 'Jin gasps, "Where am I?"'
    assert rv.script_text(p, include_dialogue=False) == "Jin gasps."


def test_script_text_adds_terminal_punctuation():
    assert rv.script_text(_panel(1, 0, 1, "he runs")) == "he runs."


# ---------------------------------------------------------------------- pan --
def test_compute_pan_tall_panel_pans_down():
    pan = rv.compute_pan(800, 3600)
    assert pan.kind == "pan_down"
    assert pan.scaled_w == 1080
    assert pan.scaled_h == 4860
    assert pan.travel_px == 4860 - 1920


def test_compute_pan_wide_panel_pans_right():
    pan = rv.compute_pan(4000, 1000)
    assert pan.kind == "pan_right"
    assert pan.scaled_h == 1920
    assert pan.travel_px == pan.scaled_w - 1080 > 0


def test_compute_pan_exact_fit_is_static():
    pan = rv.compute_pan(1080, 1920)
    assert pan.kind == "static" and pan.travel_px == 0
    with pytest.raises(ValueError):
        rv.compute_pan(0, 10)


def test_compute_pan_small_panel_keeps_uniform_scale():
    """A tiny panel (e.g. 100x100) with the 4x cap must NOT be stretched
    to 1080x1920 — it should retain uniform scale and be padded, not distorted."""
    pan = rv.compute_pan(100, 100)
    assert pan.kind == "static"
    assert pan.travel_px == 0
    assert pan.scaled_w == pan.scaled_h  # uniform
    assert pan.scaled_w == 400  # 100 * 4.0 cap
    assert pan.scaled_w < rv.WIDTH or pan.scaled_h < rv.HEIGHT  # will be padded


# ---------------------------------------------------------------- duration --
def test_display_seconds_rules():
    cfg = rv.VideoConfig(gap_seconds=0.35, min_display_seconds=2.0,
                         max_pan_px_per_sec=450, silent_wpm=120)
    # spoken: audio + gap, never below min
    assert rv.display_seconds(audio_seconds=0.5, words=3, travel_px=0,
                              cfg=cfg) == 2.0
    assert rv.display_seconds(audio_seconds=5.0, words=3, travel_px=0,
                              cfg=cfg) == 5.35
    # pan floor wins when travel is large
    assert rv.display_seconds(audio_seconds=1.0, words=3, travel_px=4500,
                              cfg=cfg) == 10.0
    # silent: reading speed, capped
    assert rv.display_seconds(audio_seconds=None, words=120, travel_px=0,
                              cfg=cfg) == 12.0
    assert rv.display_seconds(audio_seconds=None, words=0, travel_px=0,
                              cfg=cfg) == 2.0


# ---------------------------------------------------------------- timeline --
def test_timeline_is_contiguous_and_silent_mode(cut_dir):
    d, art = cut_dir
    cfg = rv.VideoConfig(tts="none")
    nar = rv.build_narration(art, cfg, panels_hash="h")
    aud = rv.synthesize_audio(nar, d / "audio", cfg)
    assert aud.entries == [] and aud.voice == "none"
    tl = rv.build_timeline(art, d, nar, aud, d / "audio", cfg, panels_hash="h")
    assert [e.panel_id for e in tl.entries] == ["001", "002", "003"]
    t = 0.0
    for e in tl.entries:
        assert e.start_seconds == pytest.approx(t, abs=1e-3)
        assert e.audio_path is None
        assert Path(e.source_image).is_file()
        t += e.duration_seconds
    assert rv.total_seconds(tl) == pytest.approx(t, abs=1e-3)
    assert tl.entries[1].pan.kind == "pan_down"       # 800x3600 panel
    # empty panel: min display, unless its pan needs longer to stay readable
    p3 = tl.entries[2]
    assert p3.duration_seconds == pytest.approx(
        max(cfg.min_display_seconds, p3.pan.travel_px / cfg.max_pan_px_per_sec),
        abs=1e-3)


def test_timeline_uses_measured_audio(cut_dir):
    d, art = cut_dir
    cfg = rv.VideoConfig()
    nar = rv.build_narration(art, cfg, panels_hash="h")
    aud = AudioArtifact(meta=_meta(), voice="v", entries=[
        AudioEntry(entry_id="001", path="001.mp3", duration_seconds=3.2,
                   words=[{"start": 0.0, "end": 0.4, "text": "Jin"},
                          {"start": 0.4, "end": 0.9, "text": "wakes"}]),
        AudioEntry(entry_id="002", path="002.mp3", duration_seconds=1.0),
    ])
    tl = rv.build_timeline(art, d, nar, aud, d / "audio", cfg, panels_hash="h")
    assert tl.entries[0].duration_seconds == pytest.approx(3.55)
    assert tl.entries[0].audio_path.endswith("001.mp3")
    # panel 2: 1.0s audio but 2940px of pan -> pan floor 6.533s
    assert tl.entries[1].duration_seconds == pytest.approx(2940 / 450, abs=1e-3)


def test_missing_panel_image_is_skipped_not_an_error(cut_dir):
    """A panel whose PNG is missing must be SKIPPED (with a warning) instead
    of raising, so that one missing file (e.g. a stale panels.json from a
    previous run, or a panel cut that was dropped as too thin) does not kill
    the whole timeline build."""
    d, art = cut_dir
    assert len(art.panels) >= 2, "fixture must have at least 2 panels"
    (d / "panel_002.png").unlink()
    cfg = rv.VideoConfig(tts="none")
    nar = rv.build_narration(art, cfg, panels_hash="h")
    aud = rv.synthesize_audio(nar, d / "audio", cfg)
    tl = rv.build_timeline(art, d, nar, aud, d / "audio", cfg, panels_hash="h")
    # Only the present panel (001) is in the timeline; 002 was skipped.
    panel_ids = [e.panel_id for e in tl.entries]
    assert "002" not in panel_ids
    assert any(pid.endswith("001") for pid in panel_ids)


# --------------------------------------------------------------------- srt --
def test_srt_time_format():
    assert rv.srt_time(0) == "00:00:00,000"
    assert rv.srt_time(61.5) == "00:01:01,500"
    assert rv.srt_time(3723.004) == "01:02:03,004"


def test_write_srt_uses_word_timings(cut_dir, tmp_path):
    d, art = cut_dir
    cfg = rv.VideoConfig()
    nar = rv.build_narration(art, cfg, panels_hash="h")
    words = [{"start": i * 0.3, "end": i * 0.3 + 0.25, "text": w}
             for i, w in enumerate(["Jin", "wakes", "up", "in", "the", "dungeon", "Where", "am", "I"])]
    aud = AudioArtifact(meta=_meta(), voice="v", entries=[
        AudioEntry(entry_id="001", path="001.mp3", duration_seconds=3.0,
                   words=words)])
    tl = rv.build_timeline(art, d, nar, aud, d / "audio", cfg, panels_hash="h")
    out = tmp_path / "recap.srt"
    n = rv.write_srt(tl, nar, aud, out)
    text = out.read_text("utf-8")
    # panel 1: 9 words -> one word-timed cue; panel 2: one cue over the whole
    # (un-voiced) panel; panel 3: empty narration -> no cue
    assert n == 2
    assert text.startswith("1\n00:00:00,000 --> ")
    assert "He runs down an endless corridor." in text
    assert text.count("-->") == 2


# ------------------------------------------------------------- ffmpeg cmd --
def test_render_command_builds_without_ffmpeg(cut_dir):
    d, art = cut_dir
    cfg = rv.VideoConfig(tts="none")
    nar = rv.build_narration(art, cfg, panels_hash="h")
    aud = rv.synthesize_audio(nar, d / "audio", cfg)
    tl = rv.build_timeline(art, d, nar, aud, d / "audio", cfg, panels_hash="h")
    cmd = build_command(tl, d / "recap.mp4")
    assert cmd[0] == "ffmpeg" and "-filter_complex" in cmd
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "concat=n=3:v=1:a=0" in fc
    assert "anullsrc" in " ".join(cmd)          # silent panels get silence
    assert "loudnorm" not in fc                 # no audio -> no loudnorm


# ------------------------------------------------------------ orchestrator --
def test_make_recap_video_dry_run_writes_sidecars(cut_dir):
    d, art = cut_dir
    summary = rv.make_recap_video(d / "panels.json", d / "recap.mp4",
                                  rv.VideoConfig(tts="none"), dry_run=True)
    assert summary["video"] is None
    assert summary["panels"] == 3
    assert (d / "timeline.json").is_file()
    assert (d / "narration.json").is_file()
    assert (d / "recap.srt").is_file()
    tl = json.loads((d / "timeline.json").read_text("utf-8"))
    assert tl["width"] == 1080 and tl["height"] == 1920
    assert len(tl["entries"]) == 3
