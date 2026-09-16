# tests/test_narration_quality.py
"""Quality-contract tests for the narration/video fixes:

  * script.json (Phase 2.5) is the PREFERRED narration source, mapped by
    panel_id; panels without a line get silent entries (dropped later by
    the timeline unless audio exists)
  * consecutive duplicate text is never spoken twice
  * non-lexical strings ("...", "—") never reach script_text/TTS
  * dead-air panels (no text, no audio) are dropped from the timeline
    with an auditable skip record
  * pacing varies by panel class: action cuts below min_display,
    reveal holds a beat
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

import recap_video as rv
from guided_cutter import CutArtifact, CutPanel


def _panel(i: int, y0: int, y1: int, narration: str = "",
           dialogue: str = "", **kw) -> CutPanel:
    return CutPanel(id=f"panel_{i:03d}", panel_index=i, y_start=y0,
                    y_end=y1, narration=narration, dialogue=dialogue,
                    panel_type="panel", confidence=0.9,
                    image_file=f"panel_{i:03d}.png", **kw)


def _artifact(panels: list[CutPanel]) -> CutArtifact:
    return CutArtifact(source="strip.png", width=800,
                       height=panels[-1].y_end, plan_hash="x",
                       config={}, panels=panels)


@pytest.fixture()
def session(tmp_path: Path) -> Path:
    panels = [
        _panel(1, 0, 800, "Jin faces the door.", '"Where am I?"'),
        _panel(2, 800, 1600, "He charges the beast with a fist."),
        _panel(3, 1600, 2400, "The secret awakens at last."),
        _panel(4, 2400, 3200, "..."),            # non-lexical
        _panel(5, 3200, 4000, ""),                # empty caption
    ]
    art = _artifact(panels)
    (tmp_path / "panels.json").write_text(art.model_dump_json(), "utf-8")
    for p in panels:
        Image.new("RGB", (800, 800), "white").save(
            tmp_path / p.image_file)
    return tmp_path


# ------------------------------------------------------- script.json preference --
def test_build_narration_prefers_script_json(session):
    d = session
    art = CutArtifact.model_validate_json(
        (d / "panels.json").read_text("utf-8"))
    cfg = rv.VideoConfig()
    # no script.json yet -> per-panel captions
    nar = rv.build_narration(art, cfg, panels_hash="h", work_dir=d)
    by_id = {e.panel_id: e for e in nar.entries}
    assert by_id["panel_001"].text == 'Jin faces the door. "Where am I?"'
    # non-lexical caption -> empty text entry
    assert by_id["panel_004"].text == ""
    assert by_id["panel_005"].text == ""

    # with script.json -> mapped lines win; unmapped panels silent
    script = {
        "version": 1,
        "lines": [
            {"panel_id": "panel_001", "panel_index": 1,
             "text": "Jin was ordinary — until the door.",
             "part": "hook", "quote": None},
            {"panel_id": "panel_002", "panel_index": 2,
             "text": "He charges.",
             "part": "escalation", "quote": None},
            {"panel_id": "panel_003", "panel_index": 3,
             "text": "He charges.",          # duplicate of previous line
             "part": "escalation", "quote": None},
        ],
    }
    (d / "script.json").write_text(json.dumps(script), "utf-8")
    nar2 = rv.build_narration(art, cfg, panels_hash="h", work_dir=d)
    by_id2 = {e.panel_id: e for e in nar2.entries}
    assert by_id2["panel_001"].text == \
        "Jin was ordinary — until the door."
    assert by_id2["panel_002"].text == "He charges."
    # duplicate line never spoken twice
    assert by_id2["panel_003"].text == ""
    # unmapped panels stay silent
    assert by_id2["panel_004"].text == ""
    assert by_id2["panel_005"].text == ""


def test_script_json_quote_becomes_entry_quote(session):
    d = session
    art = CutArtifact.model_validate_json(
        (d / "panels.json").read_text("utf-8"))
    script = {
        "version": 1,
        "lines": [
            {"panel_id": "panel_001", "panel_index": 1,
             "text": "Hook line.", "part": "hook",
             "quote": "Where am I?"},
            {"panel_id": "panel_002", "panel_index": 2,
             "text": "Escalation line.", "part": "escalation",
             "quote": None},
        ],
    }
    (d / "script.json").write_text(json.dumps(script), "utf-8")
    nar = rv.build_narration(art, rv.VideoConfig(), panels_hash="h",
                             work_dir=d)
    by_id = {e.panel_id: e for e in nar.entries}
    assert by_id["panel_001"].quotes == ["Where am I?"]
    assert by_id["panel_002"].quotes == []


# ------------------------------------------------------------------ dead air --
def test_dead_air_panels_dropped_with_skip_record(session):
    d = session
    art = CutArtifact.model_validate_json(
        (d / "panels.json").read_text("utf-8"))
    cfg = rv.VideoConfig(tts="none")
    nar = rv.build_narration(art, cfg, panels_hash="h", work_dir=d)
    aud = rv.synthesize_audio(nar, d / "audio", cfg)
    tl = rv.build_timeline(art, d, nar, aud, d / "audio", cfg,
                           panels_hash="h")
    # 001/002/003 have text; 004 (non-lexical) + 005 (empty) are dead air
    assert [e.panel_id for e in tl.entries] == \
        ["panel_001", "panel_002", "panel_003"]
    reasons = {s["panel_id"]: s["reason"] for s in tl.skipped_panels}
    assert reasons["panel_004"] == "no_text_no_audio"
    assert reasons["panel_005"] == "no_text_no_audio"


def test_panel_with_audio_but_no_text_stays(session, tmp_path):
    """A panel holding a still-relevant audio clip (e.g. TTS cached from an
    edited script line) is NOT dead air even when its text entry is
    empty — audio presence keeps the frame."""
    from adapters.schemas import AudioArtifact, AudioEntry, Meta

    d = session
    art = CutArtifact.model_validate_json(
        (d / "panels.json").read_text("utf-8"))
    cfg = rv.VideoConfig(tts="none")
    nar = rv.build_narration(art, cfg, panels_hash="h", work_dir=d)
    aud = AudioArtifact(
        meta=Meta(schema_version=1, generator="t", config_hash="c",
                  input_hashes={}), voice="v",
        entries=[AudioEntry(entry_id="panel_004", path="panel_004.mp3",
                            duration_seconds=2.5)])
    (d / "audio" / "panel_004.mp3").parent.mkdir(exist_ok=True)
    (d / "audio" / "panel_004.mp3").write_bytes(b"")
    tl = rv.build_timeline(art, d, nar, aud, d / "audio", cfg,
                           panels_hash="h")
    ids = [e.panel_id for e in tl.entries]
    assert "panel_004" in ids        # audio keeps it
    assert "panel_005" not in ids    # no audio, no text -> dropped


# ---------------------------------------------------------------- pacing --
class TestPacingByClass:
    def test_action_cuts_below_min_display(self):
        cfg = rv.VideoConfig(min_display_seconds=2.0)
        dur = rv.display_seconds(audio_seconds=0.5, words=3,
                                 travel_px=0, cfg=cfg,
                                 panel_class="action")
        assert dur < 2.0            # below the calm floor
        assert dur >= cfg.action_floor_seconds

    def test_calm_keeps_min_display(self):
        cfg = rv.VideoConfig(min_display_seconds=2.0)
        dur = rv.display_seconds(audio_seconds=0.5, words=3,
                                  travel_px=0, cfg=cfg,
                                  panel_class="calm")
        assert dur == 2.0

    def test_reveal_holds_a_beat(self):
        cfg = rv.VideoConfig(min_display_seconds=2.0)
        calm = rv.display_seconds(audio_seconds=1.0, words=3,
                                   travel_px=0, cfg=cfg,
                                   panel_class="calm")
        reveal = rv.display_seconds(audio_seconds=1.0, words=3,
                                     travel_px=0, cfg=cfg,
                                     panel_class="reveal")
        assert reveal == pytest.approx(calm * 1.35, abs=0.01)

    def test_pan_floor_always_applies(self):
        cfg = rv.VideoConfig(min_display_seconds=2.0)
        dur = rv.display_seconds(audio_seconds=0.5, words=3,
                                 travel_px=4500, cfg=cfg,
                                 panel_class="action")
        assert dur >= 4500 / cfg.max_pan_px_per_sec

    def test_pan_fit_speech_speeds_pan_into_speech_window(self):
        # 4500px at 450 px/s = 10s pan, but only 5.35s of speech+gap.
        # Opt-in speed-up (900 px/s => 5.0s) lands the pan inside the
        # speech window, so the panel does not outlast the narration.
        cfg = rv.VideoConfig(min_display_seconds=2.0, pan_fit_speech=True)
        dur = rv.display_seconds(audio_seconds=5.0, words=3,
                                 travel_px=4500, cfg=cfg,
                                 panel_class="calm")
        assert dur == pytest.approx(5.0 + cfg.gap_seconds, abs=0.01)

    def test_pan_fit_speech_bounds_extreme_travel(self):
        # 45000px / 900 = 50s: even 2x speed cannot fit the speech window,
        # so the bounded pan floor still applies (readable, not instant).
        cfg = rv.VideoConfig(min_display_seconds=2.0, pan_fit_speech=True)
        dur = rv.display_seconds(audio_seconds=5.0, words=3,
                                 travel_px=45000, cfg=cfg,
                                 panel_class="calm")
        assert dur >= 45000 / (cfg.max_pan_px_per_sec
                               * cfg.pan_fit_speech_speedup)

    def test_pan_fit_speech_off_by_default(self):
        # Default behaviour unchanged: the classic pan floor always wins.
        cfg = rv.VideoConfig(min_display_seconds=2.0)
        dur = rv.display_seconds(audio_seconds=5.0, words=3,
                                 travel_px=4500, cfg=cfg,
                                 panel_class="calm")
        assert dur >= 4500 / cfg.max_pan_px_per_sec

    def test_pan_fit_speech_changes_config_hash(self):
        assert rv.VideoConfig().hash() != rv.VideoConfig(
            pan_fit_speech=True).hash()

    def test_class_config_hash_changes(self):
        base = rv.VideoConfig()
        tuned = rv.VideoConfig(class_duration_multiplier={
            "action": 0.9, "reveal": 1.5, "dialogue": 1.0, "calm": 1.0})
        assert base.hash() != tuned.hash()


# ---------------------------------------------------------- classification --
def test_build_timeline_classifies_panels(session):
    d = session
    art = CutArtifact.model_validate_json(
        (d / "panels.json").read_text("utf-8"))
    # action text on panel 2 ("charges... fist") -> action class affects
    # duration even though narration is short
    cfg = rv.VideoConfig(tts="none")
    nar = rv.build_narration(art, cfg, panels_hash="h", work_dir=d)
    aud = rv.synthesize_audio(nar, d / "audio", cfg)
    tl = rv.build_timeline(art, d, nar, aud, d / "audio", cfg,
                           panels_hash="h")
    by_id = {e.panel_id: e for e in tl.entries}
    # panel 2 is action-classified (regex from cinematic_effects)
    dur2 = by_id["panel_002"].duration_seconds
    assert dur2 > 0
    # reveal panel 3 ("awakens") gets the held-beat multiplier
    dur3 = by_id["panel_003"].duration_seconds
    words3 = rv._word_count("The secret awakens at last.")
    read = (words3 / cfg.silent_wpm) * 60.0
    assert dur3 >= (read + cfg.gap_seconds) * 1.35 - 0.01


# ------------------------------------------------------------ non-lexical --
class TestNonLexical:
    def test_script_text_drops_ellipsis(self):
        p = _panel(1, 0, 10, "...", '"Wait."')
        assert rv.script_text(p) == '"Wait."'

    def test_script_text_dashes_and_empty(self):
        assert rv.script_text(_panel(1, 0, 10, "—")) == ""
        assert rv.script_text(_panel(1, 0, 10, "")) == ""

    def test_narrator_join_filters_non_lexical(self):
        from narrator import make_script_from_cut
        panels = [
            _panel(1, 0, 400, "..."),
            _panel(2, 400, 800, "Jin runs."),
        ]
        art = _artifact(panels)
        assert make_script_from_cut(art) == "Jin runs."


# ------------------------------------------------- script.json precedence edge --
def test_stale_script_json_ignored_when_no_lines(session):
    d = session
    art = CutArtifact.model_validate_json(
        (d / "panels.json").read_text("utf-8"))
    (d / "script.json").write_text(json.dumps(
        {"version": 1, "lines": []}), "utf-8")     # empty lines
    nar = rv.build_narration(art, rv.VideoConfig(), panels_hash="h",
                             work_dir=d)
    # falls back to per-panel captions
    by_id = {e.panel_id: e for e in nar.entries}
    assert by_id["panel_001"].text == 'Jin faces the door. "Where am I?"'
