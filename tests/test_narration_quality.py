# tests/test_narration_quality.py
"""Quality-contract tests for the narration/video fixes:

  * script.json (Phase 2.5) is the PREFERRED narration source, mapped by
    panel_id; panels without a line get silent entries
  * consecutive duplicate text is never spoken twice
  * non-lexical strings ("...", "—") never reach script_text/TTS
  * un-narrated panels (no text, no audio) stay in the timeline as short
    silent beats (min_silent) with an auditable record — dropping them
    made the video visibly skip most of the chapter's panels
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


def test_script_json_found_beside_panels_when_output_dir_differs(tmp_path, session):
    # Regression: `guided video DIR/panels.json --out elsewhere/x.mp4` puts the
    # persona script beside the cut, NOT in the output folder. build_narration
    # must still find it via panels_dir instead of silently reverting to the
    # flat per-panel caption.
    d = session
    art = CutArtifact.model_validate_json(
        (d / "panels.json").read_text("utf-8"))
    (d / "script.json").write_text(json.dumps({
        "version": 3,
        "lines": [{"panel_id": "panel_001", "panel_index": 1,
                   "text": "Persona hook line.", "part": "hook",
                   "quote": None}],
    }), "utf-8")
    out_dir = tmp_path / "elsewhere"
    out_dir.mkdir()
    cfg = rv.VideoConfig()
    # work_dir (output) has no script.json; panels_dir (cut) does.
    nar = rv.build_narration(art, cfg, panels_hash="h",
                             work_dir=out_dir, panels_dir=d)
    by_id = {e.panel_id: e for e in nar.entries}
    assert by_id["panel_001"].text == "Persona hook line."
    # Without the panels_dir fallback the same call reverts to the caption.
    nar2 = rv.build_narration(art, cfg, panels_hash="h", work_dir=out_dir)
    by_id2 = {e.panel_id: e for e in nar2.entries}
    assert by_id2["panel_001"].text != "Persona hook line."


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
def test_spoken_lines_are_stripped_of_foreign_text_and_emoji(tmp_path):
    """The last gate before TTS must be English-only.

    A script written before the v6 contract carries raw Hangul syllables
    (U+D130) and even an emoji (U+1F3B5) inside spoken lines, and edge-tts
    reads those literally. Cleaning the prompt only guards what goes INTO the
    model, so the outgoing line is scrubbed here instead -- which fixes already
    narrated chapters without re-spending a single vision call.
    """
    panels = [_panel(1, 0, 800, "Jin faces the door."),
              _panel(2, 800, 1600, "He strikes the ground.")]
    art = _artifact(panels)
    (tmp_path / "panels.json").write_text(art.model_dump_json(), "utf-8")
    for p in panels:
        Image.new("RGB", (800, 800), "white").save(tmp_path / p.image_file)
    (tmp_path / "script.json").write_text(json.dumps({
        "version": 1,
        "lines": [
            {"panel_id": "panel_001", "panel_index": 1,
             "text": "\ud55c\ub9d0 DID YOU HEAR ABOUT THAT? YOU LEFT OUT THE "
                     "SUBJECT.",
             "part": "hook", "quote": "\ub41c!"},
            {"panel_id": "panel_002", "panel_index": 2,
             "text": "A bright strip shows \U0001F3B5 music over the hall.",
             "part": "cliffhanger", "quote": None},
        ],
    }), "utf-8")

    nar = rv.build_narration(art, rv.VideoConfig(), panels_hash="h",
                             work_dir=tmp_path, panels_dir=tmp_path)
    for e in nar.entries:
        assert all(ord(c) < 128 for c in e.text), e.text
        for q in e.quotes:
            assert all(ord(c) < 128 for c in q), q
    blob = " ".join(e.text for e in nar.entries)
    # the English words survive; only the foreign characters go
    assert "DID YOU HEAR ABOUT THAT?" in blob
    assert "music over the hall" in blob
    # a quote that was nothing but foreign text is dropped, not spoken
    assert all(e.quotes == [] for e in nar.entries)


def test_folded_voice_over_bubbles_are_also_english_only(tmp_path):
    """A context_only panel folds its bubble onto the NEIGHBOURING entry, which
    bypasses the spoken-line gate entirely. Real case: gallery chapter 5 panel
    049's untranslated Korean sound effect landed inside panel 048's spoken
    line, so the fold path needs the same strip."""
    panels = [
        _panel(1, 0, 800, "Jin faces the door."),
        _panel(2, 800, 1600, "", "\ub41c\ud55c WELL DONE", context_only=True),
        _panel(3, 1600, 2400, "", "\uc544\ubb34\ub807\ub3c4", context_only=True),
        _panel(4, 2400, 3200, "He strikes the ground."),
    ]
    art = _artifact(panels)
    (tmp_path / "panels.json").write_text(art.model_dump_json(), "utf-8")
    for p in panels:
        Image.new("RGB", (800, 800), "white").save(tmp_path / p.image_file)
    (tmp_path / "script.json").write_text(json.dumps({
        "version": 1,
        "lines": [
            {"panel_id": "panel_001", "panel_index": 1,
             "text": "Jin found the door he was never meant to open.",
             "part": "hook", "quote": None},
            {"panel_id": "panel_004", "panel_index": 4,
             "text": "So the whole hall finally shut up.",
             "part": "cliffhanger", "quote": None},
        ],
    }), "utf-8")

    nar = rv.build_narration(art, rv.VideoConfig(), panels_hash="h",
                             work_dir=tmp_path, panels_dir=tmp_path)
    blob = " ".join(e.text for e in nar.entries)
    assert all(ord(c) < 128 for c in blob), blob
    # the English half of a mixed bubble still carries over; the fully
    # untranslated one folds in as nothing at all
    assert "WELL DONE" in blob
    assert "never meant to open" in blob
    assert "finally shut up" in blob


def test_promo_card_is_never_spoken_even_as_a_voice_over(tmp_path):
    """Demoting an ad panel only removes its FRAME: context_only panels fold
    their text onto the nearest scene panel, so the narrator kept saying "Read
    at ASURASCANS.COM for the fastest releases" out loud over real art. Both
    leak routes are silenced: the folded dialogue, and an ad line the script
    pass wrote for an ordinary scene panel.
    """
    panels = [
        _panel(1, 0, 800, "Jin faces the door."),
        _panel(2, 800, 1600, "",
               "Read at ASURASCANS.COM for the fastest releases",
               context_only=True),
        _panel(3, 1600, 2400, "The beast awakens at last."),
        _panel(4, 2400, 3200, "He strikes the ground."),
    ]
    art = _artifact(panels)
    (tmp_path / "panels.json").write_text(art.model_dump_json(), "utf-8")
    for p in panels:
        Image.new("RGB", (800, 800), "white").save(tmp_path / p.image_file)
    (tmp_path / "script.json").write_text(json.dumps({
        "version": 1,
        "lines": [
            {"panel_id": "panel_001", "panel_index": 1,
             "text": "Jin found the door he was never meant to open.",
             "part": "hook", "quote": None},
            {"panel_id": "panel_002", "panel_index": 2,
             "text": "Read the full story at asurascans.com.",
             "part": "setup", "quote": None},
            {"panel_id": "panel_003", "panel_index": 3,
             "text": "The beast woke up hungry.",
             "part": "escalation", "quote": None},
            {"panel_id": "panel_004", "panel_index": 4,
             "text": "So hit up AsuraScans for more, folks.",
             "part": "cliffhanger", "quote": None},
        ],
    }), "utf-8")

    nar = rv.build_narration(art, rv.VideoConfig(), panels_hash="h",
                             work_dir=tmp_path, panels_dir=tmp_path)
    blob = " ".join(e.text for e in nar.entries).lower()
    assert "asurascans" not in blob
    assert ".com" not in blob
    assert "fastest releases" not in blob
    # the actual story survives: nothing is silently swallowed wholesale
    assert "door" in blob
    assert "beast" in blob


def test_dead_air_panels_become_silent_beats(session):
    """Un-narrated panels (no text, no audio) must NOT vanish from the video:
    the chapter script speaks fewer lines than there are panels, and dropping
    them made the recap visibly skip most of the chapter's art. They stay as
    short silent beats (min_silent) with an auditable record."""
    d = session
    art = CutArtifact.model_validate_json(
        (d / "panels.json").read_text("utf-8"))
    cfg = rv.VideoConfig(tts="none")
    nar = rv.build_narration(art, cfg, panels_hash="h", work_dir=d)
    aud = rv.synthesize_audio(nar, d / "audio", cfg)
    tl = rv.build_timeline(art, d, nar, aud, d / "audio", cfg,
                           panels_hash="h")
    # 001/002/003 have text; 004 (non-lexical) + 005 (empty) stay as beats
    assert [e.panel_id for e in tl.entries] == \
        ["panel_001", "panel_002", "panel_003", "panel_004", "panel_005"]
    reasons = {s["panel_id"]: s["reason"] for s in tl.skipped_panels}
    assert reasons["panel_004"] == "no_text_no_audio_silent_beat"
    assert reasons["panel_005"] == "no_text_no_audio_silent_beat"
    # the beats are short (min_silent) and carry no audio
    for pid in ("panel_004", "panel_005"):
        e = next(e for e in tl.entries if e.panel_id == pid)
        assert e.duration_seconds == pytest.approx(cfg.min_silent, abs=0.05)
        assert e.audio_path is None


def test_silent_beat_travel_stays_readable(session):
    """The closer framing (2.0/2.2x) makes a reveal ~670px, and the pan floor
    (travel / max_pan_px_per_sec) would stretch an un-narrated beat past
    min_silent -- recreating the dead air this file exists to prevent. A filler
    beat keeps its short hold, so its travel is capped to what is readable
    inside that hold.
    """
    d = session
    art = CutArtifact.model_validate_json(
        (d / "panels.json").read_text("utf-8"))
    cfg = rv.VideoConfig(tts="none")
    nar = rv.build_narration(art, cfg, panels_hash="h", work_dir=d)
    aud = rv.synthesize_audio(nar, d / "audio", cfg)
    tl = rv.build_timeline(art, d, nar, aud, d / "audio", cfg,
                           panels_hash="h")
    filler = {s["panel_id"] for s in tl.skipped_panels
              if s["reason"] == "no_text_no_audio_silent_beat"}
    assert filler
    budget = cfg.min_silent * cfg.max_pan_px_per_sec   # px readable per hold
    by_id = {e.panel_id: e for e in tl.entries}
    for pid in filler:
        assert by_id[pid].duration_seconds == pytest.approx(
            cfg.min_silent, abs=0.05)
        assert by_id[pid].pan.travel_px <= budget
        assert (by_id[pid].motion or {}).get("travel_px", 0) <= budget


def test_min_silent_zero_drops_unnarrated_panels(session):
    """min_silent=0 is the speech-driven contract: panels the chapter script
    skipped leave the timeline instead of becoming 1s filler beats. Measured on
    a real chapter whose script spoke 11 of 68 panels: the beats added 57s of
    voiceless runtime (worst consecutive stretch 12s) and forced a whole camera
    move into 1s, which reads as "the narration stops" + "the video speeds up".
    """
    d = session
    art = CutArtifact.model_validate_json(
        (d / "panels.json").read_text("utf-8"))
    cfg = rv.VideoConfig(tts="none", min_silent=0.0)
    nar = rv.build_narration(art, cfg, panels_hash="h", work_dir=d)
    aud = rv.synthesize_audio(nar, d / "audio", cfg)
    tl = rv.build_timeline(art, d, nar, aud, d / "audio", cfg,
                           panels_hash="h")
    assert [e.panel_id for e in tl.entries] == \
        ["panel_001", "panel_002", "panel_003"]
    reasons = {s["panel_id"]: s["reason"] for s in tl.skipped_panels}
    assert reasons["panel_004"] == "no_text_no_audio_dead_air_drop"
    assert reasons["panel_005"] == "no_text_no_audio_dead_air_drop"
    # the survivors are exactly the panels that carry a spoken line (with
    # tts="none" even those have no audio clip, so text is the contract)
    spoken = {e.panel_id for e in nar.entries if e.text.strip()}
    assert {e.panel_id for e in tl.entries} == spoken


def test_panel_with_audio_but_no_text_stays(session, tmp_path):
    """A panel holding a still-relevant audio clip (e.g. TTS cached from an
    edited script line) keeps its AUDIO-driven duration, not the silent-beat
    floor — audio presence wins over the short filler hold."""
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
    by_id = {e.panel_id: e for e in tl.entries}
    # spoken beat: audio-driven (2.5s), NOT the min_silent filler hold
    assert by_id["panel_004"].duration_seconds == pytest.approx(2.5, abs=0.1)
    assert "panel_005" in ids        # silent beat
    assert by_id["panel_005"].duration_seconds == pytest.approx(
        cfg.min_silent, abs=0.05)


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
        # Legacy floor mechanics: window pinned off to isolate them (the
        # 5-7s backstop is covered in test_speech_window.py).
        cfg = rv.VideoConfig(min_display_seconds=2.0, speech_window=False)
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
        # Window pinned off to isolate the legacy bound.
        cfg = rv.VideoConfig(min_display_seconds=2.0, pan_fit_speech=True,
                             speech_window=False)
        dur = rv.display_seconds(audio_seconds=5.0, words=3,
                                 travel_px=45000, cfg=cfg,
                                 panel_class="calm")
        assert dur >= 45000 / (cfg.max_pan_px_per_sec
                               * cfg.pan_fit_speech_speedup)

    @pytest.mark.parametrize("travel", [
        # durations are lower bounds: rounding must never truncate a pan
        # floor below the exact readability bound it is derived from.
        45000,   # the case that regressed: 45000/1350 = 33.3333 -> 33.333
        4500,    # floor inside the speech window -> bounded speed-up applies
        900,     # exact 2.0s at base speed, no speed-up needed
    ])
    def test_rounding_never_truncates_pan_floor(self, travel):
        # Window pinned off: this test guards the legacy floor arithmetic
        # against millisecond rounding, orthogonal to the speech backstop.
        cfg = rv.VideoConfig(min_display_seconds=2.0, pan_fit_speech=True,
                             speech_window=False)
        dur = rv.display_seconds(audio_seconds=0.5, words=0,
                                 travel_px=travel, cfg=cfg,
                                 panel_class="calm")
        # The floor is travel/base_speed, but pan_fit_speech may raise the
        # speed (bounded by pan_fit_speech_speedup) to fit the speech window.
        # Whichever floor actually applies, the returned duration must not
        # be truncated below it by the millisecond rounding.
        base = 0.5 + cfg.gap_seconds
        base_speed_floor = travel / cfg.max_pan_px_per_sec
        bounded_floor = travel / (cfg.max_pan_px_per_sec
                                  * cfg.pan_fit_speech_speedup)
        # pan_fit_speech only ever raises speed when the floor beats speech,
        # so the operative floor is the max of bounded floor and speech/base
        operative = max(bounded_floor if base_speed_floor > base else 0.0,
                        base, cfg.min_display_seconds)
        assert dur >= operative - 1e-9, \
            f"{dur} truncated below operative floor {operative}"

    def test_exact_millisecond_durations_unchanged(self):
        """Values already on a millisecond boundary must not be bumped."""
        assert rv._round_ms_up(2.0) == 2.0
        assert rv._round_ms_up(0.85) == 0.85
        assert rv._round_ms_up(12.0) == 12.0
        # and must never fall below the input
        assert rv._round_ms_up(45000 / 1350) >= 45000 / 1350

    def test_pan_fit_speech_flag_off_by_default(self):
        # The pan_fit_speech flag defaults off (legacy pan-floor-wins pacing
        # when the speech window is also off; the window backstop itself is
        # covered in test_speech_window.py).
        cfg = rv.VideoConfig(min_display_seconds=2.0, speech_window=False)
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
    # duration even though narration is short. This exercises the
    # geometry-driven automation pacing (class multipliers), so opt out of
    # the default strict reference-motion cycle which forces calm pacing.
    cfg = rv.VideoConfig(tts="none", motion_preset="none")
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
