"""Tests for the sub-5s speech window (short recap beats, in-sync panels)."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import recap_video as rv
from adapters.schemas import AudioArtifact, Meta
from guided_cutter import CutArtifact, CutPanel


def _panel(i: int, y0: int, y1: int, narration: str) -> CutPanel:
    return CutPanel(id=f"{i:03d}", panel_index=i, y_start=y0, y_end=y1,
                    narration=narration, dialogue="",
                    panel_type="single", confidence=0.9,
                    image_file=f"panel_{i:03d}.png")


def _meta() -> Meta:
    return Meta(schema_version=1, generator="t", config_hash="c",
                input_hashes={})


LONG = ("Jinwoo wakes in the dark hospital room. His heart pounds as the "
        "shadows gather around the bed. He remembers the double dungeon and "
        "the statues that moved when no one watched them closely at all.")


# ------------------------------------------------------------------ trimming
def test_short_text_passes_through_untouched():
    cfg = rv.VideoConfig()
    text = "He runs down the corridor."
    assert rv.fit_text_to_speech_window(text, cfg) == text


def test_long_text_trims_to_whole_sentences():
    cfg = rv.VideoConfig()
    short = rv.fit_text_to_speech_window(LONG, cfg)
    # first sentence (8 words) fits the ~9-word target; the rest is cut
    assert short == "Jinwoo wakes in the dark hospital room."
    assert rv._word_count(short) <= int(cfg.speech_max_seconds
                                         * cfg.speech_wpm / 60)


def test_single_huge_sentence_is_hard_cut_with_punctuation():
    cfg = rv.VideoConfig()
    text = " ".join(f"word{i}" for i in range(60))
    short = rv.fit_text_to_speech_window(text, cfg)
    assert rv._word_count(short) == int(cfg.speech_max_seconds
                                        * cfg.speech_wpm / 60)
    assert short[-1] in ".!?…\"'”’"


def test_normal_long_sentence_is_never_chopped_mid_clause():
    # Regression: a single ~19-word recap line used to be hard-cut to the
    # 10-word budget, speaking "...even the martial clans are." and sounding
    # like the narration stopped. It fits inside the whole-panel ceiling, so
    # it must pass through complete and let the panel last to the voice end.
    cfg = rv.VideoConfig()
    text = ("Meanwhile, nearby fighters gossip that even the martial clans "
            "are coming, probably just to pick over Subaru's scraps.")
    assert rv.fit_text_to_speech_window(text, cfg) == text


def test_line_with_in_quote_question_mark_survives_whole():
    # The sentence splitter breaks on the '?' inside the quote; the whole line
    # still fits the panel ceiling so it must stay intact rather than trim to
    # the dangling fragment 'A familiar ... shake hands — "Hold-up?'.
    cfg = rv.VideoConfig()
    text = ('A familiar blonde man calls out and they shake hands — '
            '"Hold-up? You\'re literally the first ones here."')
    assert rv.fit_text_to_speech_window(text, cfg) == text


def test_window_off_disables_trimming():
    cfg = rv.VideoConfig(speech_window=False)
    assert rv.fit_text_to_speech_window(LONG, cfg) == LONG


def test_empty_and_non_lexical_are_safe():
    cfg = rv.VideoConfig()
    assert rv.fit_text_to_speech_window("", cfg) == ""
    assert rv.fit_text_to_speech_window("...", cfg) == "..."


def test_estimate_is_conservative():
    cfg = rv.VideoConfig()
    # 10 budgeted words must estimate within the window
    text = " ".join(["word"] * 10)
    assert rv.estimate_speech_seconds(text, cfg) <= cfg.speech_max_seconds


# --------------------------------------------------------------- narration
def test_build_narration_trims_long_panel_text(tmp_path: Path):
    panels = [_panel(1, 0, 800, LONG)]
    Image.new("RGB", (800, 800), "white").save(tmp_path / "panel_001.png")
    art = CutArtifact(source="s.png", width=800, height=800,
                      plan_hash="x", config={}, panels=panels)
    nar = rv.build_narration(art, rv.VideoConfig(), panels_hash="x")
    assert nar.entries[0].text == "Jinwoo wakes in the dark hospital room."


def test_build_narration_keeps_short_text(tmp_path: Path):
    panels = [_panel(1, 0, 800, "He runs.")]
    art = CutArtifact(source="s.png", width=800, height=800,
                      plan_hash="x", config={}, panels=panels)
    nar = rv.build_narration(art, rv.VideoConfig(), panels_hash="x")
    assert nar.entries[0].text == "He runs."


# --------------------------------------------------------------- durations
def test_spoken_cap_backstops_but_never_cuts_speech():
    cfg = rv.VideoConfig()
    # measured speech is sacred: even absurdly long audio (stale cache, odd
    # voice) is never truncated — the sub-5s guarantee comes from trimming the
    # text BEFORE synthesis, not from cutting rendered speech
    assert rv.display_seconds(audio_seconds=20.0, words=3, travel_px=0,
                              cfg=cfg) == pytest.approx(20.0 + cfg.gap_seconds)
    # normal short narration untouched
    assert rv.display_seconds(audio_seconds=5.0, words=3, travel_px=0,
                              cfg=cfg) == pytest.approx(5.0 + cfg.gap_seconds)
    # window off restores the legacy never-capped behaviour
    off = rv.VideoConfig(speech_window=False)
    assert rv.display_seconds(audio_seconds=20.0, words=3, travel_px=0,
                              cfg=off) == pytest.approx(20.0 + off.gap_seconds)


def test_reveal_inflation_is_capped_but_audio_survives():
    cfg = rv.VideoConfig()
    dur = rv.display_seconds(audio_seconds=6.0, words=3, travel_px=0,
                             cfg=cfg, panel_class="reveal")
    # 6.0 * 1.35 = 8.1 would inflate past the narration; the cap backstops it
    # at audio + gap so the reveal hold never outlasts the voice
    assert dur == pytest.approx(6.0 + cfg.gap_seconds)
    assert dur >= 6.0 + cfg.gap_seconds


def test_silent_panels_capped_for_uniform_rhythm():
    cfg = rv.VideoConfig(silent_wpm=120)
    dur = rv.display_seconds(audio_seconds=None, words=120, travel_px=0,
                             cfg=cfg)
    # silent read (60s) is backstopped at the speech cap (speech_max + gap)
    assert dur == pytest.approx(cfg.speech_max_seconds + cfg.gap_seconds)


def test_pan_padding_capped_so_panels_cut_in_sync():
    cfg = rv.VideoConfig()
    dur = rv.display_seconds(audio_seconds=1.0, words=3, travel_px=4500,
                             cfg=cfg)
    # legacy floor would hold 10s of silence; the window cuts the padding
    # (the renderer plays the full travel inside the window via t/dur)
    assert dur == pytest.approx(cfg.speech_max_seconds + cfg.gap_seconds)


# ---------------------------------------------------------------- timeline
def test_timeline_panels_stay_in_window(tmp_path: Path):
    panels = [_panel(1, 0, 800, LONG),
              _panel(2, 800, 1600, "He runs down the endless corridor.")]
    for p in panels:
        Image.new("RGB", (800, 800), "white").save(tmp_path / p.image_file)
    art = CutArtifact(source="s.png", width=800, height=1600,
                      plan_hash="x", config={}, panels=panels)
    cfg = rv.VideoConfig(tts="none", motion_preset="reference")
    nar = rv.build_narration(art, cfg, panels_hash="x")
    aud = AudioArtifact(meta=_meta(), voice="none", entries=[])
    tl = rv.build_timeline(art, tmp_path, nar, aud, tmp_path, cfg,
                           panels_hash="x")
    assert len(tl.entries) == 2
    t = 0.0
    for e in tl.entries:
        assert e.duration_seconds <= cfg.speech_max_seconds + cfg.gap_seconds + 1e-6
        assert abs(e.start_seconds - round(t, 3)) < 1e-6  # contiguous
        t += e.duration_seconds


# ------------------------------------------------------------------- cache
def test_speech_fields_participate_in_hashes():
    assert (rv.VideoConfig().hash()
            != rv.VideoConfig(speech_max_seconds=10.0).hash())
    # narration text depends on the window, so the TTS/narration key must
    # change too (one re-synth, then stable)
    assert (rv.VideoConfig().hash_essentials()
            != rv.VideoConfig(speech_window=False).hash_essentials())
